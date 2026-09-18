"""
finetune.py
Fine-tunes a model using Unsloth + TRL SFTTrainer on the multilingual banking dataset.

Recommended base models (pick one):
  - google/gemma-3-12b (12B, excellent multilingual: English, Amharic, Afaan Oromo, Tigrigna, Somali)
  - unsloth/llama-3-8b-Instruct-bnb-4bit (8B, best balance for multilingual banking load in 4-bit)
  - unsloth/mistral-7b-instruct-v0.3 (7B, strong multilingual alternative)

Requirements (run once):
  pip install unsloth transformers datasets trl peft accelerate bitsandbytes pandas openpyxl
"""

try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None

import argparse
import importlib.util
import inspect
import json
import os
import subprocess
import torch
from collections import Counter
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── Config ────────────────────────────────────────────────────────────────────
# Default base model for fine-tuning.
MODEL_NAME   = os.getenv("MODEL_NAME", "google/gemma-4-12b-it")
OUTPUT_DIR   = os.getenv("OUTPUT_DIR", "./banking-model-gemma4-12b")
FINAL_DATASET_DIR = Path(__file__).resolve().parent / "Final"

DEFAULT_DATASET_FILES = [
    FINAL_DATASET_DIR / "cbe_chatbot_knowledge_base.json",
    FINAL_DATASET_DIR / "cbe_merged_knowledge_training.jsonl",
    FINAL_DATASET_DIR / "final_bank_ready_trainable_enriched.jsonl",
]

DATASET_FILES = [Path(path) for path in os.getenv("DATASET_FILE", "").split(os.pathsep) if path]
if not DATASET_FILES:
    DATASET_FILES = DEFAULT_DATASET_FILES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 🔥 NATIVE NVIDIA A100 PRODUCTION CONFIGURATION FOR GEMMA 4 12B
# =====================================================================

# 1. Deterministic Environment Control
SEED          = 42          # The industry standard seed for replicability

# 2. Context & Precision (No Quantization Losses)
MAX_SEQ_LEN   = 1024        # Shorter packed sequences improve throughput on an 80 GB A100.
LOAD_IN_4BIT  = False       # Pure 16-bit BF16 training preserves full intelligence
FLASH_ATTN_AVAILABLE = importlib.util.find_spec("flash_attn") is not None
ATTN_IMPLEMENTATION = "flash_attention_2" if FLASH_ATTN_AVAILABLE else None
PACKING       = FLASH_ATTN_AVAILABLE  # Packing requires an attention backend that supports flattened batches.
GRADIENT_CHECKPOINTING = True   # Recompute activations to keep the 80 GB run memory-safe.
MAX_STEPS    = 10000            # 24-hour target; adjust with --max-steps after measuring step time.

# 3. High-Capacity LoRA Architecture
LORA_R        = 64          # Deep capacity rank to learn precise citation tracking
LORA_ALPHA    = 128         # Mathematically optimal scale multiplier (2 × R)
LORA_DROPOUT  = 0.0         # 0.0 maximizes training efficiency on Ampere chips

# 4. Optimized Compute & Execution Loop
EPOCHS        = 2           # Sweet spot to prevent text repeating or overfitting
BATCH_SIZE    = 8           # Uses the A100's available VRAM for parallel examples.
GRAD_ACCUM    = 2           # Keeps the effective batch size at 16 with faster updates.
LEARNING_RATE = 2e-4        # Standard optimal convergence speed for LoRA
WARMUP_RATIO  = 0.03        # 3% gradual warm-up phase prevents early weight distortion

parser = argparse.ArgumentParser(description="Fine-tune a multilingual banking assistant")
parser.add_argument("--model", default=None, help="Model name or path to fine-tune")
parser.add_argument("--output-dir", default=None, help="Output directory for the fine-tuned model")
parser.add_argument(
    "--dataset",
    nargs="+",
    default=None,
    help="One or more JSON/JSONL dataset file paths",
)
parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
parser.add_argument("--max-seq-length", type=int, default=None, help="Maximum tokens per example")
parser.add_argument("--batch-size", type=int, default=None, help="Per-device training batch size")
parser.add_argument("--gradient-accumulation", type=int, default=None, help="Gradient accumulation steps")
parser.add_argument("--max-steps", type=int, default=None, help="Maximum optimizer steps; overrides epoch count")
parser.add_argument(
    "--gradient-checkpointing",
    action="store_true",
    help="Reduce memory use at the cost of training speed",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
args = parser.parse_args()

if args.model:
    MODEL_NAME = args.model
if args.output_dir:
    OUTPUT_DIR = args.output_dir
if args.dataset:
    DATASET_FILES = [Path(path) for path in args.dataset]
if args.epochs is not None:
    EPOCHS = args.epochs
if args.max_seq_length is not None:
    MAX_SEQ_LEN = args.max_seq_length
if args.batch_size is not None:
    BATCH_SIZE = args.batch_size
if args.gradient_accumulation is not None:
    GRAD_ACCUM = args.gradient_accumulation
if args.max_steps is not None:
    MAX_STEPS = args.max_steps

# Reproducibility: set global random seeds (from CLI)
SEED = int(args.seed)
set_seed(SEED)
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

# Eval / save step params from CLI
EVAL_STEPS = int(args.eval_steps)
SAVE_STEPS = int(args.save_steps)

if not torch.cuda.is_available():
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False
        )
        driver_status = nvidia_smi.stdout.strip() or nvidia_smi.stderr.strip() or "not available"
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        driver_status = f"nvidia-smi unavailable: {exc}"
    raise RuntimeError(
        "CUDA is unavailable. Unsloth fine-tuning requires a working NVIDIA GPU driver.\n"
        f"PyTorch CUDA build: {torch.version.cuda!r}\n"
        f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')!r}\n"
        f"nvidia-smi: {driver_status}\n"
        "Check the NVIDIA driver and CUDA environment before retrying."
    )


def load_model_and_tokenizer(model_name: str):
    if model_name.startswith("unsloth/"):
        if FastLanguageModel is None:
            raise RuntimeError(
                "Unsloth is not installed. Install it with 'pip install unsloth' or use a Hugging Face model name instead."
            )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = model_name,
            max_seq_length = MAX_SEQ_LEN,
            dtype          = None,
            load_in_4bit   = LOAD_IN_4BIT,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r              = LORA_R,
            lora_alpha     = LORA_ALPHA,
            lora_dropout   = LORA_DROPOUT,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            bias           = "none",
            use_gradient_checkpointing = "unsloth" if GRADIENT_CHECKPOINTING else False,
            random_state   = 42,
        )
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_kwargs = {
        "device_map": "auto",
        "dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "trust_remote_code": True,
    }
    if ATTN_IMPLEMENTATION:
        model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
    if LOAD_IN_4BIT:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["dtype"],
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if LOAD_IN_4BIT:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    if hasattr(model, "config"):
        model.config.use_cache = False

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer

print(f"Loading base model: {MODEL_NAME}")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

# Enable gradient checkpointing only when requested; disabling it improves A100 throughput.
try:
    if GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
except Exception:
    pass

print(model.print_trainable_parameters())

# ── Load dataset ──────────────────────────────────────────────────────────────
def resolve_dataset_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    for candidate in (Path(SCRIPT_DIR) / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return path


def knowledge_base_records(path: Path) -> list[dict]:
    """Convert the structured CBE knowledge base into fine-tuning examples."""
    knowledge = json.loads(path.read_text(encoding="utf-8"))
    records = []

    def add(question: str, answer: str, product: str = "Bank Information"):
        question, answer = str(question).strip(), str(answer).strip()
        if not question or not answer:
            return
        records.append({
            "instruction": question,
            "input": "",
            "output": answer,
            "language": "English",
            "domain": "banking",
            "source": path.name,
            "product": product,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "text": (
                f"<start_of_turn>user\n{question}<end_of_turn>\n"
                f"<start_of_turn>model\n{answer}<end_of_turn>"
            ),
        })

    bank = knowledge.get("bank", {})
    for field, answer in bank.items():
        if isinstance(answer, str):
            label = field.replace("_", " ")
            add(f"What is CBE's {label}?", answer)

    important = knowledge.get("important_data_for_chatbot", {})
    for category, values in important.items():
        if isinstance(values, list) and values:
            label = category.replace("_", " ")
            add(f"What {label} does CBE offer?", ", ".join(map(str, values)))

    for page in knowledge.get("pages", []):
        if not isinstance(page, dict):
            continue
        page_name = str(page.get("page_name", "CBE banking")).replace("_", " ")
        labels = page.get("labels", [])
        links = page.get("links", [])
        details = []
        if labels:
            details.append("Topics: " + ", ".join(map(str, labels)))
        if links:
            details.append("Official links: " + ", ".join(map(str, links)))
        if details:
            add(f"What information is available on CBE's {page_name} page?", " ".join(details))

    topics = knowledge.get("recommended_chatbot_topics", [])
    if topics:
        add("What topics can the CBE chatbot help with?", ", ".join(map(str, topics)))
    return records


def load_dataset_records(paths: list[Path]) -> list[dict]:
    records = []
    for configured_path in paths:
        path = resolve_dataset_path(configured_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if path.suffix.lower() == ".json":
            loaded = knowledge_base_records(path)
        else:
            loaded = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        loaded.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        print(f"Loaded {len(loaded)} records from {path}")
        records.extend(loaded)
    return records


def normalize_dataset_records(raw_records):
    normalized = []
    for index, record in enumerate(raw_records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not a JSON object: {record}")

        if record.get("text"):
            normalized.append(record)
            continue

        messages = record.get("messages")
        if isinstance(messages, list) and messages:
            rendered = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                rendered.append(f"### {role.title()}: {content}")
            record = dict(record)
            record["text"] = "\n".join(rendered)
            normalized.append(record)
            continue

        instruction = str(record.get("instruction", "")).strip()
        output = str(record.get("output", "")).strip()
        if instruction and output:
            input_text = str(record.get("input", "")).strip()
            user_text = f"{instruction}\n{input_text}" if input_text else instruction
            record = dict(record)
            record["text"] = (
                f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
                f"<start_of_turn>model\n{output}<end_of_turn>"
            )
            normalized.append(record)
            continue

        raise ValueError(
            f"Record {index} has no usable text, messages, or instruction/output fields: {record}"
        )

    return normalized


records = normalize_dataset_records(load_dataset_records(DATASET_FILES))
dataset = Dataset.from_list(records)
lang_counts = Counter(record.get("language", "unknown") for record in records)
print("Language counts:", dict(lang_counts))
print(f"Dataset loaded: {len(dataset)} examples")
print("Sample:\n", dataset[0]["text"], "\n")

# ── Training ──────────────────────────────────────────────────────────────────
def build_sft_config() -> SFTConfig:
    """Build SFTConfig across compatible TRL versions."""
    parameters = inspect.signature(SFTConfig).parameters
    kwargs = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": EPOCHS,
        "max_steps": MAX_STEPS,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "cosine",
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": 10,
        "load_best_model_at_end": False,
        "save_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "save_total_limit": 3,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "optim": "adamw_torch_fused",
        "weight_decay": 0.01,
        "dataset_text_field": "text",
        "loss_type": "nll",
        "report_to": "none",
    }
    if "warmup_ratio" in parameters:
        kwargs["warmup_ratio"] = WARMUP_RATIO
    elif "warmup_steps" in parameters:
        kwargs["warmup_steps"] = 0

    evaluation_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    if evaluation_key in parameters:
        kwargs[evaluation_key] = "no"

    sequence_key = "max_seq_length" if "max_seq_length" in parameters else "max_length"
    if sequence_key in parameters:
        kwargs[sequence_key] = MAX_SEQ_LEN
    if "packing" in parameters:
        kwargs["packing"] = PACKING
    if "padding_free" in parameters:
        kwargs["padding_free"] = bool(PACKING and ATTN_IMPLEMENTATION)
    if "dataloader_num_workers" in parameters:
        kwargs["dataloader_num_workers"] = 8
    if "dataloader_pin_memory" in parameters:
        kwargs["dataloader_pin_memory"] = True
    if "dataloader_persistent_workers" in parameters:
        kwargs["dataloader_persistent_workers"] = True
    if "dataloader_prefetch_factor" in parameters:
        kwargs["dataloader_prefetch_factor"] = 4

    return SFTConfig(**{key: value for key, value in kwargs.items() if key in parameters})


def build_sft_trainer():
    """Build SFTTrainer across TRL versions using tokenizer or processing_class."""
    parameters = inspect.signature(SFTTrainer).parameters
    kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": build_sft_config(),
    }
    tokenizer_key = "processing_class" if "processing_class" in parameters else "tokenizer"
    kwargs[tokenizer_key] = tokenizer
    return SFTTrainer(**kwargs)


trainer = build_sft_trainer()

print("Starting fine-tuning...")
trainer.train()

# ── Save fine-tuned model ─────────────────────────────────────────────────────
print(f"\nSaving model to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Fine-tuning complete!")