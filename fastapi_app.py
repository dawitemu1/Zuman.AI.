"""
FastAPI backend for the local multilingual banking assistant.

Endpoints:
    POST /chat              — local fine-tuned CBE chat
  POST /chat/stream       — LLM chat with Server-Sent Events streaming
  POST /speech-to-text   — Transcribe an uploaded audio file
  POST /text-to-speech   — Synthesise speech from text (returns audio)
  POST /translate        — Translate text between en / am / or
  POST /speech-to-speech — Full pipeline: audio -> translated audio
  GET  /health           — Liveness check
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import threading
import time
import urllib3
from typing import AsyncGenerator, Literal

import requests
import inference as local_inference

try:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )
except Exception:  # pragma: no cover - optional dependency fallback
    torch = None
    PeftModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TextIteratorStreamer = None

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    import redis
except ImportError:  # pragma: no cover - optional infrastructure dependency
    redis = None

import logging

logger = logging.getLogger("local-banking-assistant")
logging.basicConfig(level=logging.INFO)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_SESSION_TTL = int(os.getenv("REDIS_SESSION_TTL", "1800"))
REDIS_FAQ_TTL = int(os.getenv("REDIS_FAQ_TTL", "86400"))
# Redis is intentionally disabled until stale cached question echoes are purged.
REDIS_ENABLED = False
_REDIS_CLIENT = None


def _get_redis_client():
    """Return a lazy Redis client, or None when Redis is unavailable."""
    global _REDIS_CLIENT
    if not REDIS_ENABLED or redis is None:
        return None
    if _REDIS_CLIENT is None:
        try:
            _REDIS_CLIENT = redis.Redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            _REDIS_CLIENT.ping()
        except Exception as exc:
            logger.warning("Redis unavailable; continuing without memory/cache: %s", exc)
            _REDIS_CLIENT = None
    return _REDIS_CLIENT


def _session_key(session_id: str) -> str:
    return f"chat:session:{session_id}:messages"


def _load_session_messages(session_id: str) -> list[dict[str, str]]:
    client = _get_redis_client()
    if client is None:
        return []
    try:
        return [json.loads(item) for item in client.lrange(_session_key(session_id), 0, -1)]
    except Exception as exc:
        logger.warning("Could not load Redis session %s: %s", session_id, exc)
        return []


def _append_session_messages(session_id: str, messages: list[dict[str, str]]) -> None:
    client = _get_redis_client()
    if client is None or not messages:
        return
    try:
        key = _session_key(session_id)
        with client.pipeline() as pipe:
            for message in messages:
                pipe.rpush(key, json.dumps(message, ensure_ascii=False))
            pipe.ltrim(key, -40, -1)
            pipe.expire(key, REDIS_SESSION_TTL)
            pipe.execute()
    except Exception as exc:
        logger.warning("Could not save Redis session %s: %s", session_id, exc)


def _faq_cache_key(query: str, language: str) -> str:
    normalized = f"{language}:{query.strip().lower()}".encode("utf-8")
    digest = hashlib.sha256(normalized).hexdigest()
    return f"chat:faq:{digest}"


def _get_cached_faq(query: str, language: str) -> str | None:
    client = _get_redis_client()
    if client is None:
        return None
    try:
        return client.get(_faq_cache_key(query, language))
    except Exception as exc:
        logger.warning("Could not read FAQ cache: %s", exc)
        return None


def _cache_faq(query: str, language: str, reply: str) -> None:
    client = _get_redis_client()
    if client is None:
        return
    try:
        client.setex(_faq_cache_key(query, language), REDIS_FAQ_TTL, reply)
    except Exception as exc:
        logger.warning("Could not write FAQ cache: %s", exc)

# DuckDuckGo search helper (optional)
try:
    from duckduckgo_search import ddg
except Exception:
    ddg = None

try:
    from langdetect import detect as _detect_lang
except Exception:
    _detect_lang = None


# Map language codes to human-readable names for generation instructions
_LANG_CODE_TO_NAME = {
    "en": "English",
    "am": "Amharic",
    "or": "Oromo",
    "ti": "Tigrinya",
    "so": "Somali",
}

CBE_SYSTEM_INSTRUCTION = """
You are the official Multilingual AI Banking Assistant for the Commercial Bank of Ethiopia (CBE) (የኢትዮጵያ ንግድ ባንክ).
Only answer questions related to CBE banking products, services, branches, accounts, payments, cards, loans, and customer support. For unrelated topics such as visas, immigration, travel, or government applications, briefly explain that you are a CBE banking assistant and redirect the customer to the relevant official authority. Do not provide generic step-by-step instructions for those topics.
Express financial values and transaction caps only in Ethiopian Birr: use Qarshii in Afaan Oromoo, ብር in Amharic, and ETB or Birr in English. Never use US Dollars or $.
A standard CBE savings account requires a minimum initial deposit of 50 ETB, and valid identification such as a renewed Kebele ID, Passport, or Digital ID is mandatory.
Use CBE Birr or CBE Mobile Banking App for mobile banking and CBE CyberBank for internet banking.
For failed transactions or ATM errors, instruct the customer to call CBE Customer Care at 951 or visit the nearest branch.
All operations are in Ethiopia; use localized terminology and answer in the user's language.
When providing CBE links, use only verified URLs from the official CBE website. Never complete, shorten, or invent a URL. The verified loan page is https://combanketh.et/products/loan and the verified CBE home page is https://combanketh.et/home. If no verified direct online application URL exists, say so clearly and provide the relevant official product page instead.
""".strip()


def _verified_link_reply(question: str, language: str) -> str | None:
    """Answer direct CBE link requests from verified URLs instead of model guesses."""
    if language != "en":
        return None
    normalized = re.sub(r"[^a-z0-9\u1200-\u137f]+", " ", question.casefold()).strip()
    asks_for_link = bool(re.search(r"\b(?:url|link|website|portal|online|apply|application|mapply)\b", normalized))
    asks_about_loan = bool(re.search(r"\b(?:loan|credit|borrow|mapply)\b", normalized))
    if asks_for_link and asks_about_loan:
        return (
            "For CBE loan information and application guidance, use the official loan page: "
            "https://combanketh.et/products/loan. A separate direct online loan-application URL "
            "could not be verified; follow the instructions on that page or call CBE Customer Care at 951."
        )
    if asks_for_link and re.search(r"\b(?:cbe|bank|official)\b", normalized):
        return "The official Commercial Bank of Ethiopia website is https://combanketh.et/home."
    return None


def _out_of_scope_reply(question: str, language: str) -> str | None:
    """Keep unrelated requests out of CBE banking generation."""
    normalized = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
    social_message = bool(re.fullmatch(
        r"(?:hi|hello|hey|selam|good morning|good afternoon|good evening|"
        r"thanks|thanks a lot|thank you|thank you very much|many thanks|galatoomi|you are helpful|who are you|what are you|"
        r"what can you do|how are you|ሰላም|እንዴት ነህ|አመሰግናለሁ|እናመሰግናለን)",
        normalized,
    ))
    cbe_context = bool(re.search(
        r"\b(?:cbe|commercial\s+bank|bank|banking|account|balance|deposit|"
        r"withdraw|transfer|send money|receive money|payment|transaction|card|"
        r"atm|loan|credit|birr|cyberbank|mobile banking|internet banking|branch|"
        r"savings|customer care|951|ethiopia|financial|ባንክ|ሂሳብ|ገንዘብ|ብድር|ካርድ|ክፍያ|ኤቲኤም|ቅርንጫፍ)\b",
        normalized,
    ))
    if social_message or cbe_context or not normalized:
        return None
    if language == "am":
        return "እኔ የኢትዮጵያ ንግድ ባንክ የባንክ ረዳት ነኝ። ስለ ኢትዮጵያ ንግድ ባንክ ሂሳቦች፣ ክፍያዎች፣ ካርዶች ወይም ብድሮች ልረዳዎ እችላለሁ። ለሌላ ጉዳይ ትክክለኛ መረጃ የሚመለከተውን ይፋዊ ተቋም ያነጋግሩ።"
    if re.search(r"\b(?:visa|immigration|immigrant|embassy|consulate|ds[ -]?160)\b", normalized):
        return "I’m the Commercial Bank of Ethiopia banking assistant, so I can’t provide immigration instructions. Please check the official U.S. visa website at https://travel.state.gov/content/travel/en/us-visas.html or contact the nearest U.S. Embassy for current guidance. I can help with CBE banking services."
    return "I’m the Commercial Bank of Ethiopia banking assistant. I can help with CBE accounts, payments, cards, loans, transfers, and other banking services. For this unrelated topic, please consult the responsible official institution or its official website for accurate information."


def _detect_user_language(messages: list[dict]) -> str:
    """Detect the language of the last user message; default to 'en'."""
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    if not last_user:
        return "en"
    return _detect_language_from_text(last_user)


def _ensure_reply_language(messages: list[dict]) -> tuple[list[dict], str]:
    """Return messages possibly prefixed with a system instruction to respond in user's language.

    Returns (messages_with_instruction, detected_code)
    """
    code = _detect_user_language(messages)
    lang_name = _LANG_CODE_TO_NAME.get(code, code)

    instr = {
        "role": "system",
        "content": f"{CBE_SYSTEM_INSTRUCTION}\n\nIdentify the user's language from the actual message before answering. The application detected {lang_name} as a hint only; if it conflicts with the user's words, follow the user's actual language and script. Respond in exactly the user's language.",
    }
    if any(CBE_SYSTEM_INSTRUCTION in m.get("content", "") for m in messages if m.get("role") == "system"):
        return messages, code
    return [instr] + messages, code


def _search_cbe(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for Commercial Bank of Ethiopia related pages and social posts.

    Returns a list of dicts with 'title','href','body','source'. Requires `duckduckgo_search`.
    """
    if ddg is None:
        return []

    known_cbe_links = [
        "https://combanketh.et/home",
        "https://web.facebook.com/combanketh",
        "https://www.linkedin.com/company/commercialbankofethiopia/posts/?feedView=all",
        "https://t.me/combankethofficial",
        "https://www.tiktok.com/@combankethiopia",
    ]

    results = []
    seen = set()

    # try official site first
    queries = [f"{query} site:combanketh.et", f"{query} Commercial Bank of Ethiopia", query]
    for q in queries:
        try:
            raw = ddg(q, max_results=max_results)
        except Exception:
            raw = None
        if not raw:
            continue
        for r in raw:
            href = (r.get("href") or r.get("url") or "").strip()
            if not href or href in seen:
                continue
            seen.add(href)
            results.append({
                "title": r.get("title"),
                "href": href,
                "body": r.get("body") or r.get("snippet") or "",
                "source": "web",
            })
            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    # include known social links if missing
    for link in known_cbe_links:
        if link not in seen:
            results.insert(0, {"title": link, "href": link, "body": "", "source": "official"})
            seen.add(link)
        if len(results) >= max_results:
            break

    return results[:max_results]

# Local-only configuration
VLLM_URL = None
VLLM_API_KEY = ""

# vLLM health flag and session (set at startup check)
VLLM_ONLINE = False
_VLLM_SESSION = None


def _check_vllm_connection(timeout: int = 5) -> bool:
    """Try a minimal OpenAI-compatible request to the configured vLLM URL.

    Sets `VLLM_ONLINE` and returns True when the proxy responds with HTTP 200.
    This is a lightweight probe used on startup to surface connectivity issues early.
    """
    global VLLM_ONLINE, _VLLM_SESSION
    VLLM_ONLINE = False
    _VLLM_SESSION = None

    if not VLLM_URL:
        logger.info("vLLM URL not configured (VLLM_URL unset)")
        return False

    probe_url = f"{VLLM_URL.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "openai_compatible": True,
    }

    try:
        s = requests.Session()
        s.verify = False
        resp = s.post(probe_url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            VLLM_ONLINE = True
            _VLLM_SESSION = s
            logger.info("vLLM proxy reachable at %s", probe_url)
            return True
        logger.warning("vLLM probe returned status %s", resp.status_code)
    except Exception as exc:
        logger.warning("vLLM probe failed: %s", exc)

    VLLM_ONLINE = False
    _VLLM_SESSION = None
    return False


def _check_bearer(auth_header: str | None):
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    token = parts[1]
    if token != VLLM_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# Suppress the InsecureRequestWarning we get from verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

Language = Literal["en", "am", "or", "ti", "so"]


def _remote_service_disabled(service: str) -> None:
    raise HTTPException(
        status_code=501,
        detail=f"{service} requires a remote gateway and is disabled in local-only mode.",
    )

LOCAL_CHAT_ENABLED = True
LOCAL_ADAPTER_PATH = os.path.abspath(local_inference.FINETUNED_MODEL)
LOCAL_BASE_MODEL = LOCAL_ADAPTER_PATH
LOCAL_MODEL = local_inference.model
LOCAL_TOKENIZER = local_inference.tokenizer
LOCAL_MODEL_ERROR = None if LOCAL_MODEL is not None and LOCAL_TOKENIZER is not None else "local fine-tuned model failed to load"
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_SEQ_LEN", "16384"))
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))


def _count_sentence_endings(text: str) -> int:
    """Count sentence punctuation while ignoring dots inside URLs."""
    url_pattern = re.compile(r"(?:https?://|www\.)[^\s<>()]+")
    masked = []
    last_end = 0
    for match in url_pattern.finditer(text):
        masked.append(text[last_end:match.start()])
        masked.append(" " * len(match.group(0)))
        last_end = match.end()
    masked.append(text[last_end:])
    return len(re.findall(r"[.!?።፧]", "".join(masked)))


class _SentenceLimitStop(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int, max_sentences: int = 8):
        self.decoder = getattr(tokenizer, "tokenizer", tokenizer)
        self.prompt_length = prompt_length
        self.max_sentences = max_sentences

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0, self.prompt_length:]
        generated_text = self.decoder.decode(generated_ids, skip_special_tokens=False)
        return _count_sentence_endings(generated_text) >= self.max_sentences


class _ChatTurnStop(StoppingCriteria):
    """Stop Gemma before it starts generating a second chat turn."""

    def __init__(self, tokenizer):
        decoder = getattr(tokenizer, "tokenizer", tokenizer)
        convert_tokens_to_ids = getattr(decoder, "convert_tokens_to_ids", None)
        self.stop_token_ids = set()
        if callable(convert_tokens_to_ids):
            for token in ("<end_of_turn>", "<start_of_turn>", "<turn|>"):
                token_id = convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id >= 0:
                    self.stop_token_ids.add(token_id)
        eot_token_id = getattr(decoder, "eot_token_id", None)
        if isinstance(eot_token_id, int):
            self.stop_token_ids.add(eot_token_id)

    def __call__(self, input_ids, scores, **kwargs):
        return bool(self.stop_token_ids and input_ids[0, -1].item() in self.stop_token_ids)


def _stopping_criteria(
    tokenizer,
    prompt_length: int,
    max_sentences: int,
) -> StoppingCriteriaList:
    return StoppingCriteriaList([
        _ChatTurnStop(tokenizer),
        _SentenceLimitStop(tokenizer, prompt_length, max_sentences),
    ])


def _dynamic_generation_limits(
    messages: list[dict[str, str]],
    requested_tokens: int,
) -> tuple[int, int]:
    """Scale token and sentence budgets from the user's language and intent."""
    user_text = next(
        (message.get("content", "") for message in reversed(messages) if message.get("role") == "user"),
        "",
    )
    is_ethiopic = any("\u1200" <= char <= "\u137F" for char in user_text)
    is_detailed = bool(re.search(
        r"\b(?:step|steps|how|open|apply|application|requirement|eligib\w*|criter\w*|qualif\w*|loan|account|procedure|process)\b|"
        r"(?:መስፈርት|ብቁ|እንዴት|ከፈት|ሂደት)",
        user_text,
        re.IGNORECASE,
    ))

    # Banking requirement lists are usually numbered and must finish completely.
    # Use a larger headroom for the requirement-style answers that naturally
    # produce several list items, while staying short enough for local streaming.
    if is_ethiopic:
        max_tokens = min(max(1, requested_tokens), 1200 if is_detailed else 300)
    else:
        max_tokens = min(max(1, requested_tokens), 1000 if is_detailed else 260)
    max_sentences = 18 if is_detailed else 5
    return max_tokens, max_sentences

# ---------------------------------------------------------------------------
# Shared HTTP session retained for compatibility with the local route helpers.
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.verify = False
    return session

_SESSION = _make_session()

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Local Banking Assistant API",
    description="Local chat, speech-to-text, and text-to-speech services",
    version="1.0.0",
)

# Allow all origins during development — tighten this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Translated-Text", "X-Source-Language", "X-Target-Language"],
)


# Startup checks: probe vLLM proxy so we fail fast when misconfigured
@app.on_event("startup")
def _startup_checks():
    ok = _check_vllm_connection()
    logger.info("vLLM proxy configured at %s — online=%s", VLLM_URL or "(unset)", ok)


@app.get("/vllm-health", summary="vLLM proxy health")
def vllm_health():
    """Return basic vLLM proxy status for diagnostics."""
    return {"vllm_url": VLLM_URL, "online": VLLM_ONLINE}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _gateway_json_headers() -> dict[str, str]:
    return {"Content-Type": "application/json", "Accept": "application/json"}


def _raise_upstream(response: requests.Response, action: str) -> dict:
    """Raise an HTTPException when the upstream call fails."""
    if response.status_code != 200:
        try:
            detail = response.json().get("message") or response.text[:300]
        except Exception:
            detail = response.text[:300]
        raise HTTPException(status_code=502, detail=f"{action} failed: {detail}")

    content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        # Raw binary (audio) — caller handles it
        return {"_raw": response.content, "_content_type": content_type}

    body = response.json()
    if body.get("success") is False:
        raise HTTPException(
            status_code=502,
            detail=f"{action} failed: {body.get('message', 'Unknown upstream error')}",
        )
    return body


def _extract_audio(body: dict, raw_response: requests.Response) -> bytes:
    """Pull audio bytes out of a gateway response (base64, URL, or raw)."""
    if "_raw" in body:
        return body["_raw"]

    data = body.get("data", body)
    for key in ("audio_base64", "audio", "speech", "output_audio"):
        if data.get(key):
            return base64.b64decode(data[key])

    for key in ("audio_url", "url"):
        if data.get(key):
            r = _SESSION.get(data[key], timeout=60)
            r.raise_for_status()
            return r.content

    if raw_response.content and "json" not in raw_response.headers.get("Content-Type", ""):
        return raw_response.content

    raise HTTPException(status_code=502, detail="Upstream did not return audio data.")


def _clean_generated_text(text: str) -> str:
    """Remove Gemma control markers and text after the first assistant turn."""
    text = re.sub(r"<NA>|<na>|<unk>|<pad>", "", text)
    marker_pattern = r"(?:<end_of_turn>|<start_of_turn>|<turn\|>|<\|turn\|>|(?:^|\n)\s*(?:model|assistant)\s*(?::|\n))"
    return re.split(marker_pattern, text, maxsplit=1)[0].strip()


def _load_local_chat_model() -> tuple[object | None, object | None]:
    """Return the model loaded by inference.py from the local fine-tuned directory."""
    global LOCAL_MODEL_ERROR

    if LOCAL_MODEL is not None and LOCAL_TOKENIZER is not None:
        return LOCAL_MODEL, LOCAL_TOKENIZER

    if not LOCAL_CHAT_ENABLED:
        LOCAL_MODEL_ERROR = "local chat model disabled"
        return None, None

    LOCAL_MODEL_ERROR = f"local fine-tuned model unavailable at {LOCAL_ADAPTER_PATH}"
    return None, None


def _generate_local_reply(messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
    """Generate a reply with the local PEFT model when available."""
    model, tokenizer = _load_local_chat_model()
    if model is None or tokenizer is None:
        raise RuntimeError(LOCAL_MODEL_ERROR or "local chat model unavailable")

    dynamic_max_tokens, max_sentences = _dynamic_generation_limits(messages, max_tokens)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    original_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        text=prompt,
        images=None,
        return_tensors="pt",
        truncation=True,
        max_length=max(1, MAX_CONTEXT_LENGTH - dynamic_max_tokens),
    )
    tokenizer.truncation_side = original_truncation_side
    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=dynamic_max_tokens,
            use_cache=True,
            temperature=temperature,
            do_sample=temperature > 0,
            stopping_criteria=_stopping_criteria(
                tokenizer,
                inputs["input_ids"].shape[1],
                max_sentences,
            ),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    start = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(output[0][start:], skip_special_tokens=True)
    return _clean_generated_text(generated)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness check")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# /chat  —  LLM
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        description="Conversation history. Include a system message if desired.",
    )
    model: str = Field(LOCAL_BASE_MODEL, description="Local fine-tuned model path")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=4096)
    session_id: str | None = Field(None, description="Optional Redis-backed conversation session")


class ChatResponse(BaseModel):
    reply: str
    model: str
    usage: dict | None = None


@app.post("/chat", response_model=ChatResponse, summary="LLM chat completion")
def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    last_user_message = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        "",
    )
    stored_messages = _load_session_messages(req.session_id) if req.session_id else []
    language = _detect_user_language(messages)
    out_of_scope_reply = _out_of_scope_reply(last_user_message, language)
    if out_of_scope_reply is not None:
        return ChatResponse(reply=out_of_scope_reply, model=LOCAL_BASE_MODEL, usage={"source": "scope-guard"})
    verified_reply = _verified_link_reply(last_user_message, language)
    if verified_reply is not None:
        if req.session_id:
            _append_session_messages(req.session_id, [
                {"role": "user", "content": last_user_message},
                {"role": "assistant", "content": verified_reply},
            ])
        return ChatResponse(reply=verified_reply, model=LOCAL_BASE_MODEL, usage={"source": "verified-link"})
    faq_cache_allowed = bool(last_user_message and not stored_messages and len(messages) <= 2)
    if faq_cache_allowed:
        cached_reply = _get_cached_faq(last_user_message, language)
        if cached_reply is not None:
            if req.session_id:
                _append_session_messages(req.session_id, [
                    {"role": "user", "content": last_user_message},
                    {"role": "assistant", "content": cached_reply},
                ])
            return ChatResponse(reply=cached_reply, model=LOCAL_BASE_MODEL, usage={"source": "redis_faq_cache"})

    if req.session_id and stored_messages and len(messages) <= 2:
        system_messages = [message for message in messages if message["role"] == "system"]
        messages = system_messages + stored_messages + [
            message for message in messages if message["role"] == "user"
        ]
    messages, _ = _ensure_reply_language(messages)

    try:
        reply = _generate_local_reply(messages, req.temperature, req.max_tokens)
        if not _reply_is_usable(last_user_message, reply, language):
            snippets = _search_cbe(last_user_message, max_results=5)
            if snippets:
                sources = "\n\n".join(
                    f"[{index}] {item.get('body') or item.get('title') or ''} ({item.get('href', '')})"
                    for index, item in enumerate(snippets, 1)
                )
                reply = f"[Web-sourced answer]\n{sources}"
                source = "web-search-fallback"
            else:
                source = "local"
        else:
            source = "local"
        if req.session_id:
            _append_session_messages(req.session_id, [
                {"role": "user", "content": last_user_message},
                {"role": "assistant", "content": reply},
            ])
        if faq_cache_allowed:
            _cache_faq(last_user_message, language, reply)
        return ChatResponse(reply=reply, model=LOCAL_BASE_MODEL, usage={"source": source})
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Local chat model unavailable: {exc}") from exc


@app.post("/v1/chat/completions", summary="OpenAI-compatible chat completions (vLLM proxy)")
def v1_chat_completions(payload: dict = Body(...), authorization: str | None = Header(None)):
    """Minimal OpenAI-compatible endpoint. Secured with a simple Bearer token.
    Behavior:
      - If `VLLM_URL` env var is set, forwards the request to that URL (assumes it
        speaks OpenAI-compatible API).
      - Otherwise, uses the local PEFT model (same behaviour as `/chat`).
    """
    _check_bearer(authorization)

    # If remote vLLM server configured, proxy transparently
    if VLLM_URL:
        try:
            r = _SESSION.post(f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=120)
            r.raise_for_status()
            return Response(content=r.content, media_type=r.headers.get("Content-Type", "application/json"))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Proxy to VLLM failed: {exc}")

    # Fallback to local model (reuse existing chat logic)
    messages = payload.get("messages") or []
    # Ensure messages include a system instruction to reply in user's detected language
    messages, user_lang = _ensure_reply_language(messages)
    last_user_message = next(
        (message.get("content", "") for message in reversed(messages) if message.get("role") == "user"),
        "",
    )
    out_of_scope_reply = _out_of_scope_reply(last_user_message, user_lang)
    if out_of_scope_reply is not None:
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": LOCAL_BASE_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": out_of_scope_reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(out_of_scope_reply.split()), "total_tokens": len(out_of_scope_reply.split())},
        }
    verified_reply = _verified_link_reply(last_user_message, user_lang)
    if verified_reply is not None:
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": LOCAL_BASE_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": verified_reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(verified_reply.split()), "total_tokens": len(verified_reply.split())},
        }
    # normalize to list[Message]-like dicts
    msgs = [{"role": m.get("role"), "content": m.get("content")} for m in messages]
    temperature = float(payload.get("temperature", 0.7))
    max_tokens = int(payload.get("max_tokens", 1024))

    try:
        reply = _generate_local_reply(msgs, temperature, max_tokens)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Local chat model unavailable: {exc}") from exc

    import time

    response = {
        "id": f"cmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": LOCAL_BASE_MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(reply.split()), "total_tokens": len(reply.split())},
    }
    return response

    # Legacy remote fallback retained below for reference only.
    if False:
        # Fall back to upstream gateway LLM if local model unavailable
        try:
            gw_payload = {"model": payload.get("model", "mela4.3"), "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "openai_compatible": True}
            r = _SESSION.post(GATEWAY_LLM_URL, headers=_gateway_json_headers(), json=gw_payload, timeout=120)
            body = _raise_upstream(r, "Gateway LLM (fallback)")
            data = body.get("data", body)
            reply = data["choices"][0]["message"]["content"]
            model = data.get("model", payload.get("model", "mela4.3"))
        except Exception as exc2:
            logger.info("Gateway fallback failed: %s — trying web search fallback", exc2)
            # Final fallback: DuckDuckGo web search summarization
            try:
                snippets = _search_cbe(" ".join(m.get("content", "") for m in messages), max_results=5)
            except Exception as web_exc:
                raise HTTPException(status_code=502, detail=f"Local, gateway, and web search all failed: {web_exc}") from web_exc

            if not snippets:
                raise HTTPException(status_code=502, detail=f"Local and gateway generation failed: {exc2}")

            # Build a short summary from snippets
            combined = "\n\n".join([s.get("body", "") or s.get("title", "") for s in snippets])
            summary = combined.strip()[:3000]

            # Try to detect user's language from last user message
            last_user_text = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user_text = m.get("content", "")
                    break

            user_lang = "en"
            if _detect_lang is not None and last_user_text:
                try:
                    user_lang = _detect_lang(last_user_text)
                except Exception:
                    user_lang = "en"

            # If the user's language is not English, attempt to translate the summary via gateway translate API
            translated = None
            if user_lang and user_lang != "en":
                try:
                    tr_payload = {"text": summary, "source": "auto", "target": user_lang}
                    tr_r = _SESSION.post(TRANSLATE_URL, headers=_gateway_json_headers(), json=tr_payload, timeout=30)
                    if tr_r.ok:
                        try:
                            tr_body = tr_r.json()
                            # Try common fields
                            translated = tr_body.get("translated_text") or tr_body.get("data", {}).get("translated") or tr_body.get("output")
                        except Exception:
                            translated = tr_r.text
                except Exception:
                    translated = None

            if translated:
                reply = f"[Web-sourced answer]\n{translated}"
            else:
                reply = f"[Web-sourced answer]\n{summary}"
            model = "web-search-fallback"
    else:
        model = LOCAL_BASE_MODEL

    import time

    response = {
        "id": f"cmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": LOCAL_BASE_MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(reply.split()), "total_tokens": len(reply.split())},
    }
    return response


# ---------------------------------------------------------------------------
# /chat/stream  —  True async SSE streaming with per-token delay
# ---------------------------------------------------------------------------


@app.post("/chat/stream", summary="LLM chat with SSE token streaming")
async def chat_stream(req: ChatRequest):
    """
    Fetches the full reply from the upstream LLM in a thread (so it doesn't
    block the event loop), then streams it back token-by-token with a small
    asyncio.sleep() between each token so the browser receives them
    individually instead of in one big batch.

    SSE event format:
      data: {"token": "..."}   — partial text
      data: {"done": true}     — stream finished
      data: {"error": "..."}   — upstream error
    """
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    last_user_message = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        "",
    )
    language = _detect_user_language(messages)
    out_of_scope_reply = _out_of_scope_reply(last_user_message, language)
    verified_reply = _verified_link_reply(last_user_message, language)
    async def generate() -> AsyncGenerator[str, None]:
        try:
            if out_of_scope_reply is not None:
                yield f"data: {json.dumps({'token': out_of_scope_reply})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return
            if verified_reply is not None:
                yield f"data: {json.dumps({'token': verified_reply})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            if LOCAL_CHAT_ENABLED:
                model, tokenizer = _load_local_chat_model()
                if model is None or tokenizer is None or TextIteratorStreamer is None:
                    raise RuntimeError(LOCAL_MODEL_ERROR or "local streaming model unavailable")

                dynamic_max_tokens, max_sentences = _dynamic_generation_limits(
                    messages,
                    req.max_tokens,
                )
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                original_truncation_side = getattr(tokenizer, "truncation_side", "right")
                tokenizer.truncation_side = "left"
                inputs = tokenizer(
                    text=prompt,
                    images=None,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max(1, MAX_CONTEXT_LENGTH - dynamic_max_tokens),
                )
                tokenizer.truncation_side = original_truncation_side
                device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
                inputs = {key: value.to(device) for key, value in inputs.items()}
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                generation_error = []

                def generate_local():
                    try:
                        with torch.inference_mode():
                            model.generate(
                                **inputs,
                                streamer=streamer,
                                max_new_tokens=dynamic_max_tokens,
                                use_cache=True,
                                do_sample=False,
                                stopping_criteria=_stopping_criteria(
                                    tokenizer,
                                    inputs["input_ids"].shape[1],
                                    max_sentences,
                                ),
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )
                    except Exception as exc:
                        generation_error.append(exc)
                        streamer.end()

                generation_thread = threading.Thread(target=generate_local, daemon=True)
                generation_thread.start()
                streamed_text = ""
                emitted_length = 0
                while True:
                    chunk = await asyncio.to_thread(lambda: next(streamer, None))
                    if chunk is None:
                        break
                    if chunk:
                        streamed_text += chunk
                        safe_text = local_inference._stream_safe_text(streamed_text)
                        delta = safe_text[emitted_length:]
                        if delta:
                            emitted_length = len(safe_text)
                            yield f"data: {json.dumps({'token': delta})}\n\n"
                if generation_error:
                    raise generation_error[0]
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            yield f"data: {json.dumps({'error': 'Local fine-tuned model is unavailable.'})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    )


# ---------------------------------------------------------------------------
# /speech-to-text
# ---------------------------------------------------------------------------


@app.post("/speech-to-text", summary="Transcribe an audio file")
async def speech_to_text(
    file: UploadFile = File(..., description="Audio file (.mp3 or .wav)"),
    language: Language = Form("en", description="Spoken language: en / am / or"),
):
    _remote_service_disabled("Speech-to-text")
    audio_bytes = await file.read()
    response = _SESSION.post(
        STT_URL,
        headers={"X-API-Key": GATEWAY_API_KEY},
        data={"language": language},
        files={"file": (file.filename, audio_bytes, file.content_type or "audio/mpeg")},
        timeout=120,
    )
    body = _raise_upstream(response, "Speech-to-text")
    data = body.get("data", body)
    text = data.get("transcribed_text") or data.get("text") or ""
    if not text.strip():
        raise HTTPException(status_code=502, detail="Speech-to-text returned empty transcript.")
    return {"text": text.strip(), "language": language}


# ---------------------------------------------------------------------------
# /text-to-speech
# ---------------------------------------------------------------------------


@app.post("/text-to-speech", summary="Synthesise speech (returns audio/mpeg)")
def text_to_speech(
    text: str = Form(..., description="Text to synthesise"),
    language: Language = Form("en", description="Target language: en / am / or"),
):
    """
    Try every known TTS endpoint/format until one works.
    The gateway at 196.189.247.236:8016 speaks plain HTTP so we never hit SSL errors here.
    """
    _remote_service_disabled("Text-to-speech")
    attempts = [
        # 1. form-data (multipart) — most common gateway expectation
        dict(
            label="form-data",
            url=TTS_URL,
            kwargs=dict(
                headers={"X-API-Key": GATEWAY_API_KEY},
                data={"text": text, "language": language},
                timeout=120,
            ),
        ),
        # 2. JSON body
        dict(
            label="json",
            url=TTS_URL,
            kwargs=dict(
                headers=_gateway_json_headers(),
                json={"text": text, "language": language},
                timeout=120,
            ),
        ),
        # 3. JSON with alternate field names some gateways expect
        dict(
            label="json-alt-fields",
            url=TTS_URL,
            kwargs=dict(
                headers=_gateway_json_headers(),
                json={"input": text, "voice_language": language, "language": language},
                timeout=120,
            ),
        ),
    ]

    last_error = "No TTS endpoint responded successfully."
    for attempt in attempts:
        label = attempt["label"]
        try:
            resp = _SESSION.post(attempt["url"], **attempt["kwargs"])
            logger.info(
                "TTS attempt [%s] -> status=%s content-type=%s body_preview=%.200s",
                label,
                resp.status_code,
                resp.headers.get("Content-Type", ""),
                resp.text,
            )
            if resp.status_code == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "json" in content_type:
                    body = resp.json()
                    if body.get("success") is False:
                        last_error = body.get("message", "TTS returned success=false")
                        logger.warning("TTS [%s] success=false: %s", label, last_error)
                        continue
                    audio = _extract_audio(body, resp)
                else:
                    audio = resp.content
                if audio:
                    logger.info("TTS [%s] succeeded, returning %d bytes", label, len(audio))
                    return Response(content=audio, media_type="audio/mpeg")
                last_error = "TTS returned empty audio."
                logger.warning("TTS [%s] returned empty audio", label)
            else:
                try:
                    err_body = resp.json()
                    last_error = err_body.get("message") or err_body.get("detail") or resp.text[:400]
                except Exception:
                    last_error = resp.text[:400]
                logger.warning("TTS [%s] HTTP %s: %s", label, resp.status_code, last_error)
        except Exception as exc:
            last_error = str(exc)
            logger.warning("TTS attempt [%s] exception: %s", label, exc)

    raise HTTPException(status_code=502, detail=f"Text-to-speech failed: {last_error}")


# ---------------------------------------------------------------------------
# /tts-debug  —  shows raw upstream response to help diagnose 502s
# ---------------------------------------------------------------------------


@app.post("/tts-debug", summary="Debug TTS: returns raw gateway response as JSON")
def tts_debug(
    text: str = Form("Hello world"),
    language: Language = Form("en"),
):
    """
    Hits the upstream TTS endpoint and returns the raw status, headers, and
    body so you can see exactly what the gateway is responding with.
    Useful when /text-to-speech keeps returning 502.
    """
    _remote_service_disabled("TTS debugging")
    results = []
    for label, kwargs in [
        ("form-data", dict(headers={"X-API-Key": GATEWAY_API_KEY}, data={"text": text, "language": language}, timeout=30)),
        ("json",      dict(headers=_gateway_json_headers(), json={"text": text, "language": language}, timeout=30)),
    ]:
        try:
            r = _SESSION.post(TTS_URL, **kwargs)
            ct = r.headers.get("Content-Type", "")
            try:
                body = r.json()
            except Exception:
                body = r.text[:500]
            results.append({
                "attempt": label,
                "status": r.status_code,
                "content_type": ct,
                "body": body,
                "audio_bytes": len(r.content) if r.status_code == 200 else 0,
            })
        except Exception as exc:
            results.append({"attempt": label, "error": str(exc)})
    return {"tts_url": TTS_URL, "results": results}


# ---------------------------------------------------------------------------
# /translate
# ---------------------------------------------------------------------------


class TranslateRequest(BaseModel):
    text: str
    source_language: Language
    target_language: Language


@app.post("/translate", summary="Translate text between en / am / or")
def translate(req: TranslateRequest):
    _remote_service_disabled("Translation")
    response = _SESSION.post(
        TRANSLATE_URL,
        headers=_gateway_json_headers(),
        json={
            "source_text": req.text,
            "source_language": req.source_language,
            "destination_language": req.target_language,
        },
        timeout=60,
    )
    body = _raise_upstream(response, "Translation")
    data = body.get("data", body)
    translated = data.get("translated_text") or data.get("text") or ""
    if not translated.strip():
        raise HTTPException(status_code=502, detail="Translation returned empty text.")
    return {
        "translated_text": translated.strip(),
        "source_language": req.source_language,
        "target_language": req.target_language,
    }


# ---------------------------------------------------------------------------
# /speech-to-speech
# ---------------------------------------------------------------------------

def _detect_language_from_text(text: str) -> str:
    """Detect the five supported languages, including mixed-script Oromo."""
    if not text or not text.strip():
        return "en"
    oromo = r"\b(?:afaan\s+oromoo|oromoo|herrega|baankii|haaraa|akkamitti|banuu|danda['’]?a|sanadoonni|waraqaalee|eenyummaa|fiduu|qabu|maal\s+fa['’]?i|galma|beekumsa|mirkanaa['’]?aa|miti|hir'isee|kennine|gochuun|ati|eenyu|maal|eessa|akkam|maaliif)\b"
    if re.search(oromo, text, re.IGNORECASE):
        return "or"
    if re.search(r"\b(?:ኣካውንት|ባንኪ|ከመይ|ክኸፍት|እኽእል|ጌረ|ሒሳብ|ኣበይ)\b", text):
        return "ti"
    if re.search(r"\b(?:የባንክ|ሂሳብ|እንዴት|እከፍታለሁ|እችላለሁ|የት)\b", text):
        return "am"
    if re.search(r"[\u1200-\u137F]", text):
        return "am"
    if re.search(r"\b(?:sidee|sideen|akoon|bangiga|lacag|xisaab|furtaa|furto|qabta)\b", text, re.IGNORECASE):
        return "so"
    if _detect_lang is not None:
        try:
            return _detect_lang(text)
        except Exception:
            pass
    return "en"


def _reply_is_usable(question: str, reply: str, language: str) -> bool:
    normalized_question = re.sub(r"\s+", " ", question).strip().casefold()
    normalized_reply = re.sub(r"\s+", " ", reply).strip().casefold()
    if not normalized_reply or normalized_reply == normalized_question:
        return False
    if language in {"am", "ti"}:
        return bool(re.search(r"[\u1200-\u137F]", reply))
    if language in {"or", "so"}:
        return _detect_language_from_text(reply) == language
    return _detect_language_from_text(reply) == "en"


def _detect_best_audio_language(audio_bytes: bytes, filename: str, mime: str) -> tuple[str, str]:
    """Try STT with candidate languages and choose the best transcript/language."""
    import re

    best_language = "en"
    best_transcript = ""
    best_score = -1
    candidates = ["am", "or", "en"]

    for lang in candidates:
        try:
            stt_resp = _SESSION.post(
                STT_URL,
                headers={"X-API-Key": GATEWAY_API_KEY},
                data={"language": lang},
                files={"file": (filename, audio_bytes, mime)},
                timeout=120,
            )
            stt_body = _raise_upstream(stt_resp, "Speech-to-text")
            stt_data = stt_body.get("data", stt_body)
            transcript = (stt_data.get("transcribed_text") or stt_data.get("text") or "").strip()
            if not transcript:
                continue

            score = len(transcript)
            if lang == "am" and re.search(r"[\u1200-\u137F]", transcript):
                score += 1000
            if lang == "or" and re.search(r"\b(dh|ph|ch|fi|xi|af\s+oromoo|oromoo|galma|beekumsa)\b", transcript, re.IGNORECASE):
                score += 600
            if lang == "en" and re.search(r"[A-Za-z]", transcript):
                score += 300

            if score > best_score:
                best_score = score
                best_language = lang
                best_transcript = transcript
        except Exception:
            continue

    if best_score < 0:
        raise HTTPException(status_code=502, detail="Auto language detection STT failed for all languages.")

    return best_language, best_transcript


def _build_search_context(transcript: str, max_results: int = 5) -> str:
    """Perform a search and return a compact, prompt-friendly summary of snippets."""
    snippets = _search_cbe(transcript, max_results=max_results)
    if not snippets:
        return ""

    pieces = []
    for i, result in enumerate(snippets, start=1):
        source = result.get("source", "web")
        title = (result.get("title") or "").strip()
        body = (result.get("body") or result.get("snippet") or "").strip()
        href = (result.get("href") or "").strip()
        summary = title or body or href
        pieces.append(f"[{i}] [{source}] {summary} ({href})")
        if len(pieces) >= max_results:
            break

    return "\n".join(pieces)


@app.post("/speech-to-speech", summary="Auto-detect language, translate speech and return audio")
async def speech_to_speech(
    file: UploadFile = File(..., description="Source audio file (.mp3 or .wav)"),
    source_language: str = Form("auto", description="Source language (en/am/or) or 'auto' to detect"),
    target_language: str = Form("auto", description="Target language (en/am/or) or 'auto' to match source"),
):
    _remote_service_disabled("Speech-to-speech")
    audio_bytes = await file.read()
    filename    = file.filename or "audio.webm"
    mime        = file.content_type or "audio/mpeg"

    # ── Step 1: if source is auto, run STT with all three languages and pick best ──
    if source_language == "auto":
        # Try STT with "en" first (most common); gateway will return whatever it hears
        stt_resp = _SESSION.post(
            STT_URL,
            headers={"X-API-Key": GATEWAY_API_KEY},
            data={"language": "en"},
            files={"file": (filename, audio_bytes, mime)},
            timeout=120,
        )
        stt_body = _raise_upstream(stt_resp, "STT (auto-detect)")
        stt_data = stt_body.get("data", stt_body)
        transcript = stt_data.get("transcribed_text") or stt_data.get("text") or ""
        source_language = _detect_language_from_text(transcript)
        # If Ethiopic detected, re-run STT with correct language for better accuracy
        if source_language != "en":
            stt_resp2 = _SESSION.post(
                STT_URL,
                headers={"X-API-Key": GATEWAY_API_KEY},
                data={"language": source_language},
                files={"file": (filename, audio_bytes, mime)},
                timeout=120,
            )
            try:
                stt_body2 = _raise_upstream(stt_resp2, "STT (rerun)")
                stt_data2 = stt_body2.get("data", stt_body2)
                transcript2 = stt_data2.get("transcribed_text") or stt_data2.get("text") or ""
                if transcript2.strip():
                    transcript = transcript2
            except Exception:
                pass  # keep first transcript

    # If target is auto, reply in the same language as the source
    if target_language == "auto":
        target_language = source_language

    # ── Step 2: try direct S2S endpoint ──────────────────────────────────────
    response = _SESSION.post(
        S2S_URL,
        headers={"X-API-Key": GATEWAY_API_KEY},
        data={
            "source_language":      source_language,
            "destination_language": target_language,
            "language":             source_language,
        },
        files={"file": (filename, audio_bytes, mime)},
        timeout=180,
    )

    if response.status_code != 404:
        body = _raise_upstream(response, "Speech-to-speech")
        data = body.get("data", body)
        translated_text = (
            data.get("translated_text") or data.get("transcribed_text") or data.get("text") or ""
        )
        audio = _extract_audio(body, response)
        return Response(
            content=audio,
            media_type="audio/mpeg",
            headers={
                "X-Translated-Text":  translated_text,
                "X-Source-Language":  source_language,
                "X-Target-Language":  target_language,
                "Access-Control-Expose-Headers": "X-Translated-Text,X-Source-Language,X-Target-Language",
            },
        )

    # ── Step 3: fallback STT → translate → TTS ───────────────────────────────
    stt_resp = _SESSION.post(
        STT_URL,
        headers={"X-API-Key": GATEWAY_API_KEY},
        data={"language": source_language},
        files={"file": (filename, audio_bytes, mime)},
        timeout=120,
    )
    stt_body = _raise_upstream(stt_resp, "STT (fallback)")
    stt_data = stt_body.get("data", stt_body)
    source_text = stt_data.get("transcribed_text") or stt_data.get("text") or ""
    if not source_text.strip():
        raise HTTPException(status_code=502, detail="STT returned empty transcript.")

    translated_text = source_text
    if source_language != target_language:
        tr_resp = _SESSION.post(
            TRANSLATE_URL,
            headers=_gateway_json_headers(),
            json={
                "source_text":        source_text,
                "source_language":    source_language,
                "destination_language": target_language,
            },
            timeout=60,
        )
        tr_body = _raise_upstream(tr_resp, "Translate (fallback)")
        tr_data = tr_body.get("data", tr_body)
        translated_text = tr_data.get("translated_text") or tr_data.get("text") or source_text

    tts_resp = _SESSION.post(
        TTS_URL,
        headers={"X-API-Key": GATEWAY_API_KEY, "Accept": "application/json"},
        data={"text": translated_text, "language": target_language},
        timeout=120,
    )
    if tts_resp.status_code in (404, 415, 422):
        tts_resp = _SESSION.post(
            TTS_URL,
            headers=_gateway_json_headers(),
            json={"text": translated_text, "language": target_language},
            timeout=120,
        )
    tts_body = _raise_upstream(tts_resp, "TTS (fallback)")
    audio = _extract_audio(tts_body, tts_resp)

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "X-Translated-Text":  translated_text,
            "X-Source-Language":  source_language,
            "X-Target-Language":  target_language,
            "Access-Control-Expose-Headers": "X-Translated-Text,X-Source-Language,X-Target-Language",
        },
    )


@app.post("/speech-chat", summary="Voice chat: STT -> text -> language detect -> search/chat -> TTS with transcript")
async def speech_chat(
    file: UploadFile = File(..., description="Source audio file (.mp3 or .wav)"),
    language: str = Form("auto", description="Spoken language for STT and TTS, or auto to detect"),
):
    """Transcribe audio, detect language, search and answer, then return speech plus transcript."""
    _remote_service_disabled("Speech chat")
    audio_bytes = await file.read()
    filename = file.filename or "audio.wav"
    mime = file.content_type or "audio/mpeg"

    # 1) Transcribe speech to text
    initial_lang = "en" if language == "auto" else language
    stt_resp = _SESSION.post(
        STT_URL,
        headers={"X-API-Key": GATEWAY_API_KEY},
        data={"language": initial_lang},
        files={"file": (filename, audio_bytes, mime)},
        timeout=120,
    )
    stt_body = _raise_upstream(stt_resp, "Speech-to-text")
    stt_data = stt_body.get("data", stt_body)
    transcript = (stt_data.get("transcribed_text") or stt_data.get("text") or "").strip()
    if not transcript:
        raise HTTPException(status_code=502, detail="Speech-to-text returned empty transcript.")

    # 2) Detect spoken language from the transcript text
    detected_language = _detect_language_from_text(transcript)
    if language == "auto":
        language = detected_language
    elif language != detected_language:
        logger.info("Explicit language %s differs from detected language %s; using explicit setting.", language, detected_language)

    if language == "auto":
        language = detected_language

    if language != initial_lang:
        # Re-run STT in the detected language for better accuracy
        stt_resp2 = _SESSION.post(
            STT_URL,
            headers={"X-API-Key": GATEWAY_API_KEY},
            data={"language": language},
            files={"file": (filename, audio_bytes, mime)},
            timeout=120,
        )
        try:
            stt_body2 = _raise_upstream(stt_resp2, "Speech-to-text (detected language)")
            stt_data2 = stt_body2.get("data", stt_body2)
            transcript2 = (stt_data2.get("transcribed_text") or stt_data2.get("text") or "").strip()
            if transcript2:
                transcript = transcript2
        except Exception:
            pass

    transcript = transcript.strip()
    if not transcript:
        raise HTTPException(status_code=502, detail="Speech-to-text returned empty transcript after language detection.")

    # 3) Search using the transcript
    search_context = _build_search_context(transcript, max_results=5)
    lang_name = _LANG_CODE_TO_NAME.get(language, language)
    system_instructions = [
        {
            "role": "system",
            "content": (
                f"Answer in {lang_name}. Use the search results below when relevant. Keep the answer concise "
                "and in the same language as the user."
            ),
        }
    ]
    if search_context:
        system_instructions.append({"role": "system", "content": f"Search results:\n{search_context}"})

    messages = system_instructions + [{"role": "user", "content": transcript}]

    # 4) Generate the reply with local model first, then gateway fallback
    reply = ""
    try:
        reply = _generate_local_reply(messages, temperature=0.7, max_tokens=1024)
    except Exception as exc:
        logger.warning("Local speech-chat failed, falling back to gateway: %s", exc)
        gw_payload = {
            "model": "mela4.3",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "openai_compatible": True,
        }
        try:
            response = _SESSION.post(GATEWAY_LLM_URL, headers=_gateway_json_headers(), json=gw_payload, timeout=120)
            gw_body = _raise_upstream(response, "Gateway LLM")
            data = gw_body.get("data", gw_body)
            reply = data["choices"][0]["message"]["content"]
        except Exception as gw_exc:
            logger.warning("Gateway LLM fallback failed: %s", gw_exc)
            if not search_context:
                raise HTTPException(status_code=502, detail=f"Speech chat failed: {gw_exc}") from gw_exc
            combined = "\n\n".join(search_context.splitlines())[:3000]
            if language != "en":
                translated = None
                try:
                    tr_resp = _SESSION.post(
                        TRANSLATE_URL,
                        headers=_gateway_json_headers(),
                        json={
                            "source_text": combined,
                            "source_language": "auto",
                            "destination_language": language,
                        },
                        timeout=60,
                    )
                    tr_body = _raise_upstream(tr_resp, "Translate (fallback)")
                    tr_data = tr_body.get("data", tr_body)
                    translated = tr_data.get("translated_text") or tr_data.get("text")
                except Exception:
                    translated = None
                reply = f"[Search fallback]\n{translated or combined}"
            else:
                reply = f"[Search fallback]\n{combined}"

    if not reply.strip():
        raise HTTPException(status_code=502, detail="Failed to generate a reply.")

    # 5) Produce speech audio in the detected language
    tts_resp = _SESSION.post(
        TTS_URL,
        headers={"X-API-Key": GATEWAY_API_KEY},
        data={"text": reply, "language": language},
        timeout=120,
    )
    if tts_resp.status_code in (404, 415, 422):
        tts_resp = _SESSION.post(
            TTS_URL,
            headers=_gateway_json_headers(),
            json={"text": reply, "language": language},
            timeout=120,
        )
    tts_body = _raise_upstream(tts_resp, "Text-to-speech")
    audio = _extract_audio(tts_body, tts_resp)

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "X-Transcript-Text": transcript,
            "X-Reply-Text": reply.strip(),
            "X-Detected-Language": language,
            "Access-Control-Expose-Headers": "X-Transcript-Text,X-Reply-Text,X-Detected-Language",
        },
    )

# ---------------------------------------------------------------------------
# Run directly with: python fastapi_app.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Local STT / TTS demo (uses faster-whisper and pyttsx3 when available)
# ---------------------------------------------------------------------------
LOCAL_STT_MODEL = None


def _load_local_stt_model(model_name: str = "small"):
    """Lazy-load a faster-whisper WhisperModel. Model will be downloaded
    from Hugging Face if not present locally — choose a small model for
    quick testing."""
    global LOCAL_STT_MODEL
    if LOCAL_STT_MODEL is not None:
        return LOCAL_STT_MODEL

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError("faster-whisper not installed") from exc

    # Device selection: prefer CUDA if available
    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_name, device=device)
    LOCAL_STT_MODEL = model
    return model


@app.post("/stt-local", summary="Local STT using faster-whisper")
async def stt_local(file: UploadFile = File(...), language: Language = Form("en")):
    """Transcribe uploaded audio using a locally installed faster-whisper model.
    If the package or model is missing a 501 is returned with guidance."""
    try:
        import tempfile
        model = _load_local_stt_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc))

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    loop = asyncio.get_event_loop()

    def do_transcribe():
        # faster-whisper returns (segments, info)
        segments, info = model.transcribe(tmp_path, language=language)
        text = "".join(s.text for s in segments)
        return text

    try:
        text = await loop.run_in_executor(None, do_transcribe)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Local STT failed: {exc}")

    return {"text": text, "language": language}


@app.post("/tts-local", summary="Local TTS using pyttsx3")
async def tts_local(text: str = Form(...), language: Language = Form("en")):
    """Synthesize speech locally using pyttsx3 and return a WAV file.
    Returns 501 if `pyttsx3` is not installed."""
    try:
        import pyttsx3
        import tempfile
    except Exception as exc:
        raise HTTPException(status_code=501, detail="pyttsx3 not installed")

    engine = pyttsx3.init()
    # Choose voice by language heuristics if available
    try:
        voices = engine.getProperty("voices")
        for v in voices:
            if language == "en" and "en" in v.languages[0].decode("utf-8"):
                engine.setProperty("voice", v.id)
                break
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

    # Run blocking TTS in thread so we don't block the event loop
    loop = asyncio.get_event_loop()

    def synth():
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return out_path

    try:
        path = await loop.run_in_executor(None, synth)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Local TTS failed: {exc}")

    return FileResponse(path, media_type="audio/wav", filename="tts.wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
