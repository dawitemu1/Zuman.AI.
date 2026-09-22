# Zuman.AI

> A multilingual, local-first banking assistant for the Commercial Bank of Ethiopia (CBE).

Zuman.AI combines a FastAPI backend, a React/Vite frontend, and a locally hosted Gemma 4 12B adapter to answer CBE-focused questions in English, Amharic, Afaan Oromoo, Tigrinya, and Somali.

## Highlights

- Local chat inference with a fine-tuned Gemma 4 12B model
- Streaming responses over Server-Sent Events (SSE)
- Language detection and localized banking guidance
- Verified CBE links for link and product requests
- React interface with Markdown rendering and syntax highlighting
- Optional Redis session history and FAQ caching
- Training scripts and curated CBE knowledge-base data included

> **Project status:** The chat workflow is available for local use. Speech, translation, and speech-to-speech routes require their corresponding gateway integration and currently return `501` in the local-only configuration.

## Contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [API](#api)
- [Model setup](#model-setup)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Responsible use](#responsible-use)

## Architecture

```text
React/Vite frontend
				│ HTTP / SSE
				▼
FastAPI application ──► local inference pipeline
				│                         │
				├── optional Redis        └── Gemma 4 12B + LoRA adapter
				└── CBE knowledge and verified links
```

## Requirements

- Python 3.12 or a compatible recent Python version
- Node.js and npm
- A CUDA-capable GPU recommended for 12B model inference
- Optional Redis instance for sessions and FAQ caching

The full Python dependency list is in [Requirements.txt](Requirements.txt). Large base-model weights and training checkpoints are intentionally not tracked in Git.

## Quick start

### 1. Start the backend

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Requirements.txt
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

The API starts at `http://localhost:8000`. OpenAPI documentation is available at `http://localhost:8000/docs`.

### 2. Start the frontend

In a second terminal:

```bash
cd Frontend
npm install
npm run dev
```

The Vite development server proxies API requests to port `8000`.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Check service and model availability |
| `POST` | `/chat` | Return a complete assistant response |
| `POST` | `/chat/stream` | Stream a response as SSE |
| `POST` | `/speech-to-text` | Transcribe uploaded audio |
| `POST` | `/text-to-speech` | Generate audio from text |
| `POST` | `/translate` | Translate between supported languages |
| `POST` | `/speech-to-speech` | Translate audio and return audio |

### Example request

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
	-H 'Content-Type: application/json' \
	-d '{
		"messages": [
			{"role": "user", "content": "What documents are needed to open a CBE account?"}
		],
		"temperature": 0.7,
		"max_tokens": 512
	}'
```

For streaming output, use the same request body with `curl -N` and the `/chat/stream` endpoint.

## Model setup

The inference pipeline uses:

- Base model: `google/gemma-4-12b-it`
- Fine-tuned adapter: `./banking-model-gemma4-12b`
- Runtime: PyTorch, Transformers, PEFT, and Unsloth

Set a different location when needed:

```bash
export BASE_MODEL=google/gemma-4-12b-it
export FINETUNED_MODEL=./banking-model-gemma4-12b
```

The base model must be available locally or downloadable by the Hugging Face tooling. CPU-only execution may be too slow or exceed available memory. For supported GPUs, `LOAD_IN_4BIT=1` can reduce memory usage.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `BASE_MODEL` | `google/gemma-4-12b-it` | Base Hugging Face model |
| `FINETUNED_MODEL` | `./banking-model-gemma4-12b` | LoRA adapter directory |
| `LOAD_IN_4BIT` | `0` | Enable 4-bit model loading |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `REDIS_SESSION_TTL` | `1800` | Session lifetime in seconds |
| `REDIS_FAQ_TTL` | `86400` | FAQ cache lifetime in seconds |

Redis is optional and disabled by default in the current application configuration.

## Repository layout

```text
.
├── fastapi_app.py                 # FastAPI routes and service configuration
├── inference.py                   # Model loading and response generation
├── finetune.py                    # Unsloth/TRL fine-tuning workflow
├── Requirements.txt               # Python dependencies
├── Frontend/                      # React/Vite client
├── Final/                         # CBE knowledge-base and training data
└── banking-model-gemma4-12b/      # Adapter and tokenizer files
```

## Development

Run the available checks from the repository root:

```bash
python -m compileall fastapi_app.py inference.py
cd Frontend
npm run lint
npm run build
```

To build and preview the frontend without the development server:

```bash
npm run build
npm run preview
```

## Troubleshooting

### The model does not load

- Confirm the adapter directory and base model are available.
- Check that PyTorch matches the installed CUDA runtime.
- Verify that the active virtual environment contains `Requirements.txt` dependencies.
- Inspect the backend startup log for the original model-loading error.

### The frontend cannot reach the backend

- Start FastAPI on port `8000` first.
- Confirm `curl http://localhost:8000/health` succeeds.
- Restart Vite after changing its proxy configuration.

### Inference is slow or runs out of memory

- Set `LOAD_IN_4BIT=1` on supported hardware.
- Reduce the maximum sequence or generated-token settings in the inference configuration.
- Avoid starting multiple model workers on the same GPU.

## Responsible use

Zuman.AI provides informational assistance and must not be treated as a replacement for official banking support or account-specific decisions. Confirm important information with CBE Customer Care at `951` or at a CBE branch.

Before production deployment:

- Keep API keys, credentials, model weights, and local environments out of Git.
- Restrict CORS to trusted frontend origins.
- Use HTTPS, authentication, request limits, logging, and monitoring.
- Validate banking policies and product information with CBE.