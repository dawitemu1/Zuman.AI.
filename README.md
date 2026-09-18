# Zuman.AI

Zuman.AI is a multilingual banking assistant for Commercial Bank of Ethiopia (CBE). It combines a FastAPI service with a React/Vite interface and supports chat, streaming responses, speech workflows, translation, and local model inference.

## Features

- Chat in English, Amharic, Afaan Oromoo, Tigrinya, and Somali
- Streaming chat responses over Server-Sent Events
- Speech-to-text, text-to-speech, and speech-to-speech workflows
- CBE-focused answers with verified links and localized banking guidance
- Optional Redis-backed sessions and FAQ caching
- React interface with Markdown rendering and code highlighting

## Project Layout

```text
.
├── fastapi_app.py                 # FastAPI application and API routes
├── inference.py                   # Local model loading and generation
├── finetune.py                    # Fine-tuning workflow
├── Requirements.txt               # Python dependencies
├── Frontend/                      # React/Vite client
├── Final/                         # Knowledge-base source data
└── banking-model-gemma4-12b/      # Local adapter configuration and tokenizer
```

Large model weights, optimizer checkpoints, virtual environments, build output, and generated training data are intentionally excluded from Git. Download or copy the required weights into the expected local model directory before running local inference.

## Requirements

- Python 3.12 or a compatible recent Python version
- Node.js and npm
- CUDA-capable GPU recommended for local 12B model inference
- Optional: Redis, if session persistence or FAQ caching is enabled

## Backend Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Requirements.txt
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

The API is available at `http://localhost:8000`. Interactive API documentation is available at `/docs`, and the health check is available at `/health`.

## Frontend Setup

In a second terminal:

```bash
cd Frontend
npm install
npm run dev
```

For a production build:

```bash
npm run build
npm run preview
```

The Vite development server proxies API requests to the local FastAPI service on port `8000`.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness check |
| `POST` | `/chat` | Return a complete assistant response |
| `POST` | `/chat/stream` | Stream an assistant response as SSE |
| `POST` | `/speech-to-text` | Transcribe uploaded audio |
| `POST` | `/text-to-speech` | Generate audio from text |
| `POST` | `/translate` | Translate between supported languages |
| `POST` | `/speech-to-speech` | Translate audio and return translated audio |

## Configuration

Useful environment variables include:

```bash
export REDIS_URL=redis://localhost:6379/0
export REDIS_SESSION_TTL=1800
export REDIS_FAQ_TTL=86400
export LOAD_IN_4BIT=0
```

The application can continue without Redis. Local model inference uses CUDA automatically when it is available; set `LOAD_IN_4BIT=1` when 4-bit loading is required by the available hardware.

## Development Checks

```bash
python -m compileall fastapi_app.py inference.py
cd Frontend
npm run lint
npm run build
```

## Notes

- Do not commit API keys, credentials, local environments, model weights, or generated checkpoints.
- Banking answers should be treated as informational. Customers should confirm account-specific matters with CBE Customer Care at `951` or a branch.
- The frontend currently uses the browser Speech Synthesis API for text-to-speech when available.