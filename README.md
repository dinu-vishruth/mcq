# Agentic RAG MCQ Generator

## Overview

An AI-powered learning platform for generating and managing MCQ assessments from uploaded study material. The application features a progressive RAG pipeline that falls back to legacy single-shot generation when retrieval is unavailable or models fail.

## What the App Does

- File Processing: Upload PDF, DOCX, PPTX, or TXT files
- Text Extraction: Clean and extract source text from documents
- Library Management: Save resources to per-user libraries
- Content Indexing: Index resource text for intelligent retrieval
- MCQ Generation:
  - Upload material to save and index a resource, then configure a quiz
  - Generate from retrieved library content; verification stages run when `AI_PIPELINE=rag`
- Session Tracking: Monitor quiz sessions and student results
- Learning Analytics: Identify weak topics after quizzes
- Explanation Generation: Create revision guidance from source material
- Cross-Platform: local and Vercel deployments

## Current Architecture

```text
app.py                    # FastAPI entry point
worker.py                 # RQ background worker for MCQ generation (optional)
fastapi_app/              # FastAPI app, routers, deps, middleware, DB bootstrap
config.py                 # Environment configuration and runtime defaults
core/
  llm/                    # LLM providers and provider auto-detection
  embeddings/             # Hashing / SentenceTransformer / remote backends
  vectorstore/            # Chroma, SQLite+NumPy, FAISS, pgvector
  rag/                    # Chunking, retrieval, context-building
  agents/                 # Planner, retriever, fact-checking, evaluation, etc.
  services/                # Orchestration for generation and learning flows
  repositories/           # SQLite/Postgres data access
  storage/                # Upload object storage (local or S3-compatible)
  observability.py        # Structured JSON logging and tracing hooks
  prompts/                # Prompts for MCQ generation and verification
  models/                 # Database migrations and schema helpers
models/                   # Legacy generation fallback path
utils/                    # Text cleaning and session helpers
frontend/                 # Vite + React + TypeScript UI source
static/dist/              # Built frontend assets
templates/                # Server-rendered shell templates
database/                 # SQLite database and schema files, numbered migrations
evals/                    # Evaluation dataset, harness and latest report
```

## Tech Stack

Backend:
- FastAPI (Python 3.10+)
- SQLite by default; Postgres via `DATABASE_URL` (psycopg)
- Signed cookies with CSRF protection

Frontend:
- React + TypeScript
- Vite
- Tailwind CSS
- Framer Motion
- Lucide React
- Recharts

AI/ML:
- LLM Providers: Groq, xAI, OpenAI, Gemini, Anthropic (one default model per provider)
- Embedding Backends: SentenceTransformers, remote APIs, hashing fallback
- Vector Stores: ChromaDB, SQLite+NumPy, FAISS, pgvector
- Document Parsing: PDF, Word, PowerPoint, TXT
- Background Jobs: RQ + Redis (optional; inline by default)
- Deployment: Vercel/Render-ready and local Uvicorn compatible

## Typical User Flow

1. Upload or Library: user uploads file or saves study resource
2. Content Processing: text cleaned and divided into chunks
3. Embedding Generation: vectors created/stored
4. Quiz Configuration: choose a saved resource, difficulty, question count, topic focus, and timer
   - New uploads redirect to this configuration step after indexing
   - Generation retrieves source content and uses the configured legacy or RAG pipeline
5. Quiz Creation: session created and student answers recorded
6. Learning Analysis: weak-topic analysis and revision guidance generated

## Key Runtime Behavior

Passwords are preserved exactly, including leading/trailing spaces. New accounts
and password changes require at least 10 characters and two of lowercase,
uppercase, numbers or symbols. Existing passwords remain usable at login.
Login lockout applies to a `(username, IP)` pair after five failures in five
minutes. With SQLite on Vercel this protection is instance-local until the
database becomes persistent; persistent hosting is a separate follow-up.

Environment variable compatibility:
- `GROQ_API_KEY` and `GROK_API_KEY` both accepted
- `GROQ_MODEL` and `GROK_MODEL` both accepted
- Blank values treated as unset, preventing silent overrides

Vercel:
- Uses `/tmp` for writable storage
- Forces SQLite vector-store fallback (read-only filesystem)
- Reads environment variables directly from platform
- Maintains working state without local `.env`

Library:
- Adding a resource saves and indexes it inline; the library lists it once indexing finishes.
- Uploading through Make Quiz & Test saves and indexes the resource, then redirects to quiz configuration.
- Failed indexing surfaces as an error instead of a phantom library entry.

## Environment Variables

Minimal (local dev):
```dotenv
SECRET_KEY=your_long_random_secret
GROK_API_KEY=gsk_your_key_here
# or GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=auto
AI_PIPELINE=legacy
```

Common provider/model settings:
```dotenv
GROQ_API_KEY=
GROK_API_KEY=
GROQ_MODEL=
GROK_MODEL=
XAI_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
LLM_API_KEY=
LLM_MODEL=            # empty = provider's own default model
```

Storage/persistence:
```dotenv
DATABASE_URL=         # empty = SQLite at DB_PATH; set for Postgres
DATA_DIR=             # default: project dir locally, /tmp on serverless
DB_PATH=              # default: $DATA_DIR/database/mcq.db
UPLOAD_BACKEND=local  # or s3
S3_UPLOAD_BUCKET=
S3_UPLOAD_REGION=
S3_UPLOAD_PREFIX=uploads
S3_UPLOAD_ENDPOINT_URL=
```

RAG/retrieval:
```dotenv
EMBEDDING_BACKEND=auto
VECTOR_STORE=          # default: chroma locally, sqlite on serverless
CHUNK_SIZE=1500
CHUNK_OVERLAP=80
RETRIEVAL_TOP_K=8
CONTEXT_MAX_CHARS=12000
```

Generation limits:
```dotenv
GENERATION_COOLDOWN_SECONDS=3
GENERATION_MAX_PER_5MIN=10
GENERATION_JOB_DEADLINE_SECONDS=900
GENERATION_BACKEND=auto   # auto | inline | rq
QUEUE_NAME=mcq
REDIS_URL=                # required when GENERATION_BACKEND=rq
MAX_UPLOAD_BYTES=4718592
MAX_PDF_PAGES=300
MAX_REQUEST_BODY_BYTES=8388608
```

Observability:
```dotenv
LOG_LEVEL=INFO
LOG_FORMAT=json           # or text
OBSERVABILITY_TRACE_HOOK=  # optional module:function, e.g. my_observability:on_event
OBSERVABILITY_BACKEND=auto  # auto | langsmith | langfuse | hook | none
```

Deployed-only:
```dotenv
SEED_DEMO_USERS=false     # local opt-in only; random passwords printed once
```

Notes:
- Each provider has its own default model (see `config.PROVIDER_DEFAULT_MODELS`).
  `LLM_MODEL` is no longer aliased to `GROK_MODEL`; a foreign model name is
  replaced with the resolved provider's default rather than sent to the wrong API.
- `SECRET_KEY` is required when `VERCEL` or `RENDER` is set; the app refuses to
  boot without it rather than using a per-process random key.

## Local Setup

```bash
pip install -r requirements.txt
copy .env.example .env
python app.py
```

Then `http://127.0.0.1:8000/`.

Tests:
```bash
pip install pytest ruff
pytest -q
ruff check .
```
`pytest` deselects `live` tests by default; run them with `pytest -m live` (needs
real credentials).

## Vercel / Render

Vercel-ready via `vercel.json` and `app.py`; set env vars in project settings and
read platform env vars; no additional config changes. Render works the same way
with `DATA_DIR` pointed at the mounted disk. `SECRET_KEY` is required on both.

## Database / Storage

Local:
- DB: `database/mcq.db` (SQLite)
- Uploads: `uploads/`

Vercel:
- DB: `/tmp/mcq.db` (ephemeral)
- Uploads: `/tmp/uploads/`

Production (persistent):
- Set `DATABASE_URL` to a Postgres instance (optionally with pgvector) and
  `DATA_DIR` to a persistent disk. Uploads can go to S3-compatible storage via
  `UPLOAD_BACKEND=s3`.

Vercel storage is temporary and is not a durable database for user accounts,
resources, or results. A production deployment needs persistent database/storage
services.

Persisted:
- document metadata
- chunked text content
- cached embeddings
- vector index entries
- generation job history

## Background Worker (optional)

By default generation runs inline in the web process. For long generations on a
persistent host you can move it to a worker:

```bash
GENERATION_BACKEND=rq REDIS_URL=redis://... python worker.py
```

With `GENERATION_BACKEND=auto` the app picks RQ when `REDIS_URL` is set and the
`redis`/`rq` packages are installed, otherwise it stays inline. Queue mode
changes the `POST /api/practice/generate` response from
`{"ok": true, "session_key": ..., "count": N}` to
`{"ok": true, "job_id": ..., "status": "queued"}`; results are fetched from
`GET /api/jobs/{id}` or streamed from `GET /api/jobs/{id}/events`. Inline mode
keeps the original shape.

## Testing

Tests cover:
- resource save-to-library behavior
- no-duplicate resource saves per user
- MCQ generation flow (fake LLM)
- weak-topic evaluation/review flow
- verification and grading logic
- persistence, background jobs, model resolution, security, retrieval, eval scoring
- observability (JSON logging, request ids, stage timing, LLM usage)
- upload storage (local + S3-compatible, faked client)
- live provider generation (credential-gated, deselected by default)

Commands:

```bash
pytest -q
ruff check .
```

`pytest` deselects `live` tests by default; run them with `pytest -m live` (needs
real credentials).

## Evaluation

The repository ships an evaluation harness and a small dataset:

```bash
# deterministic, no credentials (verifies the harness only)
python evals/run_eval.py --dry-run

# real provider (costs tokens)
python evals/run_eval.py
```

It writes `evals/results/latest.json` and `evals/results/latest.md`. See
`docs/observability.md` for log schema and tracing.

## Deployment Notes

Graceful degradation:
- degrades gracefully when AI services or retrieval unavailable
- safe fallback paths prevent hard failures
- explicit handling for env var mistakes
- blank overrides and Groq/Grok naming conflicts resolved

Performance:
- LLM rate limiting
- pipeline deadline enforcement for serverless environments
- chunking balances retrieval quality and indexing speed
- caching reduces redundant processing

## Observability

Structured JSON logging (one object per line) with request ids, stage timings,
and LLM token usage. Configure via `LOG_LEVEL`, `LOG_FORMAT`, and the optional
`OBSERVABILITY_BACKEND` / `OBSERVABILITY_TRACE_HOOK`. See
`docs/observability.md` for the event schema and LangSmith/Langfuse/OTLP hook
examples.

## Compatibility Notes

- Public API routes are unchanged. The only response-shape change is in queue
  mode (see Background Worker); inline mode keeps the original shape.
- `.env`, `uploads/`, `database/*.db`, `chroma_db/` are gitignored; never commit
  secrets or local databases.
- Orchestration is hand-rolled (no LangChain); `AI_PIPELINE=rag` opts into the
  retrieval-grounded path, `legacy` (default) keeps single-shot generation.
- New runtime dependencies are limited to `psycopg[binary]` (Postgres); Redis/RQ,
  boto3 and sentence-transformers remain optional.

## Demo GIF (TODO)

A short screencast of upload -> quiz configuration -> quiz -> results would
belong here. Not yet recorded; tracked as a follow-up.

## Project Summary

- FastAPI Backend: API server, auth, session management | FastAPI, SQLite/Postgres, CSRF
- React Frontend: UI, visualization | React, TypeScript, Vite
- AI Pipeline: MCQ generation, RAG retrieval | Groq/xAI/OpenAI/Gemini/Anthropic, ChromaDB/SQLite/pgvector
- Document Processing: extraction, cleaning | Custom NLP, chunking
- Learning Analytics: performance tracking, weak topics | SQLite/Postgres, custom algorithms
- Observability: structured JSON logs, request ids, tracing hooks | stdlib logging

## Getting Started

Clone, install requirements, configure `.env`, run `python app.py`, access localhost.
