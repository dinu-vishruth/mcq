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
fastapi_app/              # FastAPI app, routers, deps, middleware, DB bootstrap
config.py                 # Environment configuration and runtime defaults
core/
  llm/                    # LLM providers and provider auto-detection
  embeddings/             # Hashing / SentenceTransformer / remote backends
  vectorstore/            # Chroma, SQLite+NumPy, FAISS
  rag/                    # Chunking, retrieval, context-building
  agents/                 # Planner, retriever, fact-checking, evaluation, etc.
  services/               # Orchestration for generation and learning flows
  prompts/                # Prompts for MCQ generation and verification
  repositories/           # SQLite repositories for docs, results, learning data
  models/                 # Database migrations and schema helpers
models/                   # Legacy generation fallback path
utils/                    # Text cleaning and session helpers
frontend/                 # Vite + React + TypeScript UI source
static/dist/              # Built frontend assets
templates/                # Server-rendered shell templates
database/                 # SQLite database and schema files
```

## Tech Stack

Backend:
- FastAPI (Python 3)
- SQLite
- Signed cookies with CSRF protection

Frontend:
- React + TypeScript
- Vite
- Tailwind CSS
- Framer Motion
- Lucide React
- Recharts

AI/ML:
- LLM Providers: Groq, xAI, OpenAI, Gemini, Anthropic
- Embedding Backends: SentenceTransformers, remote APIs, hashing fallback
- Vector Stores: ChromaDB, SQLite+NumPy, FAISS
- Document Parsing: PDF, Word, PowerPoint, TXT
- Deployment: Vercel-ready and local Gunicorn/Uvicorn compatible

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
- Adding a resource starts indexing in a background thread and redirects to the library without generating a quiz.
- Uploading through Make Quiz & Test saves and indexes the resource, then redirects to quiz configuration.
- Background indexing currently relies on the application process remaining alive; failures are logged server-side.

## Environment Variables

Minimal:
```dotenv
SECRET_KEY=your_long_random_secret
GROK_API_KEY=gsk_your_key_here
# or GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=auto
AI_PIPELINE=legacy
```

Common:
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
LLM_MODEL=

EMBEDDING_BACKEND=auto
VECTOR_STORE=chroma
CHUNK_SIZE=1500
CHUNK_OVERLAP=80
RETRIEVAL_TOP_K=8
CONTEXT_MAX_CHARS=12000
```

## Local Setup

```bash
pip install -r requirements.txt
copy .env.example .env
python app.py
```

Then `http://127.0.0.1:8000/`.

## Vercel

Vercel-ready via `vercel.json` and `app.py`; set env vars in project settings; reads platform env vars; no additional config changes.

## Database/Storage

Local:
- DB: `database/mcq.db`
- Uploads: `uploads/`

Vercel:
- DB: `/tmp/mcq.db`
- Uploads: `/tmp/uploads/`

Vercel storage is temporary and is not a durable database for user accounts, resources, or results. A production deployment needs persistent database/storage services.

Persisted:
- document metadata
- chunked text content
- cached embeddings
- vector index entries

## Testing

Tests cover:
- resource save-to-library behavior
- no-duplicate resource saves per user
- MCQ generation flow
- weak-topic evaluation/review flow
- verification and grading logic

Commands:

```bash
python -m unittest tests.test_features -v
python -m unittest tests.test_verification -v
```

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

## Project Summary

- FastAPI Backend: API server, auth, session management | FastAPI, SQLite, CSRF
- React Frontend: UI, visualization | React, TypeScript, Vite
- AI Pipeline: MCQ generation, RAG retrieval | Groq/xAI/OpenAI, ChromaDB
- Document Processing: extraction, cleaning | Custom NLP, chunking
- Learning Analytics: performance tracking, weak topics | SQLite, custom algorithms

## Getting Started

Clone, install requirements, configure `.env`, run `python app.py`, access localhost.
