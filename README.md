# AI-Powered MCQ Generator

An **agentic RAG** learning platform built on Flask. Teachers upload course material; the app ingests and indexes it, then uses a pipeline of specialized AI agents to generate high-quality Multiple Choice Questions, grade student performance, and produce grounded explanations and revision notes.

The system ships with a safe **legacy single-shot** path (the historical behaviour) and an opt-in **retrieval-augmented (RAG) pipeline**. You switch between them with a single environment variable — no code changes.

## Features

### For Teachers
- **Account management**: signup, login with lockout protection, secure filesystem sessions, and profile/account deletion.
- **Document ingestion**: upload `.pdf`, `.docx`, or `.pptx`. Text is extracted, cleaned, chunked, embedded, and indexed for retrieval.
- **AI question generation**: choose the number of questions, difficulty (easy/medium/hard), and time limit. In RAG mode a multi-agent pipeline retrieves the most relevant passages and generates questions grounded in the source, following Bloom's taxonomy.
- **Session management**: every test gets a unique **Session Key** to share with students. Clone, edit, archive/unarchive, and delete sessions.
- **Reports & analytics**: per-session student performance reports with visual analytics; export as PDF or CSV.

### For Students
- **Easy access**: join any active test with your name and the Session Key.
- **Interactive test**: randomized questions and shuffled options within a configurable time limit.
- **Instant results & explanations**: immediate scoring plus AI-driven, source-grounded explanations for each answer.
- **Learning intelligence**: post-submission analysis detects weak topics and surfaces them on your dashboard, with retrieval-grounded revision notes.
- **History**: track past results and performance from your dashboard.

## Architecture

The application evolved in additive phases. Everything new lives under `core/` and is gated behind config flags with graceful fallbacks, so the original single-shot flow keeps working untouched.

```
app.py                     # thin Flask entry point (imported by Procfile & vercel.json)
config.py                  # all env-var configuration (via python-dotenv)
core/
  llm/                     # provider abstraction (Groq, xAI, OpenAI, Together, Gemini, Anthropic)
  embeddings/              # sentence-transformer / remote-API / pure-python hashing backends
  vectorstore/             # sqlite+numpy (portable), Chroma, FAISS
  rag/                     # paragraph-aware chunker + MMR retriever
  agents/                  # single-responsibility AI agents (see below)
  services/                # orchestration entry points used by routes
  prompts/                 # prompt builders per task
  repositories/            # data access (documents, sessions, learning history)
  models/                  # get_db() + additive, idempotent migrations
  routes/                  # Flask blueprints (auth, teacher, student, documents)
models/                    # legacy generators used as the safe fallback path
utils/                     # text cleaning, difficulty classifier, session manager
templates/                 # Jinja2 templates (+ shared partials)
database/                  # SQLite storage + schema
```

### The agent pipeline (RAG mode)

When `AI_PIPELINE=rag`, MCQ generation is orchestrated through `core/services/mcq_pipeline.py`:

1. **Planner** — extracts intent and parameters (count, difficulty, topic).
2. **Document Processing** — extract → clean → chunk → metadata.
3. **Embedding** — content-hash-cached, batched, incremental vector indexing.
4. **Retriever** — MMR retrieval with document "spread" so large question sets draw from the whole document.
5. **Context Builder** — assembles retrieved chunks into a token-budgeted context.
6. **Question** — generates MCQs grounded in the context using Bloom's taxonomy.
7. **Quality Assurance** — deterministic structural validation (4 options, A/B/C/D, answer matches an option, no dupes).
8. **Difficulty** — lexical pre-filter + LLM grading, regenerating any mismatches.

Additional **learning-intelligence** agents run after tests: **Explanation** (grounded answer rationales), **Evaluation** (weak-topic detection persisted to the DB), and **Revision** (retrieval-grounded revision notes).

If any step fails, the pipeline falls back to the legacy single-shot generator, preserving the exact return shape.

### Extensible content generation

Beyond MCQs, `core/services/content_service.py` exposes a registry-driven spine (ingest → retrieve → context → prompt → JSON) that generates **flashcards, summaries, interview questions, coding questions, and topic explanations**. Adding a new content type is a two-line registry entry.

## Tech Stack

- **Backend**: Python 3, Flask, Flask-Session, Flask-WTF (CSRF)
- **Database**: SQLite3 (additive, idempotent migrations)
- **LLM providers**: pluggable — Groq, xAI/Grok, OpenAI, Together, Google Gemini, Anthropic (selected by config)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`), remote embedding APIs, or a pure-Python hashing fallback
- **Vector stores**: SQLite + NumPy (portable default), ChromaDB, or FAISS
- **Document parsing**: PDF, Word (`.docx`), and PowerPoint (`.pptx`)
- **PDF/CSV export**: ReportLab
- **Frontend**: HTML5, CSS3, vanilla JavaScript, Jinja2
- **Security**: Werkzeug password hashing, login lockout, CSRF protection, randomized option shuffling
- **Deployment**: Gunicorn (`Procfile`) and Vercel (`vercel.json`) ready

## Getting Started

### 1. Prerequisites
Python 3 and an API key for at least one supported LLM provider (Groq, xAI, OpenAI, Gemini, or Anthropic).

### 2. Installation

```bash
git clone https://github.com/dinu-vishruth/mcq.git
cd mcq
pip install -r requirements.txt
```

> Some dependencies (SentenceTransformers, ChromaDB, FAISS, torch) are heavy and are only needed for the local RAG backends. On constrained/serverless hosts the app automatically falls back to the pure-Python hashing embedder and the SQLite+NumPy vector store.

### 3. Configuration
Create a `.env` file in the project root. At minimum, provide an LLM key:

```dotenv
# --- LLM ---
LLM_PROVIDER=auto            # auto | groq | xai | openai | gemini | anthropic
LLM_API_KEY=your_key_here    # falls back to GROK_API_KEY if unset
LLM_MODEL=                   # falls back to GROK_MODEL if unset

# Provider-specific keys (only what you use)
GROK_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=

# --- Pipeline ---
AI_PIPELINE=legacy           # legacy (default) or rag

# --- RAG (only used when AI_PIPELINE=rag) ---
EMBEDDING_BACKEND=auto       # auto | sentence_transformer | remote | hashing
VECTOR_STORE=chroma          # chroma | sqlite | faiss
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=12
CONTEXT_MAX_CHARS=12000

# --- App ---
SECRET_KEY=                  # auto-generated if unset
```

With `LLM_PROVIDER=auto`, a key beginning with `gsk_` is treated as Groq; otherwise it is treated as xAI/Grok. The database schema (`database/schema.sql`) and all RAG tables are initialized automatically on first run.

### 4. Running the Application

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

Default seeded accounts (for quick local/demo use): `teacher` and `student`.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | random | Flask session secret |
| `LLM_PROVIDER` | `auto` | `auto`/`groq`/`xai`/`openai`/`gemini`/`anthropic` |
| `LLM_API_KEY` | `GROK_API_KEY` | Unified LLM key |
| `LLM_MODEL` | `GROK_MODEL` | Unified LLM model |
| `GROK_API_KEY` / `GROK_MODEL` | `""` / `grok-2-1212` | Legacy key/model |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` | `""` | Per-provider keys |
| `LLM_TIMEOUT` / `LLM_TEMPERATURE` | `45` / `0.3` | Request tuning |
| `AI_PIPELINE` | `legacy` | `legacy` or `rag` |
| `EMBEDDING_BACKEND` | `auto` | `auto`/`sentence_transformer`/`remote`/`hashing` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` / `EMBEDDING_BATCH_SIZE` | `all-MiniLM-L6-v2` / `384` / `64` | Embedding config |
| `VECTOR_STORE` | `chroma` (`sqlite` on Vercel) | `chroma`/`sqlite`/`faiss` |
| `CHROMA_PATH` | `chroma_db` (`/tmp/...` on Vercel) | Chroma persistence dir |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Chunking |
| `RETRIEVAL_TOP_K` / `CONTEXT_MAX_CHARS` | `12` / `12000` | Retrieval budget |

## Deployment

- **Gunicorn**: `gunicorn app:app` (see `Procfile`).
- **Vercel**: `vercel.json` targets `app.py`. On Vercel the app detects the read-only filesystem and automatically routes storage to `/tmp`, forces the SQLite+NumPy vector store, and enforces HTTPS.

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to open an issue or pull request.

---
*Built to simplify assessments and enhance AI-driven learning.*
