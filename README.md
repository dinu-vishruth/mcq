# Examly — MCQ Generator

A learning application that turns uploaded study material into configurable quizzes. It combines a FastAPI backend, React and TypeScript screens, a personal resource library, and an optional retrieval-augmented generation (RAG) pipeline.

This README describes the code in this branch. A deployed application's behavior depends on the deployed commit, environment variables, and configured services.

## Features

- Upload PDF, DOCX, PPTX, or TXT files, or paste study text.
- Save and index resources in a personal Library; repeated saves are deduplicated per owner.
- Configure quizzes with 1–30 questions, easy/medium/hard difficulty, an optional topic, and a timer from 30 seconds to 180 minutes.
- Take timed quizzes, view results, identify weak topics, and generate revision practice.
- Track learning progress, preferences, XP, streaks, and achievements.
- Sign in with a password or optionally Google/GitHub OAuth. Accounts use a unified user experience; demo names confer no special privileges.
- Export account data, update profile details, and delete owned account records.
- Use Groq, xAI, OpenAI, Gemini, or Anthropic through a shared provider interface.

## How quiz creation works

1. Upload material through **Make Quiz & Test**, or add a resource to the Library.
2. The server extracts and cleans the text, chunks it, and indexes it before responding. Uploading does not generate a quiz.
3. A successful quiz upload redirects to `/knowledge?doc=<id>` for configuration. The uploaded source remains in the Library.
4. Choose the quiz settings and request generation through `/api/practice/generate`.
5. The server creates a quiz session; the learner opens it and submits answers for server-side grading.
6. Results feed the progress and weak-topic views.

Failed indexing is reported to the user. Unfinished resources older than the ingestion timeout are exposed as failed rather than remaining indefinitely pending. Generation can return fewer questions when its retry or time budget is exhausted.

## Recent updates

- Correct answers are excluded from quiz-page bootstrap data. Grading uses answers stored on the server.
- Demo accounts require explicit local opt-in and receive random passwords; deployed demo seeding is disabled.
- Renaming an account retains ownership of resources and quiz sessions.
- Login failures are tracked by `(username, IP)` pair. Passwords retain their exact leading and trailing whitespace.
- Account deletion covers owned database records and vector cleanup while preserving other learners' results and shared embeddings. Export includes sessions and document metadata.
- Regression coverage verifies upload → configuration → generation, timer validation, and playable quiz sessions.
- CI checks the backend, frontend typecheck/build, and freshness of committed frontend assets. The inactive Flask route tree has been removed.
- Retired Groq `groq/compound` and `groq/compound-mini` overrides resolve to the existing `openai/gpt-oss-20b` default.

## Local setup

The CI configuration uses Python 3.13 and Node.js 24.14.0. Node is needed when rebuilding the frontend; committed assets let the backend serve the UI without a Node process.

From the repository root, using PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` with your own session secret and provider key. A minimal Groq configuration is:

```dotenv
SECRET_KEY=replace_with_a_long_random_secret
GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
AI_PIPELINE=legacy
EMBEDDING_BACKEND=hashing
VECTOR_STORE=sqlite
```

Generate a session secret with:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Start the application:

```powershell
python app.py
```

Open <http://127.0.0.1:8000/> and create an account. New passwords require at least 10 characters and two of lowercase letters, uppercase letters, numbers, or symbols. Existing passwords remain usable at login.

On macOS/Linux, activate the environment with `source .venv/bin/activate` and copy the example using `cp .env.example .env`; the remaining Python commands are the same.

## Frontend development

```powershell
cd frontend
npm ci
npm run typecheck
npm run build
cd ..
```

Vite writes stable filenames to `static/dist/`, which FastAPI serves through the HTML templates. Commit rebuilt assets with frontend source changes: Vercel's configured Python build does not run a Node build. `npm run dev` runs a watching build; use the backend URL to view the application.

## Configuration

See [.env.example](.env.example) for examples and [config.py](config.py) for all defaults.

| Setting | Behavior |
| --- | --- |
| `SECRET_KEY` | Signs sessions; required when `VERCEL` or `RENDER` is set. Use a stable value across instances. |
| `LLM_PROVIDER` | `auto` (default), `groq`, `xai`, `openai`, `gemini`, or `anthropic`. |
| `LLM_API_KEY` | Unified credential; historical `GROK_API_KEY`, `GROQ_API_KEY`, and `XAI_API_KEY` spellings are also accepted. Other providers support `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `ANTHROPIC_API_KEY`. |
| `LLM_MODEL` | Model override. Precedence: `LLM_MODEL`, then `GROK_MODEL`, then `GROQ_MODEL`, then the resolved provider's default. Blank model/key aliases are treated as unset. |
| `LLM_REASONING_EFFORT` | Defaults to `low`; sent for GPT-OSS models to limit reasoning token overhead. |
| `AI_PIPELINE` | `legacy` by default; `rag` enables the multi-stage verification pipeline. |
| `EMBEDDING_BACKEND` | `auto`, `hashing`, `sentence_transformer`, or `remote`. Auto uses SentenceTransformers when available, otherwise hashing. |
| `VECTOR_STORE` | `chroma` locally by default; `sqlite` on deployed hosts. Optional `faiss` and `pgvector` adapters exist. Vercel forces the SQLite-backed store. |
| `DATA_DIR` | Writable state root: repository directory locally, `/tmp` on Vercel. |
| `DB_PATH` | SQLite path; defaults to `<DATA_DIR>/database/mcq.db`. |
| `DATABASE_URL` | Selects the Postgres database adapter when set. Provisioning and data migration are separate operational steps. |
| `UPLOAD_BACKEND` | `local` by default; `s3` enables optional S3-compatible upload storage. |
| `GENERATION_BACKEND` | `auto`, `inline`, or `rq`. Auto chooses RQ only when Redis is configured and the optional packages are installed. |
| `REDIS_URL` | Redis connection for queued generation. |
| `GENERATION_QUEUE_NAME` | Queue name; defaults to `mcq`. |
| `GENERATION_JOB_DEADLINE_SECONDS` | Queued pipeline budget; defaults to 900 seconds. |
| `PIPELINE_DEADLINE_SECONDS` | Inline pipeline budget; defaults to 55 seconds on Vercel and 300 elsewhere. |
| `MAX_UPLOAD_BYTES` / `MAX_PDF_PAGES` | Default upload limit is 4.5 MiB and PDF limit is 300 pages. |
| `GENERATION_COOLDOWN_SECONDS` / `GENERATION_MAX_PER_5MIN` | Default generation throttle: 3 seconds between requests and 10 generations per five minutes. |
| `SEED_DEMO_USERS` | Unset/disabled by default. `1` enables local demo seeding with random passwords printed once. |

Retrieval defaults are `CHUNK_SIZE=1500`, `CHUNK_OVERLAP=80`, `RETRIEVAL_TOP_K=8`, and `CONTEXT_MAX_CHARS=12000`.

### Groq model errors

`groq/compound` and `groq/compound-mini` were retired. If an older deployment returns `model_not_found`, set:

```dotenv
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
```

Restart the local server or redeploy the hosted application after changing configuration. `LLM_MODEL` takes precedence over `GROQ_MODEL`, so changing only the latter may leave a stale override active. The registry migrates the retired Compound IDs in the updated code; it does not guarantee access to every model for every provider account.

### Optional OAuth

Set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`, or `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`. Register the corresponding callback with the provider:

- `http://localhost:8000/auth/google/callback`
- `http://localhost:8000/auth/github/callback`

Use the application's actual HTTPS domain for hosted callbacks. `GOOGLE_REDIRECT_URI` and `GITHUB_REDIRECT_URI` can pin the callback; otherwise it is derived from the request. Use a consistent local hostname for login and callback.

## Generation architecture

The default `legacy` mode uses the existing generator without the full verification chain. Generating from a Library resource still retrieves budgeted source context.

With `AI_PIPELINE=rag`, the pipeline retrieves document chunks, builds bounded context, generates questions, applies structural QA, optionally verifies grounding, and checks difficulty. It deduplicates questions and attempts to fill a shortfall within the time budget. Selected failure paths fall back to legacy generation; missing credentials or unavailable provider models can still prevent generation.

Embeddings support hashing, local SentenceTransformers, and remote APIs. Chroma, SQLite/NumPy, FAISS, and pgvector adapters are available. SentenceTransformers, Chroma, and FAISS require optional dependencies; they are not bundled in the lightweight requirements.

## Deployment and persistence

[vercel.json](vercel.json) routes requests to the FastAPI application exported by [app.py](app.py). [Procfile](Procfile) defines a two-worker Uvicorn process for a persistent host. Configure a stable `SECRET_KEY` and provider credentials on the hosting platform.

Default storage locations:

| State | Local | Vercel |
| --- | --- | --- |
| SQLite database | `database/mcq.db` | `/tmp/database/mcq.db` |
| Original uploads | `uploads/` | `/tmp/uploads/` |
| Reports | Under `DATA_DIR` | Under `/tmp` |

Vercel's `/tmp` state is temporary and instance-local. Accounts, preferences, resources, results, and login lockouts stored there do not have durable multi-instance persistence. Database adapters and optional storage hooks are available, but they do not provision services or migrate existing production data.

For a persistent host, point `DATA_DIR` at the mounted disk. Postgres can be selected with `DATABASE_URL`; original-file storage can use `UPLOAD_BACKEND=s3` and `S3_UPLOAD_BUCKET`, `S3_UPLOAD_REGION`, `S3_UPLOAD_PREFIX`, and optionally `S3_UPLOAD_ENDPOINT_URL` (requires `boto3`). Vercel currently forces the SQLite-backed vector store even if another vector backend is requested.

Account deletion cleans database records and vectors; physical original-upload object cleanup is not included. External vector deletion cannot share a distributed transaction with the account database.

Health endpoints are `/health`, `/healthz`, `/readyz`, and `/api/health`. A successful probe does not establish that OAuth, provider model access, or durable storage works end to end. Deploy the intended commit and verify those flows separately.

## Optional background worker

Install the optional queue dependencies:

```powershell
python -m pip install redis rq
```

Configure these values for both the web process and a separately running worker, with access to the same application state:

```dotenv
GENERATION_BACKEND=rq
REDIS_URL=redis://localhost:6379/0
GENERATION_QUEUE_NAME=mcq
```

Start the worker using `python worker.py`. Inline generation returns `{ok, session_key, count}`; queued generation returns `{ok, job_id, status}`. Job state is available through `GET /api/jobs/{id}` and events through `GET /api/jobs/{id}/events`. A durable production queue and worker deployment require operational setup beyond enabling these settings.

## Testing and CI

From an activated Python environment:

```powershell
python -m pip install pytest ruff
python -m pytest -q tests
ruff check .
```

Tests use fake provider responses and cover upload/indexing, resource ownership, model selection, quiz generation and grading, login security, account deletion/export, retrieval, verification, background jobs, storage, and observability. Tests marked `live` are excluded by default; `python -m pytest -m live` explicitly enables credential-dependent provider calls.

[GitHub Actions](.github/workflows/ci.yml) runs offline backend tests and Ruff with demo seeding disabled. Its frontend job uses locked dependencies, typechecks, rebuilds, and rejects tracked changes or new uncommitted files in `static/dist/`.

The evaluation harness can run offline using `python evals/run_eval.py --dry-run`. Running without `--dry-run` calls the configured provider. Results are written to `evals/results/latest.json` and `latest.md`.

## Observability

Structured application events include request IDs, stage timings, and provider token usage. Configure `LOG_LEVEL`, `LOG_FORMAT`, and optionally `OBSERVABILITY_BACKEND` or `OBSERVABILITY_TRACE_HOOK`. See [docs/observability.md](docs/observability.md) for tracing adapters and event details.

## Repository guide

```text
app.py                    FastAPI entry point
fastapi_app/              Routers, authentication, middleware, database bootstrap
core/llm/                 Provider adapters and model resolution
core/agents/              Retrieval, generation, and verification stages
core/services/            Ingestion, generation, learning, and account operations
core/repositories/        Data access
core/embeddings/          Embedding backends
core/vectorstore/         Vector storage adapters
core/storage/             Local and optional S3-compatible upload storage
core/models/              Schema and migration helpers
models/                   Legacy generation implementation
frontend/                 React, TypeScript, Vite, Tailwind sources
static/dist/              Committed frontend build output
templates/                Server-rendered pages and React bootstrap shells
database/migrations/      Numbered database migrations
tests/                    Offline regressions and opt-in live tests
evals/                    Evaluation harness and dataset
worker.py                 Optional RQ worker entry point
```

The earlier [engineering walkthrough](docs/project_explanation.md) is a historical snapshot and predates the latest security fixes. Current handlers, configuration, and tests define present behavior.

Keep `.env`, API keys, local databases, uploaded material, and generated private reports out of Git.
