# config.py
import os
import secrets

_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Deployment platform ----------------------------------------------------
# Each host injects its own marker variable. Read BEFORE .env loading so a stray
# VERCEL/RENDER entry in a local .env cannot put a laptop into production mode.
IS_VERCEL = bool(os.environ.get("VERCEL"))
IS_RENDER = bool(os.environ.get("RENDER"))
#: Serverless: read-only filesystem, so all writable state has to live in /tmp.
IS_SERVERLESS = IS_VERCEL
#: Any hosted environment: separate processes and restarts, so CWD-relative paths
#: and per-boot random secrets are not acceptable.
IS_DEPLOYED = IS_VERCEL or IS_RENDER

# Anchor the .env lookup to this file's directory rather than the process CWD.
# Bare load_dotenv() searches relative to the caller, so launching the server
# from anywhere but the project root (a systemd unit, an IDE run config, a
# `uvicorn` invoked from a parent dir) silently found no .env -- which surfaced
# downstream as "API Key is missing" even though the key was sitting in the file.
_ENV_PATH = os.path.join(_ROOT, ".env")
DOTENV_LOADED = False
DOTENV_ERROR = ""
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - depends on the install
    # Previously swallowed in silence, so a missing python-dotenv looked
    # identical to a missing key. Recorded so startup can say which it is.
    DOTENV_ERROR = (
        "python-dotenv is not installed, so .env was not read. "
        "Install it (pip install -r requirements.txt) or set real environment variables."
    )
else:
    # override=False: real environment variables continue to win over .env,
    # which is what deployments rely on.
    DOTENV_LOADED = load_dotenv(_ENV_PATH, override=False)
    # A deployed host deliberately has no .env; its config is the dashboard's env
    # vars, so warning about the missing file is noise in the logs.
    if not DOTENV_LOADED and not os.path.exists(_ENV_PATH) and not IS_DEPLOYED:
        DOTENV_ERROR = f"No .env file at {_ENV_PATH} (fine if you set environment variables directly)."


def env_str(*names: str, default: str = "") -> str:
    """First non-blank value among `names`, else `default`.

    os.getenv(name, fallback) only falls back when the variable is *absent*. A
    variable that exists but is empty -- an env var added in a hosting dashboard
    with a blank value, or an empty shell export -- returned "" and shadowed a
    perfectly good fallback. Blank is treated as unset here, and surrounding
    whitespace/quotes are stripped so a copy-pasted `KEY="abc"` still works.
    """
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1].strip()
        if value:
            return value
    return default

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    # On Vercel each cold start is a fresh process; a random key here would be
    # different per instance, so session cookies signed by one invocation fail
    # verification on the next -- which surfaces as "CSRF session token missing".
    # Render restarts the service the same way, so the key has to be stable there
    # too: sessions would otherwise be dropped on every deploy. Require a fixed
    # key anywhere the app runs as production; only fall back to a random one for
    # local dev (single long-lived process).
    if IS_DEPLOYED:
        raise RuntimeError(
            "SECRET_KEY environment variable is required when deployed. "
            "Set it in the host's Environment Variables so session cookies "
            "stay valid across restarts and instances."
        )
    SECRET_KEY = secrets.token_hex(32)
# --- OAuth sign-in (Google + GitHub) ---------------------------------------
# Each provider is independently enabled by setting its client id/secret. Leave
# them blank to hide the corresponding button on the login page.
#
# Google: create an "OAuth client ID" (Web application) at
#   https://console.cloud.google.com/apis/credentials
# and add the callback URL to its Authorized redirect URIs:
#   local:  http://localhost:8000/auth/google/callback
#   hosted: https://<your-domain>/auth/google/callback
GOOGLE_CLIENT_ID = env_str("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = env_str("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = env_str("GOOGLE_REDIRECT_URI")
GOOGLE_OAUTH_ENABLED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

# GitHub: register an OAuth App at https://github.com/settings/developers and
# set the callback URL to:
#   local:  http://localhost:8000/auth/github/callback
#   hosted: https://<your-domain>/auth/github/callback
GITHUB_CLIENT_ID = env_str("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = env_str("GITHUB_CLIENT_SECRET")
GITHUB_REDIRECT_URI = env_str("GITHUB_REDIRECT_URI")
GITHUB_OAUTH_ENABLED = bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET)

# Accept GROQ_* and GROK_* interchangeably. These are two different companies --
# Groq (keys look like `gsk_...`) is the inference host, Grok is xAI's model --
# and the one-letter difference is easy to mistype, so a key set under either
# spelling is honoured instead of silently reading as "no key at all".
GROK_API_KEY = env_str("GROK_API_KEY", "GROQ_API_KEY", "XAI_API_KEY")
# Legacy per-host model variable. Empty by default: each provider now has its
# own default model (PROVIDER_DEFAULT_MODELS below), so an unset model resolves
# to something the resolved provider can actually serve. A stale
# GROK_MODEL=grok-... can no longer be sent to Groq (it used to 404).
GROK_MODEL = env_str("GROK_MODEL", "GROQ_MODEL", default="")

#: Model used when the operator hasn't pinned one per provider. Registry
#: resolution starts here; a configured model only survives if it plausibly
#: belongs to the resolved provider (see core/llm/registry.py).
PROVIDER_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-20b",
    "xai": "grok-2-1212",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "anthropic": "claude-sonnet-5",
}


def default_model_for(provider: str) -> str:
    """The default model for `provider`; empty string for unknown providers."""
    return PROVIDER_DEFAULT_MODELS.get((provider or "").strip().lower(), "")
# --- Writable state ---------------------------------------------------------
# Everything the app creates at runtime hangs off one anchor. DATA_DIR is the
# only variable a host needs to persist data: point it at a mounted disk and
# every path below follows. It defaults to the project directory locally, which
# keeps the existing database/mcq.db in place, and to /tmp on serverless where
# nothing else is writable. On Render, set DATA_DIR to the Disk mount (/var/data)
# or the writes land in the ephemeral source checkout instead of the disk.
DATA_DIR = env_str("DATA_DIR", default="/tmp" if IS_SERVERLESS else _ROOT)
DB_PATH = env_str("DB_PATH", default=os.path.join(DATA_DIR, "database", "mcq.db"))
UPLOAD_FOLDER = env_str("UPLOAD_FOLDER", default=os.path.join(DATA_DIR, "uploads"))
REPORTS_DIR = env_str("REPORTS_DIR", default=os.path.join(DATA_DIR, "reports"))
ALLOWED_EXT = {"pdf", "docx", "txt", "pptx"}

# --- Persistent database (production) ---------------------------------------
# When DATABASE_URL is set (e.g. postgresql://user:pass@host/db, the format
# every managed Postgres provider hands you), the repository layer talks to
# Postgres through core/models/pg_adapter.py instead of SQLite. Left unset, the
# app keeps its local-dev SQLite file at DB_PATH. DB_PATH is ignored while a
# DATABASE_URL is present.
DATABASE_URL = env_str("DATABASE_URL", default="")

#: Ceiling on a buffered request body. Vercel enforced ~4.5MB for us before our
#: own code ran; a persistent host applies no such cap, and CSRFMiddleware has to
#: hold the whole body in memory to read the token and replay it downstream. On a
#: 512MB instance an unbounded buffer is an outage, so the app enforces its own.
#: Keep this above MAX_UPLOAD_BYTES.
MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", str(8 * 1024 * 1024)))

#: Upload ceilings, enforced before any parsing work happens. The request-body
#: limit above bounds the raw HTTP body; these bound what the document pipeline
#: will accept so a small-but-pathological file (thousands of PDF pages) can't
#: monopolise the instance.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(int(4.5 * 1024 * 1024))))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "300"))

# --- Demo accounts ----------------------------------------------------------
# The seeded "demo_user" account (DemoPass123) is a local convenience and a
# liability anywhere public. On a hosted deployment it is only created when
# SEED_DEMO_USERS is explicitly true; local development keeps the historical
# behaviour so the demo flow still works out of the box.
SEED_DEMO_USERS = os.getenv("SEED_DEMO_USERS", "").strip().lower() in ("1", "true", "yes", "on")

# --- Observability ----------------------------------------------------------
# JSON is the default because hosted log stores are queried field-by-field; set
# LOG_FORMAT=text for a human-readable local stream. No prompt/document text is
# logged either way -- see docs/observability.md.
LOG_LEVEL = env_str("LOG_LEVEL", default="INFO").upper()
LOG_FORMAT = env_str("LOG_FORMAT", default="json").lower()
#: Optional "module:function" callable receiving every structured event. This is
#: the integration point for LangSmith/Langfuse/OTLP without adding an SDK here.
OBSERVABILITY_TRACE_HOOK = env_str("OBSERVABILITY_TRACE_HOOK", default="")

# Backend selector for tracing (auto | langsmith | langfuse | hook | none). In
# "auto" mode the sink is picked from whichever credentials are present.
OBSERVABILITY_BACKEND = env_str("OBSERVABILITY_BACKEND", default="auto")

# ---------------------------------------------------------------------------
# Agentic AI + RAG configuration (Phase 1+). All additive; defaults preserve
# the existing behaviour so nothing changes until a feature flag is flipped.
# ---------------------------------------------------------------------------

# --- LLM provider abstraction --------------------------------------------
# One of: "auto" | "groq" | "xai" | "openai" | "gemini" | "anthropic".
# "auto" keeps the legacy behaviour: detect Groq vs xAI from the key prefix.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()

# Unified key/model. Falls back through every historical spelling, and a blank
# LLM_API_KEY no longer shadows a working GROK_API_KEY/GROQ_API_KEY.
LLM_API_KEY = env_str("LLM_API_KEY", default=GROK_API_KEY)
# Model precedence: LLM_MODEL, then the legacy GROK_MODEL/GROQ_MODEL, then the
# resolved provider's own default. See core/llm/registry.py.
LLM_MODEL = env_str("LLM_MODEL", "GROK_MODEL", "GROQ_MODEL", default="")

# Per-provider key overrides (optional; only used when explicitly selected).
OPENAI_API_KEY = env_str("OPENAI_API_KEY")
GEMINI_API_KEY = env_str("GEMINI_API_KEY")
ANTHROPIC_API_KEY = env_str("ANTHROPIC_API_KEY")


def llm_key_present() -> bool:
    """True when some usable LLM credential is configured."""
    return bool(LLM_API_KEY or OPENAI_API_KEY or GEMINI_API_KEY or ANTHROPIC_API_KEY)


#: Shown to end users when generation can't run for lack of a key. Deliberately
#: free of variable names and file paths: this renders in the browser, and
#: server configuration detail should not leak to learners. The operator-facing
#: detail goes to the logs via missing_key_message().
USER_FACING_AI_UNAVAILABLE = (
    "Quiz generation is unavailable because the AI service isn't configured. "
    "The site administrator needs to configure it before quizzes can be generated. "
    "Your saved Library resources are unaffected."
)


def missing_key_message() -> str:
    """Actionable 'no key' message naming what was actually checked.

    OPERATOR-FACING (logs only) -- it includes the .env path and variable names.
    Use USER_FACING_AI_UNAVAILABLE for anything rendered in a browser.

    The old text named a single variable (GROK_API_KEY) that the loader did not
    even read under that spelling in every path, sending people to re-check a
    file that was already correct. This reports where config looked.
    """
    parts = [
        "No LLM API key found. Set GROQ_API_KEY (or LLM_API_KEY) to a Groq key "
        "starting with 'gsk_', or GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY."
    ]
    if DOTENV_ERROR:
        parts.append(DOTENV_ERROR)
    else:
        parts.append(f"Checked environment variables and {_ENV_PATH}.")
    if os.getenv("LLM_API_KEY") is not None and not os.getenv("LLM_API_KEY", "").strip():
        # ASCII only: this string is printed to the console at startup, and the
        # default Windows code page (cp1252) mangles an em dash into a '?'.
        parts.append("Note: LLM_API_KEY is set but empty - remove it or give it a value.")
    return " ".join(parts)

LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "45"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# gpt-oss on Groq emits hidden chain-of-thought before the JSON answer. Left at
# the provider default it exhausts max_tokens mid-reasoning, so the JSON never
# starts and the provider fails validation with an *empty* failed_generation --
# nothing recoverable, and the learner sees a hard error. "low" cuts that
# overhead from ~5,900 tokens to a handful while still returning full batches.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low").strip()

# Output token budget for MCQ generation. Providers apply a modest default cap
# (often ~1-4k) which truncates the JSON array mid-object on larger requests --
# that's what silently turned "25 questions" into a handful. We request a budget
# scaled to the batch size, floored so small batches still get room to breathe.
# ~330 tokens covers one question with 4 options plus bloom/source_hint metadata.
# Sized for the JSON answer only: it assumes reasoning overhead is capped by
# LLM_REASONING_EFFORT, without which reasoning tokens eat this budget first.
LLM_TOKENS_PER_QUESTION = int(os.getenv("LLM_TOKENS_PER_QUESTION", "330"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "16000"))


def mcq_token_budget(num_questions: int) -> int:
    """Output token ceiling for a batch of `num_questions` MCQs."""
    return min(LLM_MAX_OUTPUT_TOKENS, max(1500, num_questions * LLM_TOKENS_PER_QUESTION + 400))

# Wall-clock budget for the whole RAG MCQ pipeline. On serverless this has to
# stay under the function limit (vercel.json maxDuration=60) so generation is
# protected by a shared remaining-time budget. The pipeline stops retrying
# when that budget is exhausted and returns whatever it has
# collected. A persistent host has no such cap, so larger quiz sets are allowed
# the time they actually need.
PIPELINE_DEADLINE_SECONDS = int(os.getenv("PIPELINE_DEADLINE_SECONDS", "50" if IS_SERVERLESS else "300"))
RESULT_ENRICHMENT_DEADLINE_SECONDS = int(os.getenv("RESULT_ENRICHMENT_DEADLINE_SECONDS", "12"))

# --- Upload/generation throttle (app-level, protects the LLM from bursts) --# Minimum seconds between two MCQ-generation requests, and the max generations
# allowed per 5-minute window. Set the cooldown to 0 to disable it entirely.
# Deliberately lenient by default — a single user iterating on their own
# material shouldn't be blocked; this only guards against runaway loops.
GENERATION_COOLDOWN_SECONDS = int(os.getenv("GENERATION_COOLDOWN_SECONDS", "3"))
GENERATION_MAX_PER_5MIN = int(os.getenv("GENERATION_MAX_PER_5MIN", "10"))

# --- Background generation jobs ---------------------------------------------
# "inline" -> generate inside the HTTP request (the historical behaviour; the
#             default so local dev and Vercel lose nothing)
# "rq"     -> hand generation to a Redis Queue worker; POST /api/practice/
#             generate returns a job_id, results arrive via GET /api/jobs/{id}
# "auto"   -> "rq" when REDIS_URL is set and the redis+rq packages import,
#             else "inline".
GENERATION_BACKEND = env_str("GENERATION_BACKEND", default="auto").strip().lower()
REDIS_URL = env_str("REDIS_URL", default="")
QUEUE_NAME = env_str("GENERATION_QUEUE_NAME", default="mcq")
# Per-job wall-clock budget, in seconds. In queue mode the serverless 60s cap no
# longer applies -- the worker is a long-lived process -- so the deadline is a
# job setting the caller can pass on top of the global default below.
GENERATION_JOB_DEADLINE_SECONDS = int(os.getenv("GENERATION_JOB_DEADLINE_SECONDS", "900"))

# --- AI pipeline selection ------------------------------------------------
# "legacy"  -> single-shot generate_mcqs (current behaviour, the safe default)
# "rag"     -> retrieval-augmented agent pipeline (enabled in Phase 4)
AI_PIPELINE = os.getenv("AI_PIPELINE", "legacy").strip().lower()

# --- Generation & verification quality (RAG pipeline only) ----------------
# Generation runs at a low temperature so questions stay faithful to the
# retrieved context instead of drifting into the model's own memory. Kept
# separate from LLM_TEMPERATURE (which the legacy path and other agents use).
LLM_GENERATION_TEMPERATURE = float(os.getenv("LLM_GENERATION_TEMPERATURE", "0.1"))

# Context Validation: before generating, confirm the retrieved context actually
# supports grounded questions. When it confidently reports the context is
# insufficient, the pipeline falls back to the fuller-text legacy path rather
# than let the generator invent facts. Fail-open: any error => treated as
# sufficient so generation is never blocked by a flaky check.
CONTEXT_VALIDATION_ENABLED = os.getenv("CONTEXT_VALIDATION_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

# Fact Verification: after generation, an independent agent checks each MCQ
# against the context (answer supported, distractors wrong, one correct answer,
# unambiguous, explanation matches). Rejected questions are regenerated by the
# existing top-up loop. Fail-open: any error => questions pass through unchanged.
FACT_VERIFICATION_ENABLED = os.getenv("FACT_VERIFICATION_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

# Minimum model self-reported confidence [0,1] for a verified MCQ to be kept.
FACT_VERIFICATION_MIN_CONFIDENCE = float(os.getenv("FACT_VERIFICATION_MIN_CONFIDENCE", "0.6"))

# --- LLM rate-limit retry --------------------------------------------------
# Providers (especially free tiers) return HTTP 429 on brief bursts. A single
# generation fires several calls in quick succession (top-up loop + context
# validation + fact verification), so a lone 429 shouldn't abort the whole
# request. On 429 the LLM layer retries with exponential backoff, honoring the
# provider's Retry-After header when present.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1.5"))  # base seconds, doubled each retry
LLM_RETRY_MAX_WAIT = float(os.getenv("LLM_RETRY_MAX_WAIT", "8"))  # cap per individual wait

# Answer Evaluation: below this confidence the grader is asked to re-evaluate
# once (per the brief's failure-handling rule).
ANSWER_EVAL_REEVAL_CONFIDENCE = float(os.getenv("ANSWER_EVAL_REEVAL_CONFIDENCE", "0.7"))

# --- Embeddings -----------------------------------------------------------
# "auto"                -> SentenceTransformer if importable, else hashing
# "sentence_transformer"-> force local ST model (never available on Vercel)
# "remote"              -> OpenAI/Gemini embedding endpoint
# "hashing"             -> pure-Python fallback, zero heavy deps
#
# Keep the default on the faster side for uploaded-resource ingestion: the
# heavy embedding model is the main source of latency during save/index, and a
# larger chunk size significantly reduces work while staying retrieval-friendly.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  # MiniLM native dim
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))

# --- Vector store ---------------------------------------------------------
# "chroma" (default locally) -> ChromaDB persistent client
# "sqlite"           -> vectors as BLOBs in mcq.db + numpy cosine (always safe)
# "faiss"            -> FAISS index (opt-in; needs a compatible wheel)
# "pgvector"         -> Postgres + pgvector column; requires DATABASE_URL and a
#                       server with the pgvector extension. Falls back to the
#                       sqlite store with a warning otherwise.
# Vercel's read-only FS forces sqlite in the registry regardless of this value.
# Deployed hosts default to sqlite too: chromadb is deliberately not in
# requirements.txt (bundle size), so a "chroma" default would only ever reach
# the registry's degrade-with-a-warning path and hide that the store isn't the
# one the config claims.
VECTOR_STORE = os.getenv("VECTOR_STORE", "sqlite" if IS_DEPLOYED else "chroma").strip().lower()
CHROMA_PATH = env_str("CHROMA_PATH", default=os.path.join(DATA_DIR, "chroma_db"))

# --- Uploaded-file storage -------------------------------------------------
# "local"  -> UPLOAD_FOLDER on disk (dev / single-host deployments)
# "s3"     -> any S3-compatible object store (AWS S3, R2, MinIO). Requires the
#             optional boto3 dependency.
# "auto"   -> "s3" when an S3_UPLOAD_BUCKET is configured and boto3 is
#             importable, else "local".
UPLOAD_BACKEND = env_str("UPLOAD_BACKEND", default="auto").strip().lower()
S3_UPLOAD_BUCKET = env_str("S3_UPLOAD_BUCKET")
S3_UPLOAD_PREFIX = env_str("S3_UPLOAD_PREFIX", default="uploads")
# Endpoint URL for non-AWS S3-compatible stores (R2/MinIO); empty = AWS itself.
S3_UPLOAD_ENDPOINT = env_str("S3_UPLOAD_ENDPOINT")

# --- Chunking -------------------------------------------------------------
# Larger chunks mean fewer embeddings and faster indexing. The overlap stays
# small enough to preserve continuity without creating a large embedding fan-out.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))      # chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))   # char overlap
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "12000"))
