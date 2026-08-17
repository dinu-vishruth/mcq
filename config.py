# config.py
import os
import secrets

# Anchor the .env lookup to this file's directory rather than the process CWD.
# Bare load_dotenv() searches relative to the caller, so launching the server
# from anywhere but the project root (a systemd unit, an IDE run config, a
# `uvicorn` invoked from a parent dir) silently found no .env -- which surfaced
# downstream as "API Key is missing" even though the key was sitting in the file.
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
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
    if not DOTENV_LOADED and not os.path.exists(_ENV_PATH):
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
    # verification on the next — which surfaces as "CSRF session token missing".
    # Require a stable key in production; only fall back to a random one for
    # local dev (single long-lived process).
    if os.environ.get("VERCEL"):
        raise RuntimeError(
            "SECRET_KEY environment variable is required on Vercel. "
            "Set it in the project's Environment Variables so session cookies "
            "stay valid across serverless invocations."
        )
    SECRET_KEY = secrets.token_hex(32)
# --- Google sign-in (OAuth 2.0) -------------------------------------------
# Create these at https://console.cloud.google.com/apis/credentials by making an
# "OAuth client ID" of type "Web application", then add the callback URL to its
# Authorized redirect URIs:
#   local:  http://localhost:8000/auth/google/callback
#   Vercel: https://<your-domain>/auth/google/callback
# Sign-in is simply hidden when these are unset, so the app runs fine without
# them and the Google button only appears once it can actually work.
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
# Optional explicit override. When empty the callback URL is derived from the
# incoming request, which keeps preview deployments working without new config.
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "")
GOOGLE_OAUTH_ENABLED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

# Accept GROQ_* and GROK_* interchangeably. These are two different companies --
# Groq (keys look like `gsk_...`) is the inference host, Grok is xAI's model --
# and the one-letter difference is easy to mistype, so a key set under either
# spelling is honoured instead of silently reading as "no key at all".
GROK_API_KEY = env_str("GROK_API_KEY", "GROQ_API_KEY", "XAI_API_KEY")
GROK_MODEL = env_str("GROK_MODEL", "GROQ_MODEL", default="grok-2-1212")
DB_PATH = "/tmp/mcq.db" if os.environ.get("VERCEL") else "database/mcq.db"
UPLOAD_FOLDER = "/tmp/uploads" if os.environ.get("VERCEL") else "uploads"
ALLOWED_EXT = {"pdf", "docx", "txt", "pptx"}

# ---------------------------------------------------------------------------
# Agentic AI + RAG configuration (Phase 1+). All additive; defaults preserve
# the existing behaviour so nothing changes until a feature flag is flipped.
# ---------------------------------------------------------------------------
IS_VERCEL = bool(os.environ.get("VERCEL"))

# --- LLM provider abstraction --------------------------------------------
# One of: "auto" | "groq" | "xai" | "openai" | "gemini" | "anthropic".
# "auto" keeps the legacy behaviour: detect Groq vs xAI from the key prefix.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()

# Unified key/model. Falls back through every historical spelling, and a blank
# LLM_API_KEY no longer shadows a working GROK_API_KEY/GROQ_API_KEY.
LLM_API_KEY = env_str("LLM_API_KEY", default=GROK_API_KEY)
LLM_MODEL = env_str("LLM_MODEL", default=GROK_MODEL)

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
    "Quiz generation is temporarily unavailable because the AI service isn't "
    "configured. Your document was not lost - please try again shortly, or "
    "contact whoever administers this site."
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

# Output token budget for MCQ generation. Providers apply a modest default cap
# (often ~1-4k) which truncates the JSON array mid-object on larger requests --
# that's what silently turned "25 questions" into a handful. We request a budget
# scaled to the batch size, floored so small batches still get room to breathe.
# ~330 tokens covers one question with 4 options plus bloom/source_hint metadata.
LLM_TOKENS_PER_QUESTION = int(os.getenv("LLM_TOKENS_PER_QUESTION", "330"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "16000"))


def mcq_token_budget(num_questions: int) -> int:
    """Output token ceiling for a batch of `num_questions` MCQs."""
    return min(LLM_MAX_OUTPUT_TOKENS, max(1500, num_questions * LLM_TOKENS_PER_QUESTION + 400))

# Wall-clock budget for the whole RAG MCQ pipeline. Must stay under the serverless
# function limit (vercel.json maxDuration=60) so generation is never killed
# mid-flight; the pipeline stops retrying once this budget (minus one LLM_TIMEOUT
# of headroom) is exhausted and returns whatever it has collected.
PIPELINE_DEADLINE_SECONDS = int(os.getenv("PIPELINE_DEADLINE_SECONDS", "55"))

# --- Upload/generation throttle (app-level, protects the LLM from bursts) --
# Minimum seconds between two MCQ-generation requests, and the max generations
# allowed per 5-minute window. Set the cooldown to 0 to disable it entirely.
# Deliberately lenient by default — a single user iterating on their own
# material shouldn't be blocked; this only guards against runaway loops.
GENERATION_COOLDOWN_SECONDS = int(os.getenv("GENERATION_COOLDOWN_SECONDS", "3"))
GENERATION_MAX_PER_5MIN = int(os.getenv("GENERATION_MAX_PER_5MIN", "10"))

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
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  # MiniLM native dim
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# --- Vector store ---------------------------------------------------------
# "chroma" (default) -> ChromaDB persistent client
# "sqlite"           -> vectors as BLOBs in mcq.db + numpy cosine (Vercel-safe)
# "faiss"            -> FAISS index (opt-in; needs a compatible wheel)
# On Vercel the read-only FS forces the sqlite store regardless of this value.
VECTOR_STORE = os.getenv("VECTOR_STORE", "sqlite" if IS_VERCEL else "chroma").strip().lower()
CHROMA_PATH = os.getenv("CHROMA_PATH", "/tmp/chroma_db" if IS_VERCEL else "chroma_db")

# --- Chunking -------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))       # chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # char overlap
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "12"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "12000"))
