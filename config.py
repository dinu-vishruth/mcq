# config.py
import os
import secrets
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-2-1212")
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

# Unified key/model. LLM_API_KEY falls back to the historical GROK_API_KEY so
# existing .env files keep working untouched.
LLM_API_KEY = os.getenv("LLM_API_KEY", GROK_API_KEY)
LLM_MODEL = os.getenv("LLM_MODEL", GROK_MODEL)

# Per-provider key overrides (optional; only used when explicitly selected).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "45"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# --- AI pipeline selection ------------------------------------------------
# "legacy"  -> single-shot generate_mcqs (current behaviour, the safe default)
# "rag"     -> retrieval-augmented agent pipeline (enabled in Phase 4)
AI_PIPELINE = os.getenv("AI_PIPELINE", "legacy").strip().lower()

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
