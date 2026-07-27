"""
Embedder selection with capability auto-detection.

EMBEDDING_BACKEND:
  "auto"   -> SentenceTransformer if it (and torch) import; else hashing.
              This is what makes RAG work on both a capable host and Vercel
              with no config change.
  "sentence_transformer" | "st" -> force local ST (errors if unavailable)
  "remote" -> OpenAI/Gemini embedding endpoint
  "hashing" -> pure-Python fallback
"""
from __future__ import annotations

import config
from core.embeddings.base import Embedder

_cached: Embedder | None = None


def _st_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def build_embedder() -> Embedder:
    backend = config.EMBEDDING_BACKEND

    if backend in ("sentence_transformer", "st"):
        from core.embeddings.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder()

    if backend == "remote":
        from core.embeddings.remote_api import RemoteAPIEmbedder
        return RemoteAPIEmbedder()

    if backend == "hashing":
        from core.embeddings.hashing import HashingEmbedder
        return HashingEmbedder()

    # auto
    if _st_available():
        try:
            from core.embeddings.sentence_transformer import SentenceTransformerEmbedder
            return SentenceTransformerEmbedder()
        except Exception as e:
            print(f"[WARNING] SentenceTransformer load failed, falling back to hashing: {e}")
    from core.embeddings.hashing import HashingEmbedder
    return HashingEmbedder()


def get_embedder() -> Embedder:
    global _cached
    if _cached is None:
        _cached = build_embedder()
    return _cached


def reset_embedder() -> None:
    global _cached
    _cached = None
