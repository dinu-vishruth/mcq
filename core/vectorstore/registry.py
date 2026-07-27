"""
Vector store selection.

VECTOR_STORE: "chroma" (default) | "sqlite" | "faiss".
The store is bound to the active embedder's backend_id and dim so a change in
embedding model is recorded with the vectors. On Vercel the read-only
filesystem forces the sqlite store regardless of the configured value, since
Chroma/FAISS need writable on-disk indexes outside /tmp.
"""
from __future__ import annotations

import config
from core.embeddings import get_embedder
from core.vectorstore.base import VectorStore

_cached: VectorStore | None = None


def build_vector_store() -> VectorStore:
    emb = get_embedder()
    model_id, dim = emb.backend_id, emb.dim

    choice = config.VECTOR_STORE
    if config.IS_VERCEL:
        choice = "sqlite"  # read-only FS elsewhere; mcq.db lives in /tmp

    if choice == "chroma":
        try:
            from core.vectorstore.chroma_store import ChromaStore
            return ChromaStore(model_id, dim)
        except Exception as e:
            print(f"[WARNING] Chroma unavailable, falling back to sqlite store: {e}")
            from core.vectorstore.sqlite_numpy import SqliteNumpyStore
            return SqliteNumpyStore(model_id, dim)

    if choice == "faiss":
        try:
            from core.vectorstore.faiss_store import FaissStore
            return FaissStore(model_id, dim)
        except Exception as e:
            print(f"[WARNING] FAISS unavailable, falling back to sqlite store: {e}")
            from core.vectorstore.sqlite_numpy import SqliteNumpyStore
            return SqliteNumpyStore(model_id, dim)

    from core.vectorstore.sqlite_numpy import SqliteNumpyStore
    return SqliteNumpyStore(model_id, dim)


def get_vector_store() -> VectorStore:
    global _cached
    if _cached is None:
        _cached = build_vector_store()
    return _cached


def reset_vector_store() -> None:
    global _cached
    _cached = None
