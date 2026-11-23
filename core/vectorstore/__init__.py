"""Vector store backends. Import `get_vector_store()` for the configured instance."""
from core.vectorstore.base import SearchHit, VectorStore
from core.vectorstore.registry import get_vector_store, reset_vector_store

__all__ = ["get_vector_store", "reset_vector_store", "VectorStore", "SearchHit"]
