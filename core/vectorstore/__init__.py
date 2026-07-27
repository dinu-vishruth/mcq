"""Vector store backends. Import `get_vector_store()` for the configured instance."""
from core.vectorstore.registry import get_vector_store, reset_vector_store
from core.vectorstore.base import VectorStore, SearchHit

__all__ = ["get_vector_store", "reset_vector_store", "VectorStore", "SearchHit"]
