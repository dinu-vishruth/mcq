"""Embedding backends. Import `get_embedder()` for the configured instance."""
from core.embeddings.base import Embedder
from core.embeddings.registry import get_embedder, reset_embedder

__all__ = ["get_embedder", "reset_embedder", "Embedder"]
