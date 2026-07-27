"""Embedding backends. Import `get_embedder()` for the configured instance."""
from core.embeddings.registry import get_embedder, reset_embedder
from core.embeddings.base import Embedder

__all__ = ["get_embedder", "reset_embedder", "Embedder"]
