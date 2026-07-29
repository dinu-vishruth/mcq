"""
Embedder interface. All backends return L2-normalised float32 vectors so the
vector stores can use plain dot-product as cosine similarity.
"""
from __future__ import annotations

import abc


class Embedder(abc.ABC):
    #: Stable identifier stored alongside vectors (embeddings.model column) so a
    #: backend change can be detected and vectors re-embedded rather than mixed.
    backend_id: str = "base"
    dim: int = 0

    @abc.abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in order."""
        raise NotImplementedError

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]
