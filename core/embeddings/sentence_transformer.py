"""
Local SentenceTransformer backend (all-MiniLM-L6-v2 by default).

High quality, runs on your machine / Docker / any VPS. NEVER available on
Vercel (torch exceeds the 250 MB bundle limit) -- the registry only selects
this backend when sentence_transformers actually imports, and falls back
otherwise, so importing this module lazily is essential.
"""
from __future__ import annotations

import config
from core.embeddings.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = SentenceTransformer(self.model_name)
        # Method was renamed in sentence-transformers 5.x; support both.
        if hasattr(self._model, "get_embedding_dimension"):
            self.dim = self._model.get_embedding_dimension()
        else:
            self.dim = self._model.get_sentence_embedding_dimension()
        self.backend_id = f"st:{self.model_name}"

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,   # L2-normalised -> dot product = cosine
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32").tolist()
