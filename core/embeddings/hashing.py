"""
Pure-Python hashing embedder. Zero heavy dependencies -> works anywhere,
including Vercel where torch/sentence-transformers cannot be bundled.

This is a deterministic bag-of-words feature-hashing embedder with L2
normalisation. It is NOT semantically as strong as a neural model, but it
captures lexical overlap well, which is enough to make retrieval meaningfully
better than "send the whole document" -- and it guarantees RAG works in every
environment. On capable hosts the registry prefers SentenceTransformer.
"""
from __future__ import annotations

import hashlib
import math
import re

import config
from core.embeddings.base import Embedder

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder(Embedder):
    def __init__(self, dim: int | None = None):
        self.dim = dim or config.EMBEDDING_DIM
        self.backend_id = f"hash:{self.dim}"

    def _tokens(self, text: str) -> list[str]:
        return _TOKEN.findall((text or "").lower())

    def _hash_index(self, token: str) -> tuple[int, float]:
        h = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if (h[4] & 1) else -1.0  # signed hashing reduces collisions
        return idx, sign

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            toks = self._tokens(text)
            for t in toks:
                idx, sign = self._hash_index(t)
                # sublinear TF via +sign; length-normalised below
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out
