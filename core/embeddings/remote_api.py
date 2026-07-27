"""
Remote embedding backend via OpenAI-compatible or Gemini embedding endpoints.

Deploys anywhere (no heavy deps), good quality, at the cost of per-document API
calls. Selected when EMBEDDING_BACKEND=remote. Note: Groq does not offer an
embeddings endpoint, so a remote setup needs an OpenAI or Gemini key.
"""
from __future__ import annotations

import math

import requests

import config
from core.embeddings.base import Embedder


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


class RemoteAPIEmbedder(Embedder):
    """Supports provider 'openai' (text-embedding-3-small) and 'gemini'."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or ("openai" if config.OPENAI_API_KEY else "gemini")).lower()
        if self.provider == "openai":
            self.api_key = config.OPENAI_API_KEY
            self.model = model or "text-embedding-3-small"
            self.dim = 1536
        else:
            self.api_key = config.GEMINI_API_KEY
            self.model = model or "text-embedding-004"
            self.dim = 768
        self.backend_id = f"remote:{self.provider}:{self.model}"

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "openai":
            return self._openai(texts)
        return self._gemini(texts)

    def _openai(self, texts):
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": texts},
            timeout=config.LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [_l2(d["embedding"]) for d in data]

    def _gemini(self, texts):
        out = []
        for t in texts:  # Gemini embedContent is per-text on v1beta
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{self.model}:embedContent?key={self.api_key}")
            resp = requests.post(url, json={"content": {"parts": [{"text": t}]}},
                                 timeout=config.LLM_TIMEOUT)
            resp.raise_for_status()
            out.append(_l2(resp.json()["embedding"]["values"]))
        return out
