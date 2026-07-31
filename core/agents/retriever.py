"""
Retriever Agent: turn an intent + document into the most relevant, diverse set
of context chunks. Thin agent wrapper over core.rag.retrieval.Retriever so the
retrieval policy (MMR, spread) lives in one place and the agent stays a
single-responsibility unit the pipeline composes.
"""
from __future__ import annotations

from core.agents.base import Agent
from core.rag.retrieval.retriever import Retriever

# Cosine below this hints the top chunks aren't a strong match for the query.
# This is a SOFT signal only: it's surfaced to downstream agents and logged, but
# never blocks generation. The hashing-embedder fallback (used on Vercel / when
# SentenceTransformers is absent) yields near-zero cosines even for good matches,
# so a hard threshold here would wrongly reject valid context.
_LOW_CONFIDENCE_SCORE = 0.25


class RetrieverAgent(Agent):
    name = "retriever"

    def __init__(self):
        self._retriever = Retriever()

    def run(self, document_id: int, query: str, top_k: int, *, spread: bool = True):
        hits = self._retriever.retrieve(document_id, query, top_k, use_mmr=True, spread=spread)
        self._log(f"doc {document_id}: retrieved {len(hits)} chunks for query '{query[:40]}'")
        return hits

    @staticmethod
    def assess(hits) -> dict:
        """Summarise retrieval quality for downstream agents / logging.

        Returns {count, avg_score, min_score, max_score, low_confidence}. The
        `low_confidence` flag is advisory only (see _LOW_CONFIDENCE_SCORE) — the
        Context Validation agent, not this score, decides whether to proceed.
        """
        scores = [float(getattr(h, "score", 0.0) or 0.0) for h in hits]
        if not scores:
            return {"count": 0, "avg_score": 0.0, "min_score": 0.0,
                    "max_score": 0.0, "low_confidence": True}
        avg = sum(scores) / len(scores)
        return {
            "count": len(scores),
            "avg_score": round(avg, 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "low_confidence": avg < _LOW_CONFIDENCE_SCORE,
        }
