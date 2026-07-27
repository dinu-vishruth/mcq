"""
Retriever Agent: turn an intent + document into the most relevant, diverse set
of context chunks. Thin agent wrapper over core.rag.retrieval.Retriever so the
retrieval policy (MMR, spread) lives in one place and the agent stays a
single-responsibility unit the pipeline composes.
"""
from __future__ import annotations

from core.agents.base import Agent
from core.rag.retrieval.retriever import Retriever


class RetrieverAgent(Agent):
    name = "retriever"

    def __init__(self):
        self._retriever = Retriever()

    def run(self, document_id: int, query: str, top_k: int, *, spread: bool = True):
        hits = self._retriever.retrieve(document_id, query, top_k, use_mmr=True, spread=spread)
        self._log(f"doc {document_id}: retrieved {len(hits)} chunks for query '{query[:40]}'")
        return hits
