"""
Vector store interface.

Documents are namespaced by document_id so retrieval for one MCQ set never
leaks chunks from another document. Vectors are assumed L2-normalised, so
similarity is dot product (cosine).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class SearchHit:
    chunk_id: str          # store-local id (we use "docid:chunkindex")
    document_id: int
    chunk_index: int
    content: str
    score: float           # cosine similarity in [-1, 1]


class VectorStore(abc.ABC):
    @abc.abstractmethod
    def upsert(self, document_id: int, items: list[dict]) -> None:
        """Insert/replace vectors for a document.

        items: [{"chunk_index": int, "content": str, "vector": list[float]}]
        Idempotent per (document_id, chunk_index).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def query(self, document_id: int, query_vector: list[float], top_k: int) -> list[SearchHit]:
        """Return the top_k most similar chunks within one document."""
        raise NotImplementedError

    @abc.abstractmethod
    def has_document(self, document_id: int) -> bool:
        """True if any vectors are stored for this document."""
        raise NotImplementedError

    @abc.abstractmethod
    def delete_document(self, document_id: int) -> None:
        raise NotImplementedError
