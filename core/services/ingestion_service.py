"""
Ingestion service: turn raw document text into a retrievable, embedded document.

Flow:  raw text
         -> DocumentProcessingAgent (clean + chunk + metadata)
         -> dedup by doc_hash (skip re-ingest of identical content)
         -> persist document + chunks
         -> EmbeddingAgent (batched, cached, incremental) -> vector store
         -> mark ready, return document_id

Idempotent: the same text yields the same document_id and does no extra work.
This is the single entry point the routes/pipeline call; they never touch agents
directly.
"""
from __future__ import annotations

from core.agents.document_processing import DocumentProcessingAgent
from core.agents.embedding import EmbeddingAgent
from core.repositories import document_repo


def ingest_document(raw_text: str, *, owner: str = "", title: str = "",
                    source_type: str = "paste") -> dict:
    """Return {"document_id", "reused", "chunk_count", "embed_stats"}."""
    processed = DocumentProcessingAgent().run(raw_text, title=title, source_type=source_type)

    if not processed.chunks:
        raise ValueError("Could not extract any readable text to ingest.")

    dh = document_repo.doc_hash(processed.text)
    existing = document_repo.find_by_hash(dh)
    if existing is not None and existing["status"] == "ready":
        return {"document_id": existing["id"], "reused": True,
                "chunk_count": existing["chunk_count"], "embed_stats": None}

    if existing is not None:
        document_id = existing["id"]
    else:
        document_id = document_repo.create(
            dh, owner, title or (source_type + " document"),
            source_type, processed.char_count, meta=processed.metadata,
        )

    document_repo.set_status(document_id, "chunked", chunk_count=len(processed.chunks))
    embed_stats = EmbeddingAgent().run(document_id, processed.chunks)
    document_repo.set_status(document_id, "ready", chunk_count=len(processed.chunks))

    return {"document_id": document_id, "reused": False,
            "chunk_count": len(processed.chunks), "embed_stats": embed_stats}
