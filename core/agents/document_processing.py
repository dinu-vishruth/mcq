"""
Document Processing Agent: extract -> clean -> chunk -> metadata.

Text extraction reuses the existing, battle-tested functions in
models/pdf_processor.py (tables, group shapes, slide notes) so no extraction
capability is lost. Cleaning reuses utils.text_cleaner.clean_text. The agent's
own contribution is chunking (core.rag.chunker) and metadata assembly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from utils.text_cleaner import clean_text
from core.rag.chunker import chunk_text, Chunk
from core.agents.base import Agent


@dataclass
class ProcessedDocument:
    text: str
    chunks: list[Chunk]
    char_count: int
    metadata: dict = field(default_factory=dict)


class DocumentProcessingAgent(Agent):
    name = "document_processing"

    def run(self, raw_text: str, *, title: str = "", source_type: str = "paste") -> ProcessedDocument:
        text = clean_text(raw_text or "")
        chunks = chunk_text(text)
        meta = {
            "title": title,
            "source_type": source_type,
            "chunk_count": len(chunks),
            "avg_chunk_chars": (sum(len(c.content) for c in chunks) // len(chunks)) if chunks else 0,
        }
        self._log(f"processed '{title or source_type}': {len(text)} chars -> {len(chunks)} chunks")
        return ProcessedDocument(text=text, chunks=chunks, char_count=len(text), metadata=meta)
