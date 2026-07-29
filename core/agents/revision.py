"""
Revision Agent: generate revision notes for a student's weak topics, grounded
in the source document via retrieval. If no document context is available it
still produces notes from the topic list alone.
"""
from __future__ import annotations

import config
from core.agents.base import Agent
from core.llm import get_llm, LLMError
from core.prompts import revision as prompts


class RevisionAgent(Agent):
    name = "revision"

    def run(self, topics: list[str], *, document_id: int | None = None) -> list[dict]:
        """Return [{"topic","summary","key_points":[...]}]."""
        if not topics:
            return []

        context = self._context(topics, document_id)
        try:
            data = get_llm().complete_json([
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": prompts.revision_notes_prompt(topics, context)},
            ])
            notes = data.get("notes", []) if isinstance(data, dict) else []
            self._log(f"generated {len(notes)} revision notes")
            return notes
        except (LLMError, Exception) as e:
            self._log(f"revision notes skipped: {e}")
            return []

    def _context(self, topics, document_id) -> str:
        if not document_id:
            return ""
        try:
            from core.agents.retriever import RetrieverAgent
            from core.agents.context_builder import ContextBuilderAgent
            hits = RetrieverAgent().run(document_id, " ".join(topics), config.RETRIEVAL_TOP_K, spread=True)
            return ContextBuilderAgent().run(hits, max_chars=6000) if hits else ""
        except Exception:
            return ""
