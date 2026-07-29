"""
Context Builder Agent: assemble retrieved chunks into a single token-budgeted
context string for the generator. Never sends the whole document -- only the
retrieved chunks, trimmed to CONTEXT_MAX_CHARS, ordered by document position so
the model reads them in a natural sequence.
"""
from __future__ import annotations

import config
from core.agents.base import Agent


class ContextBuilderAgent(Agent):
    name = "context_builder"

    def run(self, hits, *, max_chars: int | None = None) -> str:
        budget = max_chars or config.CONTEXT_MAX_CHARS
        ordered = sorted(hits, key=lambda h: h.chunk_index)
        parts, used = [], 0
        for h in ordered:
            piece = h.content.strip()
            if not piece:
                continue
            block = f"[chunk {h.chunk_index}] {piece}"
            if used + len(block) > budget:
                remaining = budget - used
                if remaining > 200:  # only add a partial block if it's worthwhile
                    parts.append(block[:remaining])
                break
            parts.append(block)
            used += len(block) + 2
        context = "\n\n".join(parts)
        self._log(f"built context: {len(context)} chars from {len(parts)} chunks (budget {budget})")
        return context
