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
        budget = max_chars if max_chars is not None else config.CONTEXT_MAX_CHARS
        ordered = sorted((h for h in hits if h.content.strip()), key=lambda h: h.chunk_index)
        # Reserve a readable excerpt for each selected section before packing.
        capacity = max(1, budget // 120)
        if len(ordered) > capacity:
            ordered = [ordered[round(i * (len(ordered) - 1) / max(1, capacity - 1))] for i in range(capacity)]
        parts, used = [], 0
        for index, h in enumerate(ordered):
            piece = h.content.strip()
            if not piece:
                continue
            block = f"[chunk {h.chunk_index}] {piece}"
            allowance = max(0, (budget - used - 2 * (len(ordered) - index - 1)) // (len(ordered) - index))
            block = block[:allowance]
            if block:
                parts.append(block)
                used += len(block) + 2
        context = "\n\n".join(parts)
        self._log(f"built context: {len(context)} chars from {len(parts)} chunks (budget {budget})")
        return context
