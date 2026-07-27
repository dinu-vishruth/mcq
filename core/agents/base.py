"""Base agent. Provides a name, optional shared LLM handle, and light logging."""
from __future__ import annotations


class Agent:
    name: str = "agent"

    def _log(self, msg: str) -> None:
        print(f"[agent:{self.name}] {msg}")
