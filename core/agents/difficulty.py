"""Typed difficulty assessments. Missing and malformed verdicts never match."""
from __future__ import annotations

from core.agents.base import Agent
from core.llm import get_llm
from core.models.mcq_contract import DifficultyVerdict, indexed_verdicts
from core.prompts import difficulty as prompts


class DifficultyAgent(Agent):
    name = "difficulty"

    def run(self, mcqs: list[dict], requested: str) -> dict:
        verdicts = {}
        if mcqs:
            try:
                payload = [{"index": i, **q} for i, q in enumerate(mcqs)]
                data = get_llm().complete_json([
                    {"role": "system", "content": prompts.SYSTEM},
                    {"role": "user", "content": prompts.grade_prompt(payload, requested)},
                ])
                verdicts = indexed_verdicts(data, "grades", DifficultyVerdict, len(mcqs))
            except Exception:
                pass
        matched, mismatched, unverified = [], [], []
        for i, q in enumerate(mcqs):
            verdict = verdicts.get(i)
            if verdict is None:
                unverified.append(i)
            elif verdict.matches_requested and verdict.level == requested:
                matched.append(q)
            else:
                mismatched.append(i)
        return {"matched": matched, "mismatched_indices": mismatched, "unverified_indices": unverified}
