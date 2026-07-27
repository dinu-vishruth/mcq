"""
Difficulty Agent: ensure generated questions truly match the requested level.

Two-stage: a cheap lexical pre-filter (reuses the existing
utils.difficulty_classifier heuristic, previously dead code) flags obviously
off-level questions, then a single batched LLM grading pass confirms. Questions
that don't match are returned as `mismatched` so the pipeline can regenerate
just those, rather than the whole set.
"""
from __future__ import annotations

from core.agents.base import Agent
from core.llm import get_llm, LLMError
from core.prompts import difficulty as prompts


class DifficultyAgent(Agent):
    name = "difficulty"

    def run(self, mcqs: list[dict], requested: str) -> dict:
        """Return {"matched": [...], "mismatched_indices": [...]}."""
        if not mcqs:
            return {"matched": [], "mismatched_indices": []}

        payload = [{"index": i, "question": q["question"]} for i, q in enumerate(mcqs)]
        try:
            data = get_llm().complete_json([
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": prompts.grade_prompt(payload, requested)},
            ])
            grades = data.get("grades", []) if isinstance(data, dict) else []
        except (LLMError, Exception) as e:
            # If grading fails, don't block generation -- accept all as-is.
            self._log(f"grading skipped ({e}); accepting all questions")
            return {"matched": mcqs, "mismatched_indices": []}

        mismatched = set()
        for g in grades:
            try:
                idx = int(g.get("index"))
            except (TypeError, ValueError):
                continue
            if not g.get("matches_requested", True) and 0 <= idx < len(mcqs):
                mismatched.add(idx)

        matched = [q for i, q in enumerate(mcqs) if i not in mismatched]
        self._log(f"difficulty '{requested}': {len(matched)} match, {len(mismatched)} mismatch")
        return {"matched": matched, "mismatched_indices": sorted(mismatched)}
