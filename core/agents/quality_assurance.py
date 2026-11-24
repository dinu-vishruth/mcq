"""Shared deterministic validation for fresh candidates and saved MCQ contracts."""
from __future__ import annotations

from pydantic import ValidationError

from core.agents.base import Agent
from core.models.mcq_contract import MCQ, GroundedMCQ, normalized_text, question_key


class QualityAssuranceAgent(Agent):
    name = "quality_assurance"

    def run(self, mcqs, *, context: str | None = None, require_grounding: bool = False,
            excluded=()) -> tuple[list[dict], list[dict]]:
        if not isinstance(mcqs, list):
            return [], [{"question": "?", "reason": "questions must be an array", "status": "rejected"}]
        valid, rejected = [], []
        seen = {question_key(q["question"]) for q in excluded}
        for raw in mcqs:
            cleaned, reason = self._validate_one(raw, context=context,
                                                 require_grounding=require_grounding)
            if cleaned is not None:
                key = question_key(cleaned["question"])
                if key in seen:
                    reason = "duplicate question"
                else:
                    seen.add(key)
                    valid.append(cleaned)
                    continue
            rejected.append({"question": raw.get("question", "?") if isinstance(raw, dict) else "?",
                             "reason": reason, "status": "rejected"})
        return valid, rejected

    def _validate_one(self, raw, *, context=None, require_grounding=False):
        try:
            schema = GroundedMCQ if require_grounding else MCQ
            model = schema.model_validate(raw)
            if require_grounding:
                source = normalized_text(context or "")
                quote = normalized_text(model.source_hint)
                if not source or quote not in source:
                    return None, "source_hint must quote the supplied source context"
            return model.model_dump(exclude_none=True), None
        except ValidationError:
            return None, "invalid MCQ: nonempty strings, 4 options, distinct texts and one matching answer required"
