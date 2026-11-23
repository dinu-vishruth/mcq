"""
Quality Assurance Agent: deterministic structural validation of generated MCQs.

Enforces the exact contract the whole app depends on:
  - question is a non-empty string
  - exactly 4 options, each {"label","text"} with non-empty text
  - labels normalised to A/B/C/D
  - answer_text matches exactly one option's text (repairs A/B/C/D answers)
  - no duplicate option texts within a question

Returns (valid_mcqs, rejected) where rejected carries a reason so the pipeline
can decide whether to regenerate. This runs with no LLM call -- it's pure logic.
"""
from __future__ import annotations

from core.agents.base import Agent
from core.models.questions import validate_question, validate_questions


class QualityAssuranceAgent(Agent):
    name = "quality_assurance"

    def run(self, mcqs):
        valid, rejected = validate_questions(mcqs)
        self._log(f"QA: {len(valid)} valid, {len(rejected)} rejected")
        return valid, rejected

    def _validate_one(self, q):
        try:
            return validate_question(q), None
        except ValueError as exc:
            return None, str(exc)
