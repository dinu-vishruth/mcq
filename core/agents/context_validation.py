"""Strict source sufficiency gate. Unavailable or malformed is not approval."""
from __future__ import annotations

import config
from core.agents.base import Agent
from core.llm import get_llm
from core.models.mcq_contract import ContextVerdict
from core.prompts import context_validation as prompts


class ContextValidationAgent(Agent):
    name = "context_validation"

    def run(self, context: str, num_questions: int, difficulty: str) -> dict:
        if not (context or "").strip():
            return {"sufficient": False, "confidence": 1.0, "reason": "No source context.", "status": "rejected"}
        try:
            data = get_llm().complete_json([
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": prompts.validate_prompt(context, num_questions, difficulty)},
            ], temperature=config.LLM_GENERATION_TEMPERATURE)
            verdict = ContextVerdict.model_validate(data)
        except Exception:
            return {"sufficient": False, "confidence": 0.0,
                    "reason": "Context verification unavailable or malformed.", "status": "unverified"}
        passed = verdict.sufficient and verdict.confidence >= config.FACT_VERIFICATION_MIN_CONFIDENCE
        return {"sufficient": passed, "confidence": verdict.confidence, "reason": verdict.reason,
                "status": "verified" if passed else "rejected" if not verdict.sufficient else "unverified"}
