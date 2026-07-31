"""
Context Validation Agent: gate before MCQ generation.

Given the assembled CONTEXT, it asks the LLM whether there's enough substantive
information to generate grounded questions. If the context is confidently
insufficient, the pipeline should NOT ask the generator to invent facts — it
falls back to the fuller-text legacy path instead.

Fail-open by design: any LLM error, or a low-confidence "insufficient" verdict,
is treated as sufficient so a flaky check can never block generation. Only a
confident INSUFFICIENT verdict stops the RAG path.
"""
from __future__ import annotations

import config
from core.agents.base import Agent
from core.llm import get_llm, LLMError
from core.prompts import context_validation as prompts


class ContextValidationAgent(Agent):
    name = "context_validation"

    def run(self, context: str, num_questions: int, difficulty: str) -> dict:
        """Return {"sufficient": bool, "confidence": float, "reason": str}."""
        if not (context or "").strip():
            self._log("empty context -> INSUFFICIENT")
            return {"sufficient": False, "confidence": 1.0, "reason": "No retrieved context."}

        try:
            data = get_llm().complete_json([
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": prompts.validate_prompt(context, num_questions, difficulty)},
            ], temperature=config.LLM_GENERATION_TEMPERATURE)
        except (LLMError, Exception) as e:
            # Fail-open: don't block generation on a checker failure.
            self._log(f"validation skipped ({e}); treating context as sufficient")
            return {"sufficient": True, "confidence": 0.0, "reason": "validator unavailable"}

        sufficient = bool(data.get("sufficient", True)) if isinstance(data, dict) else True
        try:
            confidence = float(data.get("confidence", 0.0)) if isinstance(data, dict) else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        reason = (data.get("reason") if isinstance(data, dict) else "") or ""

        # Only STOP the RAG path on a CONFIDENT insufficiency verdict. A
        # low-confidence "insufficient" is treated as sufficient (fail-open),
        # so borderline material still gets a generation attempt.
        blocking = (not sufficient) and confidence >= config.FACT_VERIFICATION_MIN_CONFIDENCE
        verdict = "INSUFFICIENT" if blocking else "PASS"
        self._log(f"{verdict} (sufficient={sufficient}, confidence={confidence:.2f}) {reason}".strip())
        return {"sufficient": not blocking, "confidence": confidence, "reason": reason}
