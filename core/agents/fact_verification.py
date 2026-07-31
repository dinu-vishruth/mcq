"""
Fact Verification Agent: the verification stage that was missing before MCQs
were presented.

Runs AFTER structural QA and BEFORE questions are accepted. In one batched LLM
call it checks every MCQ against the retrieved CONTEXT and rejects any that are
unsupported, have multiple/zero correct answers, are ambiguous, have a distractor
that could be argued correct, or whose explanation contradicts the context.

Returns (verified, rejected) where rejected carries a reason, so the pipeline can
regenerate only the shortfall (mirroring the DifficultyAgent contract).

Fail-open: if the LLM call fails or returns nothing usable, all questions pass
through unchanged — verification can never make generation fail outright.
"""
from __future__ import annotations

import config
from core.agents.base import Agent
from core.llm import get_llm, LLMError
from core.prompts import fact_verification as prompts


class FactVerificationAgent(Agent):
    name = "fact_verification"

    def run(self, mcqs: list[dict], context: str) -> tuple[list[dict], list[dict]]:
        if not mcqs:
            return [], []
        if not (context or "").strip():
            # No context to verify against -> don't block (fail-open).
            self._log("no context; skipping verification")
            return mcqs, []

        payload = [{
            "index": i,
            "question": q["question"],
            "options": [o["text"] for o in q["options"]],
            "answer_text": q["answer_text"],
            "explanation": q.get("explanation", ""),
        } for i, q in enumerate(mcqs)]

        try:
            data = get_llm().complete_json([
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": prompts.verify_prompt(payload, context)},
            ], temperature=config.LLM_GENERATION_TEMPERATURE)
            results = data.get("results", []) if isinstance(data, dict) else []
        except (LLMError, Exception) as e:
            self._log(f"verification skipped ({e}); accepting all questions")
            return mcqs, []

        min_conf = config.FACT_VERIFICATION_MIN_CONFIDENCE
        verdicts: dict[int, dict] = {}
        for r in results:
            try:
                idx = int(r.get("index"))
            except (TypeError, ValueError):
                continue
            verdicts[idx] = r

        verified, rejected = [], []
        for i, q in enumerate(mcqs):
            r = verdicts.get(i)
            if r is None:
                # Not graded (model dropped it) -> fail-open, keep the question.
                verified.append(q)
                continue
            passed = bool(r.get("passed", True))
            try:
                confidence = float(r.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            # Reject on an explicit fail, OR a low-confidence pass (an uncertain
            # "looks fine" is not good enough to show a learner).
            if passed and confidence >= min_conf:
                verified.append(q)
            else:
                reason = (r.get("issue") or "").strip() or (
                    f"low verification confidence ({confidence:.2f})" if passed else "failed fact-check")
                rejected.append({"question": q.get("question", "?"), "reason": reason})

        self._log(f"verification: {len(verified)} verified, {len(rejected)} rejected")
        return verified, rejected
