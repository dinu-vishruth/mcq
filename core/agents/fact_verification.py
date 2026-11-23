"""
Fact Verification Agent: the verification stage that was missing before MCQs
were presented.

Runs AFTER structural QA and BEFORE questions are accepted. In one batched LLM
call it checks every MCQ against the retrieved CONTEXT and rejects any that are
unsupported, have multiple/zero correct answers, are ambiguous, have a distractor
that could be argued correct, or whose explanation contradicts the context.

Returns (verified, rejected) where rejected carries a reason, so the pipeline can
regenerate only the shortfall (mirroring the DifficultyAgent contract).

If the checker is unavailable, structurally valid questions remain usable but
are explicitly marked unverified. Only strict, complete verdicts grant verified
status. An explicit failure rejects the question.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, StrictBool, StrictInt, ValidationError

import config
from core.agents.base import Agent
from core.llm import LLMError, get_llm
from core.prompts import fact_verification as prompts


class Verdict(BaseModel):
    index: StrictInt = Field(ge=0)
    passed: StrictBool
    confidence: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)
    issue: str = ""


class FactVerificationAgent(Agent):
    name = "fact_verification"

    def run(self, mcqs: list[dict], context: str) -> tuple[list[dict], list[dict]]:
        mcqs = [{**q, "verification": "unverified"} for q in mcqs]
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
        verdicts, duplicates = {}, set()
        if not isinstance(results, list):
            results = []
        for raw in results:
            try:
                verdict = Verdict.model_validate(raw)
            except ValidationError:
                continue
            if verdict.index in verdicts:
                duplicates.add(verdict.index)
            verdicts[verdict.index] = verdict

        verified, rejected = [], []
        for i, q in enumerate(mcqs):
            verdict = verdicts.get(i)
            if verdict is None or i in duplicates:
                # Availability fallback is explicit: never call this verified.
                verified.append(q)
            elif verdict.passed and verdict.confidence >= min_conf:
                verified.append({**q, "verification": "verified"})
            else:
                rejected.append({"question": q["question"],
                                 "reason": verdict.issue or "Fact verification did not pass"})

        checked = sum(q["verification"] == "verified" for q in verified)
        self._log(f"verification: {checked} verified, {len(verified) - checked} unverified, {len(rejected)} rejected")
        return verified, rejected
