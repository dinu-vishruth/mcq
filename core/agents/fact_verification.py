"""Fact checking with explicit verified, rejected and unverified outcomes."""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from core.agents.base import Agent
from core.llm import get_llm
from core.models.mcq_contract import FactVerdict, indexed_verdicts
from core.prompts import fact_verification as prompts


@dataclass
class VerificationResult:
    verified: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    unverified: list[dict] = field(default_factory=list)

    def __iter__(self):
        # Compatibility for callers consuming (accepted, not_accepted).
        yield self.verified
        yield self.rejected + self.unverified


class FactVerificationAgent(Agent):
    name = "fact_verification"

    def run(self, mcqs: list[dict], context: str, *, topic: str | None = None, excluded=()) -> VerificationResult:
        result = VerificationResult()
        if not mcqs:
            return result
        verdicts = {}
        reason = "verification response is missing or malformed"
        if (context or "").strip():
            try:
                payload = [{"index": i, "question": q["question"],
                            "options": [o["text"] for o in q["options"]],
                            "answer_text": q["answer_text"],
                            "explanation": q.get("explanation", "")} for i, q in enumerate(mcqs)]
                data = get_llm().complete_json([
                    {"role": "system", "content": prompts.SYSTEM},
                    {"role": "user", "content": prompts.verify_prompt(payload, context, topic=topic, excluded=excluded)},
                ], temperature=config.LLM_GENERATION_TEMPERATURE)
                verdicts = indexed_verdicts(data, "results", FactVerdict, len(mcqs))
            except Exception:
                reason = "fact verification unavailable"
        else:
            reason = "no source context to verify"
        for i, q in enumerate(mcqs):
            verdict = verdicts.get(i)
            if verdict is None:
                result.unverified.append({"question": q.get("question", "?"),
                                          "reason": reason, "status": "unverified"})
            elif verdict.semantic_duplicate:
                result.rejected.append({"question": q["question"], "reason": "semantic duplicate",
                                        "status": "rejected"})
            elif not verdict.safe:
                result.rejected.append({"question": q["question"], "reason": "content safety policy",
                                        "status": "rejected"})
            elif not verdict.passed:
                result.rejected.append({"question": q["question"],
                                        "reason": verdict.issue or "failed fact check", "status": "rejected"})
            elif verdict.confidence < config.FACT_VERIFICATION_MIN_CONFIDENCE:
                result.unverified.append({"question": q["question"],
                                          "reason": "insufficient verification confidence", "status": "unverified"})
            else:
                result.verified.append(q)
        return result
