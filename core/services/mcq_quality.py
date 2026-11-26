"""One publication policy for primary, legacy, fallback and top-up candidates."""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from core.agents.difficulty import DifficultyAgent
from core.agents.fact_verification import FactVerificationAgent
from core.agents.quality_assurance import QualityAssuranceAgent
from core.observability import timed
from core.services.ai_safety import candidate_allowed
from core.services.source_evidence import issue


@dataclass
class AcceptanceResult:
    accepted: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    unverified: list[dict] = field(default_factory=list)


def accept_candidates(raw, context: str, difficulty: str, *, topic: str | None = None,
                      excluded=()) -> AcceptanceResult:
    """Only structurally valid, grounded, fact-checked and level-matched MCQs pass.

    Unverified candidates are excluded, including when fact checking is disabled.
    A source quote is provenance, not proof of factual correctness.
    """
    with timed("quality_assurance") as stage:
        valid, rejected = QualityAssuranceAgent().run(
            raw, context=context, require_grounding=True, excluded=excluded,
        )
        stage.update(valid=len(valid), rejected=len(rejected))
    result = AcceptanceResult(rejected=rejected)
    safe = []
    for candidate in valid:
        if candidate_allowed(candidate):
            safe.append(candidate)
        else:
            result.rejected.append({"question": "[withheld]", "status": "rejected",
                                    "reason": "content safety policy"})
    valid = safe
    if not valid:
        return result
    if not config.FACT_VERIFICATION_ENABLED:
        result.unverified = [{"question": q["question"], "status": "unverified",
                              "reason": "fact verification disabled"} for q in valid]
        return result
    with timed("fact_verification") as stage:
        facts = FactVerificationAgent().run(valid, context, topic=topic, excluded=excluded)
        stage.update(passed=len(facts.verified), rejected=len(facts.rejected), unverified=len(facts.unverified))
    result.rejected.extend(facts.rejected)
    result.unverified.extend(facts.unverified)
    with timed("difficulty") as stage:
        grades = DifficultyAgent().run(facts.verified, difficulty)
        stage.update(matched=len(grades["matched"]), unverified=len(grades["unverified_indices"]))
    for i in grades["mismatched_indices"]:
        result.rejected.append({"question": facts.verified[i]["question"],
                                "status": "rejected", "reason": "difficulty mismatch"})
    for i in grades["unverified_indices"]:
        result.unverified.append({"question": facts.verified[i]["question"],
                                  "status": "unverified", "reason": "difficulty verification unavailable"})
    result.accepted = [{**q, "verification_status": "verified", "safety_status": "verified",
                        "source_evidence": issue(q, context)}
                       for q in grades["matched"]]
    return result
