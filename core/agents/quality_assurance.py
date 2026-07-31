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

_LABELS = ["A", "B", "C", "D"]


class QualityAssuranceAgent(Agent):
    name = "quality_assurance"

    def run(self, mcqs: list[dict]) -> tuple[list[dict], list[dict]]:
        valid, rejected = [], []
        for q in mcqs:
            cleaned, reason = self._validate_one(q)
            if cleaned is not None:
                valid.append(cleaned)
            else:
                rejected.append({"question": q.get("question", "?"), "reason": reason})
        self._log(f"QA: {len(valid)} valid, {len(rejected)} rejected")
        return valid, rejected

    def _validate_one(self, q: dict):
        question = (q.get("question") or "").strip()
        if not question:
            return None, "empty question"

        options = q.get("options")
        if not isinstance(options, list) or len(options) != 4:
            return None, "must have exactly 4 options"

        texts, norm_opts = [], []
        for i, o in enumerate(options):
            if not isinstance(o, dict):
                return None, "option is not an object"
            text = (o.get("text") or "").strip()
            if not text:
                return None, "empty option text"
            norm_opts.append({"label": _LABELS[i], "text": text})
            texts.append(text)

        if len(set(texts)) != 4:
            return None, "duplicate option texts"

        answer = (q.get("answer_text") or "").strip()
        if answer not in texts:
            # Repair: model sometimes returns a label ("B") instead of the text.
            if answer in _LABELS:
                answer = norm_opts[_LABELS.index(answer)]["text"]
            else:
                return None, "answer_text does not match any option"

        cleaned = {"question": question, "options": norm_opts, "answer_text": answer}
        # Carry optional metadata forward without letting it break the contract.
        for extra in ("bloom", "source_hint", "explanation"):
            if extra in q:
                cleaned[extra] = q[extra]
        return cleaned, None
