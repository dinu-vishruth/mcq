"""The MCQ contract shared by generation, QA and session persistence."""
import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StrictStr, StringConstraints, ValidationError

Text = Annotated[StrictStr, StringConstraints(strip_whitespace=True, min_length=1)]
LABELS = ("A", "B", "C", "D")


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


class Option(BaseModel):
    label: Literal["A", "B", "C", "D"]
    text: Text


class Evidence(BaseModel):
    quote: Text
    document_id: int | None = Field(default=None, strict=True, ge=1)
    chunk_index: int | None = Field(default=None, strict=True, ge=0)


class MCQ(BaseModel):
    question: Text
    options: list[Option] = Field(min_length=4, max_length=4)
    answer_text: Text


def validate_question(raw, *, trusted_metadata=False) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("question must be an object")
    options = raw.get("options")
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError("must have exactly 4 options")
    options = [dict(option, label=LABELS[i]) if isinstance(option, dict) else option
               for i, option in enumerate(options)]
    try:
        parsed = MCQ.model_validate({**raw, "options": options})
    except ValidationError as exc:
        raise ValueError("question, options and answer must contain non-empty text") from exc
    texts = [option.text for option in parsed.options]
    if len({normalized(text) for text in texts}) != 4:
        raise ValueError("duplicate option texts")
    answer = parsed.answer_text
    if answer not in texts and answer in LABELS:
        answer = texts[LABELS.index(answer)]
    if answer not in texts:
        raise ValueError("answer_text does not match any option")
    cleaned = parsed.model_dump()
    cleaned["answer_text"] = answer
    for key in ("bloom", "source_hint", "explanation"):
        if isinstance(raw.get(key), str) and raw[key].strip():
            cleaned[key] = raw[key].strip()
    # Models cannot grant themselves provenance or verified status.
    if trusted_metadata:
        status = raw.get("verification")
        if status in ("verified", "unverified"):
            cleaned["verification"] = status
        if isinstance(raw.get("sources"), list):
            sources = []
            for source in raw["sources"]:
                try:
                    sources.append(Evidence.model_validate(source).model_dump())
                except ValidationError:
                    continue
            if sources:
                cleaned["sources"] = sources
    return cleaned


def validate_questions(raw, *, trusted_metadata=False) -> tuple[list[dict], list[dict]]:
    if not isinstance(raw, list):
        return [], [{"question": "?", "reason": "questions must be an array"}]
    valid, rejected, seen = [], [], set()
    for question in raw:
        try:
            cleaned = validate_question(question, trusted_metadata=trusted_metadata)
            key = normalized(cleaned["question"])
            if key in seen:
                raise ValueError("duplicate question")
            seen.add(key)
            valid.append(cleaned)
        except ValueError as exc:
            rejected.append({"question": question.get("question", "?") if isinstance(question, dict) else "?",
                             "reason": str(exc)})
    return valid, rejected


def attach_evidence(mcqs, context: str, *, hits=(), document_id=None):
    """Only retain quotes found in actual source text, never model-supplied IDs.

    A matching quote demonstrates provenance, not factual correctness. That is
    represented separately by the verifier's status.
    """
    candidates = [(hit.content, document_id, hit.chunk_index) for hit in hits]
    if not candidates:
        candidates = [(context, document_id, None)]
    for mcq in mcqs:
        mcq.pop("sources", None)
        mcq.setdefault("verification", "unverified")
        quote = mcq.get("source_hint", "")
        if not isinstance(quote, str) or not quote.strip() or len(quote) > 300:
            continue
        # The quote must also have appeared in the context actually sent.
        if normalized(quote) not in normalized(context):
            continue
        for text, doc_id, index in candidates:
            match = re.search(re.escape(normalized(quote)), normalized(text))
            if match:
                mcq["sources"] = [{"quote": quote.strip(), "document_id": doc_id, "chunk_index": index}]
                break
    return mcqs
