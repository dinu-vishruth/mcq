"""Typed candidate and checker contracts. No model output is trusted implicitly."""
from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)

Text = Annotated[StrictStr, StringConstraints(strip_whitespace=True, min_length=1)]


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def question_key(text: str) -> str:
    # Ignore presentation punctuation, while preserving mathematical operators.
    return re.sub(r"[?!.,;:\"'‘’“”()\s]+", " ", normalized_text(text)).strip()


class Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)


class Option(Contract):
    label: Literal["A", "B", "C", "D"]
    text: Text


class MCQ(Contract):
    question: Text
    options: Annotated[list[Option], Field(min_length=4, max_length=4)]
    answer_text: Text
    explanation: Text | None = None
    source_hint: Text | None = None
    bloom: Text | None = None

    @model_validator(mode="after")
    def consistent_answers(self):
        if not question_key(self.question):
            raise ValueError("question must contain text beyond presentation punctuation")
        if [o.label for o in self.options] != list("ABCD"):
            raise ValueError("options must use ordered labels A/B/C/D")
        texts = [o.text for o in self.options]
        if len({normalized_text(t) for t in texts}) != 4:
            raise ValueError("duplicate option texts after normalization")
        if sum(self.answer_text == t for t in texts) != 1:
            raise ValueError("answer_text must match exactly one option text")
        return self


class GroundedMCQ(MCQ):
    explanation: Text
    source_hint: Text


class ConfidenceContract(Contract):
    confidence: Annotated[float, Field(ge=0, le=1)]

    @field_validator("confidence", mode="before")
    @classmethod
    def numeric_confidence(cls, value):
        if type(value) not in (int, float):
            raise ValueError("confidence must be a finite number")
        return value


class FactVerdict(ConfidenceContract):
    index: Annotated[StrictInt, Field(ge=0)]
    passed: StrictBool
    safe: StrictBool
    issue: StrictStr = ""


class ContextVerdict(ConfidenceContract):
    sufficient: StrictBool
    reason: StrictStr = ""


class DifficultyVerdict(Contract):
    index: Annotated[StrictInt, Field(ge=0)]
    matches_requested: StrictBool
    level: Literal["easy", "medium", "hard"]


def indexed_verdicts(data, key: str, schema, count: int):
    """Invalid or duplicate verdicts never implicitly approve an item."""
    if not isinstance(data, dict) or not isinstance(data.get(key), list):
        return {}
    verdicts, seen, invalid = {}, set(), set()
    for raw in data[key]:
        if not isinstance(raw, dict):
            continue
        index = raw.get("index")
        if type(index) is not int or not 0 <= index < count:
            return {}
        if index in seen:
            invalid.add(index)
        seen.add(index)
        try:
            verdicts[index] = schema.model_validate(raw)
        except ValueError:
            invalid.add(index)
    return {i: v for i, v in verdicts.items() if i not in invalid}
