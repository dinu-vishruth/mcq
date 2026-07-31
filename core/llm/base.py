"""
Provider-agnostic LLM interface.

Every adapter implements two methods:
    complete(messages, ...)      -> raw assistant text
    complete_json(messages, ...) -> parsed dict/list from a JSON response

Agents and services depend only on this interface, never on a concrete
provider, so switching Groq <-> OpenAI <-> Gemini <-> Anthropic is a config
change with no code change.
"""
from __future__ import annotations

import abc
import json
from typing import Any


class LLMError(Exception):
    """Raised for any LLM call failure. Carries a user-safe message."""


class RateLimitError(LLMError):
    """Provider returned HTTP 429. Carries an optional retry_after hint (seconds)
    parsed from the Retry-After header so the retry layer can respect it."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMProvider(abc.ABC):
    """Abstract base for all LLM adapters."""

    def __init__(self, api_key: str, model: str, timeout: int = 45, temperature: float = 0.3):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    @abc.abstractmethod
    def _complete_once(self, messages: list[dict], *, temperature: float | None = None,
                       max_tokens: int | None = None, json_mode: bool = False) -> str:
        """Single provider call. Subclasses implement this; they must raise
        RateLimitError on HTTP 429 so the retry layer in complete() can back off."""
        raise NotImplementedError

    def complete(self, messages: list[dict], *, temperature: float | None = None,
                 max_tokens: int | None = None, json_mode: bool = False) -> str:
        """Return the assistant's raw text, retrying on 429 with exponential
        backoff (honoring Retry-After when the provider sends it).

        messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        json_mode: hint the provider to emit strict JSON when supported.
        """
        # Imported lazily so tests that stub config still see current values.
        import time
        import config

        max_retries = max(0, int(getattr(config, "LLM_MAX_RETRIES", 3)))
        base = float(getattr(config, "LLM_RETRY_BACKOFF", 1.5))
        max_wait = float(getattr(config, "LLM_RETRY_MAX_WAIT", 8))

        attempt = 0
        while True:
            try:
                return self._complete_once(messages, temperature=temperature,
                                           max_tokens=max_tokens, json_mode=json_mode)
            except RateLimitError as e:
                if attempt >= max_retries:
                    raise
                # Prefer the provider's own hint; else exponential backoff.
                wait = e.retry_after if e.retry_after is not None else base * (2 ** attempt)
                wait = min(wait, max_wait)
                attempt += 1
                print(f"[llm] 429 rate limit; retry {attempt}/{max_retries} in {wait:.1f}s")
                time.sleep(wait)

    def complete_json(self, messages: list[dict], *, temperature: float | None = None,
                      max_tokens: int | None = None) -> Any:
        """Return parsed JSON from the model. Raises LLMError on invalid JSON."""
        raw = self.complete(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        return self._parse_json(raw)

    @staticmethod
    def _parse_retry_after(resp) -> float | None:
        """Best-effort parse of the Retry-After header (delta-seconds form)."""
        try:
            val = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            if val is None:
                return None
            return max(0.0, float(val))
        except (TypeError, ValueError):
            return None

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _parse_json(raw: str) -> Any:
        """Tolerant JSON extraction: handles fenced blocks and stray prose."""
        if raw is None:
            raise LLMError("The AI service returned an empty response. Please try again.")
        text = raw.strip()
        # Strip common markdown code fences.
        if text.startswith("```"):
            text = text.split("```", 2)
            text = text[1] if len(text) > 1 else raw
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last resort: grab the outermost {...} or [...] span.
            for open_c, close_c in (("{", "}"), ("[", "]")):
                start = text.find(open_c)
                end = text.rfind(close_c)
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        continue
            # Output hit the provider's token cap and was cut mid-array. Recover
            # the objects that did complete instead of discarding the whole batch;
            # callers top up the shortfall with another request.
            salvaged = LLMProvider._salvage_objects(text)
            if salvaged:
                return {"questions": salvaged}
            raise LLMError("The AI service returned an invalid JSON response. Please try again.")

    @staticmethod
    def _salvage_objects(text: str) -> list:
        """Extract every complete top-level JSON object from a truncated array.

        Scans with a brace-depth counter, skipping braces inside string literals,
        and keeps only spans that parse cleanly. Returns [] when nothing survives.
        """
        objects: list = []
        starts: list[int] = []  # positions of currently-open '{', innermost last
        in_string = False
        escaped = False
        for i, ch in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                starts.append(i)
            elif ch == "}" and starts:
                # Objects appear at any depth (question objects sit inside the
                # {"questions": [...]} wrapper), so test every closed span and
                # keep the ones shaped like a question.
                start = starts.pop()
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and "options" in obj:
                    objects.append(obj)
        return objects
