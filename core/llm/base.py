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
import logging
from contextvars import ContextVar
from typing import Any

from core.services.generation_budget import generation_scope


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
        #: Raw provider usage dict from the most recent successful call (None
        #: when the provider didn't report it). Read by complete() for logging.
        self._usage = ContextVar(f"llm_usage_{id(self)}", default=None)

    @property
    def last_usage(self):
        return self._usage.get()

    @last_usage.setter
    def last_usage(self, value):
        self._usage.set(value)

    @abc.abstractmethod
    def _complete_once(self, messages: list[dict], *, temperature: float | None = None,
                       max_tokens: int | None = None, json_mode: bool = False) -> str:
        """Single provider call. Subclasses implement this; they must raise
        RateLimitError on HTTP 429 so the retry layer in complete() can back off."""
        raise NotImplementedError

    @generation_scope
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
        from core.observability import log_event

        max_retries = max(0, int(getattr(config, "LLM_MAX_RETRIES", 3)))
        base = float(getattr(config, "LLM_RETRY_BACKOFF", 1.5))
        max_wait = float(getattr(config, "LLM_RETRY_MAX_WAIT", 8))
        provider = getattr(self, "provider_id", "unknown")
        prompt_chars = sum(len(m.get("content") or "") for m in messages)

        from core.services.generation_budget import check, remaining

        attempt = 0
        while True:
            check()
            self.last_usage = None
            started = time.perf_counter()
            try:
                result = self._complete_once(messages, temperature=temperature,
                                             max_tokens=max_tokens, json_mode=json_mode)
            except RateLimitError as e:
                log_event(
                    "llm.rate_limited", level=logging.WARNING, provider=provider,
                    model=self.model, attempt=attempt + 1,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    retry_after=e.retry_after,
                )
                if attempt >= max_retries:
                    raise
                # Prefer the provider's own hint; else exponential backoff.
                wait = e.retry_after if e.retry_after is not None else base * (2 ** attempt)
                if e.retry_after is None:
                    wait = min(wait, max_wait)
                available = remaining(reserve=4)
                if available is not None and wait + 0.1 >= available:
                    raise
                # Never retry earlier than Retry-After; caller may retry later.
                attempt += 1
                log_event("llm.retry", provider=provider, model=self.model,
                          attempt=attempt, max_retries=max_retries,
                          wait_seconds=round(wait, 2))
                time.sleep(wait)
                continue
            except Exception as exc:
                log_event(
                    "llm.call", level=logging.ERROR, provider=provider,
                    model=self.model, status="error",
                    error_type=type(exc).__name__,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    json_mode=json_mode, prompt_chars=prompt_chars,
                )
                raise
            else:
                log_event(
                    "llm.call", provider=provider, model=self.model, status="ok",
                    attempt=attempt + 1,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    json_mode=json_mode, prompt_chars=prompt_chars,
                    max_tokens=max_tokens, usage=self.last_usage,
                )
                return result

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
            import math
            from datetime import datetime, timezone
            from email.utils import parsedate_to_datetime
            try:
                seconds = float(val)
            except ValueError:
                seconds = (parsedate_to_datetime(val) - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, seconds) if math.isfinite(seconds) else None
        except (TypeError, ValueError, OverflowError):
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
