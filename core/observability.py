"""Structured, request-scoped observability.

The app historically logged with ``print`` (human sentences, no correlation
between lines). This module adds a single JSON log stream without rewriting
every call site:

  * ``log_event(event, **fields)`` -> one JSON object per line
  * a request-id contextvar + ASGI middleware, echoed as ``X-Request-ID``
  * ``timed(stage)`` for duration measurement around pipeline stages
  * an optional, env-gated trace hook so an external system (LangSmith,
    Langfuse, an OTLP collector, ...) can receive the same events without this
    repo taking on an SDK dependency.

No prompt text, document text, API key or user content is ever put in a log
event. Fields are identifiers and counters by design.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("mcq")
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_CONFIGURED = False
_HOOK_CACHE: dict[str, object] = {}


def current_request_id() -> str | None:
    return request_id_var.get()


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON object, merging ``extra={"fields": {...}}``."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging() -> logging.Logger:
    """Configure the ``mcq`` logger once. Safe to call from every entry point.

    ``LOG_FORMAT=text`` restores a human-readable line for local development;
    JSON is the default because hosted logs are queried, not read.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return logger

    import config

    level_name = (getattr(config, "LOG_LEVEL", "INFO") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    if (getattr(config, "LOG_FORMAT", "json") or "json").lower() == "text":
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"))
    else:
        handler.setFormatter(JsonFormatter())

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _CONFIGURED = True
    return logger


def _reset_logging_for_tests() -> None:
    """Drop the cached configuration so tests can change LOG_FORMAT/LOG_LEVEL."""
    global _CONFIGURED
    logger.handlers.clear()
    _CONFIGURED = False


def _trace_hook():
    """Load ``OBSERVABILITY_TRACE_HOOK=module:function``, cached per spec.

    Not globally cached: the variable can change without a restart, and tests
    use different specs. Each spec is imported once.
    """
    spec = (os.getenv("OBSERVABILITY_TRACE_HOOK") or "").strip()
    if not spec or ":" not in spec:
        return None
    if spec in _HOOK_CACHE:
        return _HOOK_CACHE[spec]
    module_name, _, attr = spec.partition(":")
    hook = None
    try:
        import importlib

        hook = getattr(importlib.import_module(module_name), attr)
    except Exception:
        hook = None
    _HOOK_CACHE[spec] = hook
    return hook


def log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured event. This is the only logging entry point."""
    if not _CONFIGURED:
        configure_logging()
    fields = dict(fields)
    request_id = current_request_id()
    if request_id:
        # Captured at emission time, not at format time: a handler that formats
        # later (queue handler, async sink) still gets the right correlation id.
        fields.setdefault("request_id", request_id)
    logger.log(level, event, extra={"fields": fields})

    # Tracing backends (LangSmith/Langfuse/OTLP) are optional and env-gated.
    try:
        from core import observability_backends

        sink = observability_backends.resolve_sink()
    except Exception:
        sink = None
    if sink is not None:
        payload = {"event": event, "request_id": request_id, **fields}
        try:
            sink(payload)
        except Exception:
            # Tracing is best-effort; never let it break a request.
            logger.debug("trace_sink_failed", extra={"fields": {"event": event}})


@contextmanager
def timed(stage: str, **fields: Any):
    """Time a stage and emit ``stage.<name>`` with duration and caller metadata.

    Yields a mutable dict: adding keys to it (row counts, etc.) includes them in
    the event, which is how a caller records results known only after the work
    finishes.

    >>> with timed("question", attempt=1) as meta:
    ...     meta["mcqs"] = 5
    """
    meta = dict(fields)
    started = time.perf_counter()
    status = "ok"
    error: str | None = None
    try:
        yield meta
    except Exception as exc:
        status = "error"
        error = type(exc).__name__
        raise
    finally:
        meta.setdefault("duration_ms", round((time.perf_counter() - started) * 1000, 2))
        meta["status"] = status
        if error:
            meta["error_type"] = error
        log_event(f"stage.{stage}", **meta)


class RequestContextMiddleware:
    """Pure-ASGI middleware: request id, response header, one access log line.

    Pure ASGI (not BaseHTTPMiddleware) so the contextvar is visible to handlers
    and everything they call. A caller-supplied ``X-Request-ID`` is preserved,
    which makes it possible to correlate a frontend error with a backend log.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"")
        request_id = incoming.decode("latin-1")[:128] if incoming else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status: int | None = None

        async def send_wrapper(message: dict) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                response_headers = list(message.get("headers") or [])
                response_headers.append((b"x-request-id", request_id.encode("latin-1")))
                message["headers"] = response_headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            path = scope.get("path", "")
            # Static assets would otherwise dominate the log with noise.
            if not path.startswith("/static/"):
                log_event(
                    "http.request",
                    method=scope.get("method"),
                    path=path,
                    status=status,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            request_id_var.reset(token)
