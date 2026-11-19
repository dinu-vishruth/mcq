"""Optional tracing backends for LangSmith and Langfuse.

Neither SDK is a dependency of this project: the imports below are guarded, and
every sink no-ops when its SDK or credentials are missing. This keeps tracing
pluggable without forcing an extra install on operators who only want JSON logs.
"""
from __future__ import annotations

import os
from typing import Any, Callable


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def langsmith_sink(event: dict[str, Any]) -> None:
    """Forward an event to LangSmith via its (optional) SDK."""
    try:
        import langsmith
    except ImportError:
        return
    api_key = _env("LANGSMITH_API_KEY")
    if not api_key:
        return
    try:
        client = langsmith.Client(api_key=api_key)
        client.create_run(
            name=str(event.get("event", "event")),
            run_type="llm" if event.get("event") == "llm.call" else "chain",
            inputs={},
            outputs={k: v for k, v in event.items() if k != "event"},
        )
    except Exception:
        # Tracing must never fail a request.
        pass


def langfuse_sink(event: dict[str, Any]) -> None:
    """Forward an event to Langfuse via its (optional) SDK."""
    try:
        import langfuse
    except ImportError:
        return
    public_key = _env("LANGFUSE_PUBLIC_KEY")
    secret_key = _env("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        return
    try:
        client = langfuse.Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=_env("LANGFUSE_HOST") or None,
        )
        client.event(name=str(event.get("event", "event")), metadata=event)
        client.flush()
    except Exception:
        pass


def _trace_hook() -> Callable[[dict[str, Any]], None] | None:
    # Reuse the generic hook implementation in core.observability.
    from core.observability import _trace_hook as _generic

    return _generic()


def resolve_sink() -> Callable[[dict[str, Any]], None] | None:
    """Return the sink selected by OBSERVABILITY_BACKEND / available credentials."""
    backend = (_env("OBSERVABILITY_BACKEND") or "auto").lower()
    if backend in ("none", "off", "disabled", "false", "0"):
        return None
    if backend == "langsmith":
        return langsmith_sink
    if backend == "langfuse":
        return langfuse_sink
    if backend == "hook":
        return _trace_hook()
    # auto: prefer a configured SDK, then the generic hook.
    if _env("LANGSMITH_API_KEY"):
        return langsmith_sink
    if _env("LANGFUSE_PUBLIC_KEY") and _env("LANGFUSE_SECRET_KEY"):
        return langfuse_sink
    return _trace_hook()
