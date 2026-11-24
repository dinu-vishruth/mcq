"""Request-local monotonic budget shared by generation, checking and recovery."""
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_deadline = ContextVar("generation_deadline", default=None)


class GenerationDeadlineExceeded(TimeoutError):
    pass


def remaining(*, reserve=0.0):
    deadline = _deadline.get()
    return None if deadline is None else max(0.0, deadline - time.monotonic() - reserve)


def check(*, reserve=4.0):
    seconds = remaining(reserve=reserve)
    if seconds is not None and seconds <= 0:
        raise GenerationDeadlineExceeded("Generation time budget exhausted. Please try fewer questions.")


def network_timeout(ceiling):
    from urllib3.util import Timeout
    check()
    seconds = remaining(reserve=4)
    if seconds is None:
        return ceiling
    # Leave room for subsequent validation calls, not just question generation.
    cap = min(float(ceiling), max(0.05, seconds / 3))
    return Timeout(total=cap, connect=min(5, cap), read=cap)


@contextmanager
def budget(seconds):
    current = _deadline.get()
    # Nested strategies inherit the existing deadline; fallback never resets it.
    token = _deadline.set(time.monotonic() + max(0, seconds)) if current is None else None
    from core.services import generation_spend
    spend_token = generation_spend.begin() if token is not None else None
    try:
        yield
    finally:
        if token is not None:
            _deadline.reset(token)
            generation_spend.end(spend_token)


def generation_scope(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        import config
        seconds = kwargs.get("deadline_seconds") or config.PIPELINE_DEADLINE_SECONDS
        with budget(seconds):
            check()
            return function(*args, **kwargs)
    return wrapped
