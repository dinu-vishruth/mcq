"""Request-local deadlines shared by retries, pipeline stages and fallbacks.

Nested budgets can shorten an existing deadline, never extend it. The context
also propagates through Starlette's thread-pool helpers without shared mutation.
"""
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from core.llm.base import LLMError

_deadline = ContextVar("ai_deadline", default=None)


class BudgetExceeded(LLMError):
    def __init__(self):
        super().__init__("Quiz processing reached its time limit. Try fewer questions or retry shortly.")


def remaining(default: float | None = None) -> float | None:
    deadline = _deadline.get()
    if deadline is None:
        return default
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise BudgetExceeded()
    return min(default, seconds) if default is not None else seconds


@contextmanager
def time_budget(seconds: float):
    deadline = time.monotonic() + max(0, seconds)
    previous = _deadline.get()
    token = _deadline.set(min(previous, deadline) if previous is not None else deadline)
    try:
        remaining()
        yield
    finally:
        _deadline.reset(token)


def generation_budget(fn):
    @wraps(fn)
    def bounded(*args, **kwargs):
        import config
        seconds = kwargs.get("deadline_seconds")
        if seconds is None and remaining() is not None:
            return fn(*args, **kwargs)
        seconds = seconds if seconds is not None else config.PIPELINE_DEADLINE_SECONDS
        with time_budget(seconds):
            return fn(*args, **kwargs)
    return bounded


def http_timeout(default: float):
    # urllib3's total budget covers connection setup plus response waiting.
    from urllib3.util import Timeout
    seconds = remaining(default)
    return Timeout(total=seconds, connect=min(5.0, seconds), read=seconds)
