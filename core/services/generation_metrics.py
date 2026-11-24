"""Count actual work and accepted questions within one generation request."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass


@dataclass
class GenerationMetrics:
    attempts: int = 0
    rejected: int = 0
    unverified: int = 0
    errors: int = 0
    refusals: int = 0
    fallback_attempted: bool = False
    fallback_count: int = 0

    def snapshot(self) -> dict:
        return asdict(self)

    def result(self, session_key: str, *, count: int, requested: int) -> dict:
        """Public completion payload. Counts come from persisted accepted questions."""
        return {
            "session_key": session_key,
            "count": count,
            "requested_count": requested,
            "partial": count < requested,
            "quality_policy": "verified_only",
            "fallback_attempted": self.fallback_attempted,
            "fallback_count": self.fallback_count,
        }


_current: ContextVar[GenerationMetrics | None] = ContextVar("generation_metrics", default=None)


@contextmanager
def track_generation():
    metrics = GenerationMetrics()
    token = _current.set(metrics)
    try:
        yield metrics
    finally:
        _current.reset(token)


def current_metrics() -> GenerationMetrics | None:
    return _current.get()
