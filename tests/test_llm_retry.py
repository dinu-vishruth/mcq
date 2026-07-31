"""
LLM 429 retry/backoff tests. Fully offline — a fake provider raises
RateLimitError a controlled number of times; time.sleep is monkeypatched so the
suite doesn't actually wait.

Run:  python -m unittest tests.test_llm_retry -v
"""
import unittest

import config
from core.llm.base import LLMProvider, LLMError, RateLimitError


class _FlakyProvider(LLMProvider):
    """Raises RateLimitError for the first `fail_times` calls, then succeeds."""
    def __init__(self, fail_times, retry_after=None):
        super().__init__(api_key="k", model="m")
        self.fail_times = fail_times
        self.calls = 0
        self.retry_after = retry_after

    def _complete_once(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RateLimitError("429", retry_after=self.retry_after)
        return '{"ok": true}'


class TestRetry(unittest.TestCase):
    def setUp(self):
        # Don't actually sleep during tests; record the waits instead.
        import core.llm.base as base_mod
        self._orig_sleep = None
        self.waits = []

        import time as _time
        self._orig_sleep = _time.sleep
        _time.sleep = lambda s: self.waits.append(s)
        self.addCleanup(lambda: setattr(_time, "sleep", self._orig_sleep))

        # Deterministic, small backoff config.
        self._orig = (config.LLM_MAX_RETRIES, config.LLM_RETRY_BACKOFF, config.LLM_RETRY_MAX_WAIT)
        config.LLM_MAX_RETRIES = 3
        config.LLM_RETRY_BACKOFF = 1.0
        config.LLM_RETRY_MAX_WAIT = 8.0
        self.addCleanup(self._restore)

    def _restore(self):
        (config.LLM_MAX_RETRIES, config.LLM_RETRY_BACKOFF, config.LLM_RETRY_MAX_WAIT) = self._orig

    def test_succeeds_after_transient_429s(self):
        p = _FlakyProvider(fail_times=2)
        out = p.complete_json([{"role": "user", "content": "hi"}])
        self.assertEqual(out, {"ok": True})
        self.assertEqual(p.calls, 3)                 # 2 failures + 1 success
        self.assertEqual(self.waits, [1.0, 2.0])     # exponential backoff

    def test_gives_up_after_max_retries(self):
        p = _FlakyProvider(fail_times=99)
        with self.assertRaises(RateLimitError):
            p.complete_json([{"role": "user", "content": "hi"}])
        self.assertEqual(p.calls, 4)                 # 1 initial + 3 retries

    def test_honors_retry_after_header(self):
        p = _FlakyProvider(fail_times=1, retry_after=5.0)
        p.complete_json([{"role": "user", "content": "hi"}])
        self.assertEqual(self.waits, [5.0])          # used the provider's hint

    def test_retry_after_capped_by_max_wait(self):
        p = _FlakyProvider(fail_times=1, retry_after=999.0)
        p.complete_json([{"role": "user", "content": "hi"}])
        self.assertEqual(self.waits, [8.0])          # clamped to LLM_RETRY_MAX_WAIT

    def test_no_retries_when_disabled(self):
        config.LLM_MAX_RETRIES = 0
        p = _FlakyProvider(fail_times=1)
        with self.assertRaises(RateLimitError):
            p.complete_json([{"role": "user", "content": "hi"}])
        self.assertEqual(p.calls, 1)
        self.assertEqual(self.waits, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
