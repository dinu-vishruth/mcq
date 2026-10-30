"""Optional tracing backends: LangSmith, Langfuse, and the selector."""
import os
import sys
import types
import unittest
from unittest.mock import patch

from core import observability_backends as backends


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.runs = []
        self.events = []

    def create_run(self, **kwargs):
        self.runs.append(kwargs)

    def event(self, **kwargs):
        self.events.append(kwargs)

    def flush(self):
        pass


class TestLangSmithSink(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("langsmith", None)

    def test_noops_without_sdk(self):
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "secret"}, clear=False):
            sys.modules.pop("langsmith", None)
            backends.langsmith_sink({"event": "llm.call"})  # must not raise

    def test_noops_without_credentials(self):
        fake = types.ModuleType("langsmith")
        fake.Client = _FakeClient
        sys.modules["langsmith"] = fake
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANGSMITH_API_KEY", None)
            backends.langsmith_sink({"event": "llm.call"})  # must not raise
        self.assertEqual(getattr(fake, "_used", None), None)

    def test_forwards_event_when_sdk_and_key_present(self):
        fake = types.ModuleType("langsmith")
        fake.Client = _FakeClient
        sys.modules["langsmith"] = fake
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "secret"}):
            backends.langsmith_sink({"event": "llm.call", "provider": "groq"})
        # The client instance is created inside the sink; verify via a hook.
        # Re-run capturing the created client.
        captured = {}

        def factory(**kwargs):
            client = _FakeClient(**kwargs)
            captured["client"] = client
            return client

        fake.Client = factory
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "secret"}):
            backends.langsmith_sink({"event": "pipeline.complete", "count": 3})
        self.assertIn("client", captured)
        self.assertEqual(len(captured["client"].runs), 1)
        self.assertEqual(captured["client"].runs[0]["run_type"], "chain")
        self.assertEqual(captured["client"].runs[0]["outputs"]["count"], 3)

    def test_llm_call_uses_llm_run_type(self):
        captured = {}

        def factory(**kwargs):
            client = _FakeClient(**kwargs)
            captured["client"] = client
            return client

        fake = types.ModuleType("langsmith")
        fake.Client = factory
        sys.modules["langsmith"] = fake
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "secret"}):
            backends.langsmith_sink({"event": "llm.call", "provider": "groq"})
        self.assertEqual(captured["client"].runs[0]["run_type"], "llm")

    def test_swallows_sdk_errors(self):
        fake = types.ModuleType("langsmith")

        def factory(**kwargs):
            raise RuntimeError("boom")

        fake.Client = factory
        sys.modules["langsmith"] = fake
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "secret"}):
            backends.langsmith_sink({"event": "x"})  # must not raise


class TestLangfuseSink(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("langfuse", None)

    def test_noops_without_credentials(self):
        fake = types.ModuleType("langfuse")
        fake.Langfuse = _FakeClient
        sys.modules["langfuse"] = fake
        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        os.environ.pop("LANGFUSE_SECRET_KEY", None)
        backends.langfuse_sink({"event": "x"})  # must not raise

    def test_forwards_event_when_sdk_and_keys_present(self):
        captured = {}

        def factory(**kwargs):
            client = _FakeClient(**kwargs)
            captured["client"] = client
            return client

        fake = types.ModuleType("langfuse")
        fake.Langfuse = factory
        sys.modules["langfuse"] = fake
        with patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}):
            backends.langfuse_sink({"event": "llm.call", "n": 1})
        self.assertEqual(len(captured["client"].events), 1)
        self.assertEqual(captured["client"].events[0]["name"], "llm.call")

    def test_swallows_sdk_errors(self):
        fake = types.ModuleType("langfuse")

        def factory(**kwargs):
            raise RuntimeError("boom")

        fake.Langfuse = factory
        sys.modules["langfuse"] = fake
        with patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}):
            backends.langfuse_sink({"event": "x"})  # must not raise


class TestResolveSink(unittest.TestCase):
    def setUp(self):
        # Clear env that could steer auto-detection.
        for name in ("LANGSMITH_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
                     "OBSERVABILITY_TRACE_HOOK", "OBSERVABILITY_BACKEND"):
            os.environ.pop(name, None)
        import core.observability as obs

        obs._HOOK_CACHE.clear()

    def tearDown(self):
        for name in ("LANGSMITH_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
                     "OBSERVABILITY_TRACE_HOOK", "OBSERVABILITY_BACKEND"):
            os.environ.pop(name, None)
        import core.observability as obs

        obs._HOOK_CACHE.clear()

    def test_none_when_nothing_configured(self):
        self.assertIsNone(backends.resolve_sink())

    def test_none_backend_disables_tracing(self):
        with patch.dict(os.environ, {"OBSERVABILITY_BACKEND": "none"}):
            self.assertIsNone(backends.resolve_sink())

    def test_langsmith_selected_when_key_present(self):
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "secret"}):
            self.assertIs(backends.resolve_sink(), backends.langsmith_sink)

    def test_langfuse_selected_when_keys_present(self):
        with patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}):
            self.assertIs(backends.resolve_sink(), backends.langfuse_sink)

    def test_explicit_backend_overrides_auto(self):
        with patch.dict(os.environ, {"OBSERVABILITY_BACKEND": "langfuse",
                                     "LANGSMITH_API_KEY": "secret"}):
            self.assertIs(backends.resolve_sink(), backends.langfuse_sink)

    def test_hook_backend_uses_generic_hook(self):
        captured = []

        def hook(payload):
            captured.append(payload)

        import core.observability as obs

        obs._HOOK_CACHE["tests.test_observability_backends:capture"] = hook
        with patch.dict(os.environ, {"OBSERVABILITY_BACKEND": "hook",
                                     "OBSERVABILITY_TRACE_HOOK": "tests.test_observability_backends:capture"}):
            sink = backends.resolve_sink()
            self.assertIsNotNone(sink)
            sink({"event": "x"})
            self.assertEqual(captured, [{"event": "x"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
