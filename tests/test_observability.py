"""Observability: JSON formatting, request ids, stage timing, LLM usage, hooks."""
import json
import logging
import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.llm.base import LLMProvider
from core.observability import (
    JsonFormatter,
    RequestContextMiddleware,
    _reset_logging_for_tests,
    configure_logging,
    current_request_id,
    log_event,
    request_id_var,
    timed,
)

# Configure once so assertLogs' handler is not cleared by the first log_event.
configure_logging()

CAPTURED: list[dict] = []


def capture(payload: dict) -> None:
    """Trace hook used by the tests (OBSERVABILITY_TRACE_HOOK module:function)."""
    CAPTURED.append(payload)


class TestJsonLogging(unittest.TestCase):
    def test_formatter_emits_one_json_object_with_fields_and_request_id(self):
        token = request_id_var.set("req-123")
        try:
            record = logging.LogRecord("mcq", logging.INFO, __file__, 1, "unit.event",
                                       None, None)
            record.fields = {"count": 2, "status": "ok"}
            payload = json.loads(JsonFormatter().format(record))
        finally:
            request_id_var.reset(token)

        self.assertEqual(payload["event"], "unit.event")
        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["request_id"], "req-123")
        self.assertEqual(payload["count"], 2)
        self.assertIn("ts", payload)

    def test_log_event_records_fields(self):
        with self.assertLogs("mcq", level="INFO") as captured:
            log_event("unit.event", value=3)
        record = captured.records[0]
        self.assertEqual(record.getMessage(), "unit.event")
        self.assertEqual(record.fields["value"], 3)

    def test_timed_reports_duration_and_caller_metadata(self):
        with self.assertLogs("mcq", level="INFO") as captured:
            with timed("unit", attempt=1) as meta:
                meta["count"] = 7
        record = captured.records[0]
        self.assertEqual(record.getMessage(), "stage.unit")
        self.assertEqual(record.fields["attempt"], 1)
        self.assertEqual(record.fields["count"], 7)
        self.assertEqual(record.fields["status"], "ok")
        self.assertGreaterEqual(record.fields["duration_ms"], 0)

    def test_timed_marks_errors_and_reraises(self):
        with self.assertLogs("mcq", level="INFO") as captured:
            with self.assertRaises(RuntimeError):
                with timed("boom"):
                    raise RuntimeError("nope")
        record = captured.records[0]
        self.assertEqual(record.fields["status"], "error")
        self.assertEqual(record.fields["error_type"], "RuntimeError")


class _UsageProvider(LLMProvider):
    provider_id = "stub"

    def _complete_once(self, messages, *, temperature=None, max_tokens=None,
                       json_mode=False) -> str:
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5}
        return "{}"


class TestLLMCallLogging(unittest.TestCase):
    def test_successful_call_logs_timing_and_usage(self):
        provider = _UsageProvider("key", "stub-model")
        with self.assertLogs("mcq", level="INFO") as captured:
            provider.complete([{"role": "user", "content": "hi"}], max_tokens=10)
        record = next(r for r in captured.records if r.getMessage() == "llm.call")
        self.assertEqual(record.fields["status"], "ok")
        self.assertEqual(record.fields["provider"], "stub")
        self.assertEqual(record.fields["usage"]["completion_tokens"], 5)
        self.assertEqual(record.fields["prompt_chars"], 2)

    def test_failed_call_logs_error_without_prompt_text(self):
        class _Boom(LLMProvider):
            provider_id = "stub"

            def _complete_once(self, messages, **kwargs):
                raise ValueError("provider exploded")

        with self.assertLogs("mcq", level="ERROR") as captured:
            with self.assertRaises(ValueError):
                _Boom("key", "stub-model").complete(
                    [{"role": "user", "content": "secret document text"}])
        record = captured.records[0]
        self.assertEqual(record.fields["status"], "error")
        self.assertEqual(record.fields["error_type"], "ValueError")
        self.assertNotIn("secret document text", json.dumps(record.fields))


class TestRequestContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.add_middleware(RequestContextMiddleware)
        seen = {}

        @app.get("/ping")
        def ping():
            seen["request_id"] = current_request_id()
            log_event("inside.route")
            return {"ok": True}

        cls._app = app
        cls._seen = seen

    def test_supplied_request_id_is_propagated_and_echoed(self):
        client = TestClient(self._app)
        with self.assertLogs("mcq", level="INFO") as captured:
            response = client.get("/ping", headers={"X-Request-ID": "caller-id"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-request-id"], "caller-id")
        self.assertEqual(self._seen["request_id"], "caller-id")

        inside = next(r for r in captured.records if r.getMessage() == "inside.route")
        payload = json.loads(JsonFormatter().format(inside))
        self.assertEqual(payload["request_id"], "caller-id")

    def test_generated_request_id_is_echoed(self):
        response = TestClient(self._app).get("/ping")
        self.assertTrue(response.headers["x-request-id"])
        self.assertGreaterEqual(len(response.headers["x-request-id"]), 16)

    def test_access_log_records_status(self):
        client = TestClient(self._app)
        with self.assertLogs("mcq", level="INFO") as captured:
            client.get("/ping")
        access = next(r for r in captured.records if r.getMessage() == "http.request")
        self.assertEqual(access.fields["path"], "/ping")
        self.assertEqual(access.fields["status"], 200)
        self.assertGreaterEqual(access.fields["duration_ms"], 0)


class TestTraceHook(unittest.TestCase):
    def tearDown(self):
        CAPTURED.clear()

    def test_configured_hook_receives_events(self):
        with patch.dict(os.environ,
                        {"OBSERVABILITY_TRACE_HOOK": "tests.test_observability:capture"}):
            log_event("hooked.event", n=1)
        self.assertEqual(len(CAPTURED), 1)
        self.assertEqual(CAPTURED[0]["event"], "hooked.event")
        self.assertEqual(CAPTURED[0]["n"], 1)

    def test_broken_hook_never_breaks_logging(self):
        with patch.dict(os.environ, {"OBSERVABILITY_TRACE_HOOK": "no_such_mod:fn"}):
            log_event("still.fine")  # must not raise

    def test_logging_can_be_reconfigured_for_tests(self):
        _reset_logging_for_tests()
        import config

        with patch.multiple(config, LOG_FORMAT="text", LOG_LEVEL="DEBUG"):
            from core.observability import configure_logging

            logger = configure_logging()
            self.assertEqual(logger.level, logging.DEBUG)
            self.assertNotIsInstance(logger.handlers[0].formatter, JsonFormatter)
        _reset_logging_for_tests()


if __name__ == "__main__":
    unittest.main(verbosity=2)
