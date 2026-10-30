# Observability

The application emits one JSON object per line, on stdout. Hosted logs can be
queried by field without parsing prose, and every line produced while handling a
request carries the same `request_id`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | Standard logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `LOG_FORMAT` | `json` | `text` restores a human-readable local stream. |
| `OBSERVABILITY_TRACE_HOOK` | empty | Optional `module:function` called with every event (see below). |

No prompt text, document text, API key, or user content is logged. Fields are
identifiers, counters, durations, and error types.

## Example

```json
{"ts":"2026-10-29T15:40:12.184+00:00","level":"INFO","event":"llm.call","request_id":"9f1c...","provider":"groq","model":"openai/gpt-oss-20b","status":"ok","attempt":1,"duration_ms":812.4,"json_mode":true,"prompt_chars":4210,"max_tokens":2600,"usage":{"prompt_tokens":1180,"completion_tokens":642,"total_tokens":1822}}
```

## Request ids

`RequestContextMiddleware` wraps the whole ASGI stack:

* a caller-supplied `X-Request-ID` is preserved (so a frontend error report can
  be matched to a backend log), otherwise a UUID is generated;
* the id is available via `core.observability.current_request_id()` and is added
  to every event logged during the request;
* the response carries `X-Request-ID` so a browser or proxy can record it.

Static asset requests (`/static/...`) are excluded from access logging.

## Event reference

| Event | Key fields |
| --- | --- |
| `http.request` | `method`, `path`, `status`, `duration_ms` |
| `llm.call` | `provider`, `model`, `status`, `attempt`, `duration_ms`, `json_mode`, `prompt_chars`, `max_tokens`, `usage` |
| `llm.rate_limited` | `provider`, `model`, `attempt`, `duration_ms`, `retry_after` |
| `llm.retry` | `provider`, `model`, `attempt`, `max_retries`, `wait_seconds` |
| `retrieval.complete` | `document_id`, `chunks`, `avg_score`, `low_confidence` |
| `context.validated` | `sufficient`, `reason` |
| `stage.question` | `attempt`, `requested`, `mcqs`, `duration_ms`, `status` |
| `stage.quality_assurance` | `attempt`, `valid`, `rejected`, `duration_ms`, `status` |
| `stage.fact_verification` | `attempt`, `passed`, `rejected`, `duration_ms`, `status` |
| `stage.difficulty` | `attempt`, `matched`, `duration_ms`, `status` |
| `pipeline.complete` | `requested`, `collected`, `attempts`, `difficulty`, `duration_ms` |
| `pipeline.legacy_fallback` | `reason` |
| `job.created` / `job.enqueued` / `job.started` | `job_id`, `document_id`, counts |
| `job.completed` / `job.failed` | `job_id`, `count` or `error_type`/`error`, `duration_ms` |

`usage` is the raw provider object (OpenAI-compatible `usage`, Gemini
`usageMetadata`, Anthropic `usage`) and is `null` when the provider omits it.

## Pipeline stage timings

`_collect_mcqs` wraps each agent call in `core.observability.timed(stage)`, which
emits `stage.<name>` with a `duration_ms` and a status of `ok` or `error`. The
final `pipeline.complete` event gives end-to-end time and how many questions were
accepted versus requested, which is the number to watch for under-delivery.

## Trace hook (LangSmith / Langfuse / OTLP)

The repository deliberately does not depend on an observability SDK. Instead,
set `OBSERVABILITY_TRACE_HOOK` to any importable callable:

```bash
OBSERVABILITY_TRACE_HOOK=my_observability:on_event
```

```python
# my_observability.py -- adapters stay outside the app
from langsmith import Client            # or: from langfuse import Langfuse

client = Client()

def on_event(event: dict) -> None:
    # only forward the events you care about; never block the request path
    if event["event"] in ("llm.call", "pipeline.complete"):
        client.create_run(
            name=event["event"],
            run_type="llm" if event["event"] == "llm.call" else "chain",
            inputs={},
            outputs={k: v for k, v in event.items() if k != "event"},
        )
```

The hook is called in-process and its exceptions are swallowed, so a tracing
outage cannot fail a generation request. Hooks run synchronously; keep them
cheap or buffer to a background thread. The same mechanism works for an OTLP
exporter or a custom warehouse sink.

## Built-in LangSmith / Langfuse sinks

Set `OBSERVABILITY_BACKEND=langsmith` (and `LANGSMITH_API_KEY`) or
`OBSERVABILITY_BACKEND=langfuse` (and `LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY`, optionally `LANGFUSE_HOST`) and events are forwarded
through the optional SDKs. With `OBSERVABILITY_BACKEND=auto` (the default) the
backend is chosen from whichever credentials are present. Neither SDK is a
dependency: if it is not installed, or the call fails, tracing is silently
disabled and JSON logging continues. `OBSERVABILITY_BACKEND=none` turns all
tracing off.

## Local development

```bash
LOG_FORMAT=text LOG_LEVEL=DEBUG python app.py
```

`python worker.py` configures the same logger, so web and worker logs share the
schema and can be correlated by `job_id`.
