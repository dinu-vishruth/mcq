"""Paired comparison contracts. Providers and databases never leave the test process."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from core.services.generation_metrics import current_metrics
from core.services.generation_spend import reserve, snapshot
from evals.compare import compare_cases, main
from evals.run_eval import DEFAULT_DATASET, generate_stub, load_cases


def test_paired_runs_share_settings_and_start_with_independent_budgets(monkeypatch):
    case = load_cases(DEFAULT_DATASET, limit=1)[0]
    monkeypatch.setattr(config, "CONTEXT_MAX_CHARS", 120)
    monkeypatch.setattr(config, "GENERATION_MAX_COST_USD", 0.0)
    calls = []

    def generate(settings, evidence, strategy):
        assert snapshot()["calls"] == 0
        reserve([{"content": evidence}], 20)
        calls.append((strategy, evidence, settings["difficulty"], settings["num_questions"], settings["topic"]))
        questions = generate_stub({**settings, "document": evidence})
        return questions if strategy == "primary" else questions[:2]

    report = compare_cases([case], repeats=2, generate=generate, mode="live", model="mock/test")
    assert [strategy for strategy, *_ in calls] == ["primary", "fallback", "fallback", "primary"]
    assert all(call[1:] == (case["document"][:120], case["difficulty"], case["num_questions"], case["topic"])
               for call in calls)
    assert report["summary"]["primary"]["complete"] == 2
    assert report["summary"]["fallback"]["partial"] == 2
    for summary in report["summary"].values():
        assert summary["llm_attempts"] == 2
        assert summary["reserved_tokens"] > 0
        assert summary["reserved_usd"] is None
    assert len({pair["evidence_sha256"] for pair in report["pairs"]}) == 1
    assert case["document"] not in json.dumps(report)
    assert snapshot() is None


def test_runtime_failure_or_unexplained_empty_output_cannot_pass_a_refusal_case():
    case = {**load_cases(DEFAULT_DATASET, limit=1)[0], "expected_outcome": "withhold"}

    def generate(_case, _evidence, strategy):
        if strategy == "primary":
            raise RuntimeError("private provider response")
        return []

    report = compare_cases([case], repeats=1, generate=generate, mode="live", model="mock/test")
    for strategy in report["summary"]:
        assert report["summary"][strategy]["failed"] == 1
        assert report["summary"][strategy]["deterministic_passes"] == 0
    assert report["pairs"][0]["arms"]["primary"]["error_type"] == "RuntimeError"
    assert "private provider response" not in json.dumps(report)


def test_refusal_credit_excludes_unverified_candidates():
    case = {**load_cases(DEFAULT_DATASET, limit=1)[0], "expected_outcome": "withhold"}

    def generate(_case, _evidence, strategy):
        metrics = current_metrics()
        metrics.refusals += 1
        if strategy == "fallback":
            metrics.unverified += 1
        return []

    report = compare_cases([case], repeats=1, generate=generate, mode="live", model="mock/test")
    assert report["summary"]["primary"]["deterministic_passes"] == 1
    assert report["summary"]["primary"]["withheld"] == 1
    assert report["summary"]["fallback"]["deterministic_passes"] == 0
    assert report["summary"]["fallback"]["failed"] == 1


def test_dry_run_is_explicit_and_uses_no_provider(monkeypatch, tmp_path):
    def forbidden():
        pytest.fail("dry-run attempted to configure a provider")

    monkeypatch.setattr("core.llm.get_llm", forbidden)
    output = tmp_path / "compare.json"
    assert main(["--dry-run", "--limit", "1", "--repeats", "2", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "stub"
    assert report["model"] is None
    assert len(report["pairs"]) == 2
    assert report["summary"]["primary"]["llm_attempts"] == 0
    assert "unmeasured" in report["quality_assessment"]


def test_live_path_uses_temporary_storage_and_keeps_requested_topic(monkeypatch, tmp_path):
    url = "postgresql://must-not-be-used.invalid/examly"
    monkeypatch.setattr(config, "DATABASE_URL", url)
    monkeypatch.setattr(config, "llm_key_present", lambda: True)
    monkeypatch.setattr("core.llm.get_llm", lambda: SimpleNamespace(provider_id="mock", model="test"))
    calls = []

    def fake_candidates(evidence, count, difficulty, *, topic, strategy):
        assert config.DATABASE_URL == ""
        assert Path(config.DB_PATH).parent.name.startswith("examly_eval_")
        calls.append((evidence, count, difficulty, topic, strategy))
        return []

    monkeypatch.setattr("core.services.mcq_pipeline.evaluate_candidates", fake_candidates)
    output = tmp_path / "compare.json"
    assert main(["--live", "--limit", "1", "--repeats", "1", "--output", str(output)]) == 0
    case = load_cases(DEFAULT_DATASET, limit=1)[0]
    settings = (case["document"][:config.CONTEXT_MAX_CHARS], case["num_questions"], case["difficulty"], case["topic"])
    assert calls == [(*settings, "primary"), (*settings, "fallback")]
    assert config.DATABASE_URL == url
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["model"] == "mock/test"
    assert url not in output.read_text(encoding="utf-8")


def test_review_packet_and_mapping_paths_must_be_distinct(tmp_path):
    path = str(tmp_path / "same.json")
    with pytest.raises(SystemExit) as exc:
        main(["--dry-run", "--review-packet", path, "--review-key", path])
    assert exc.value.code == 2


def test_review_packet_does_not_disclose_strategy_and_preserves_requested_count(monkeypatch, tmp_path):
    def partial_fixture(case, evidence, _strategy):
        return generate_stub({**case, "document": evidence})[:1]

    monkeypatch.setattr("evals.compare._stub_candidates", partial_fixture)
    output, packet_path, key_path = (tmp_path / name for name in ("result.json", "packet.json", "key.json"))
    assert main(["--dry-run", "--limit", "1", "--repeats", "1", "--output", str(output),
                 "--review-packet", str(packet_path), "--review-key", str(key_path)]) == 0
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    expected = load_cases(DEFAULT_DATASET, limit=1)[0]["num_questions"]
    assert packet["mode"] == "stub"
    assert len(packet["items"]) == 2
    assert all("strategy" not in item and item["requested_count"] == expected and len(item["questions"]) == 1
               for item in packet["items"])
    assert {row["strategy"] for row in key["runs"].values()} == {"primary", "fallback"}
