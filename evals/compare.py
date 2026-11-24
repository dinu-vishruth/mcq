"""Compare the two candidate generators against the same bounded source text.

Run with ``python -m evals.compare --dry-run`` to check the workflow. A live
run requires ``--live`` and uses the configured provider with a fresh request
budget per arm. The results describe candidate generation and validation;
ingestion, retrieval, queue wait, and end-to-end recovery are outside scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import config
from core.services.generation_budget import budget
from core.services.generation_spend import snapshot
from evals.isolation import isolated_evaluation
from evals.run_eval import DEFAULT_DATASET, generate_stub, load_cases, score_case

Strategy = Literal["primary", "fallback"]
_STRATEGIES: tuple[Strategy, ...] = ("primary", "fallback")
_DEFAULT_OUTPUT = DEFAULT_DATASET.parent / "results" / "comparison.json"
Generator = Callable[[dict, str, Strategy], list[dict]]


def _live_candidates(case: dict, evidence: str, strategy: Strategy) -> list[dict]:
    from core.services.mcq_pipeline import evaluate_candidates

    return evaluate_candidates(evidence, case["num_questions"], case["difficulty"],
                               topic=case.get("topic"), strategy=strategy)


def _stub_candidates(case: dict, evidence: str, _strategy: Strategy) -> list[dict]:
    # Exercise the report with both successful and withheld fixtures. This
    # function deliberately does not test the production publication policy.
    if case.get("expected_outcome") == "withhold":
        return []
    return generate_stub({**case, "document": evidence})


def _run_arm(case: dict, evidence: str, strategy: Strategy, generate: Generator,
             mode: str) -> tuple[dict, list[dict]]:
    from core.services.generation_metrics import track_generation

    with track_generation() as generation, budget(config.PIPELINE_DEADLINE_SECONDS):
        start = time.perf_counter()
        error = None
        try:
            questions = generate(case, evidence, strategy)
            if not isinstance(questions, list):
                raise TypeError("Expected an MCQ list.")
        except Exception as exc:
            # Report an error category, never the raw provider body or source.
            error = type(exc).__name__
            questions = []
        elapsed = time.perf_counter() - start
        metrics = generation.snapshot()
        reservations = snapshot()

    withhold = (not questions and not error and not metrics["errors"] and not metrics["unverified"]
                and bool(mode == "stub" or metrics["refusals"] or metrics["rejected"]))
    result = score_case(case, questions, elapsed, evidence=evidence, withheld=withhold)
    # Keep the comparison report shareable without copying source keywords.
    result.pop("matched_keywords", None)
    result.update(strategy=strategy, seconds=round(elapsed, 6), **metrics,
                  llm_attempts=reservations["calls"], reserved_tokens=reservations["reserved_tokens"],
                  reserved_usd=reservations["reserved_usd"] if config.GENERATION_MAX_COST_USD > 0 else None)
    if error or (not questions and (metrics["errors"] or metrics["unverified"])):
        result.update(passed=False, status="failed", error_type=error or "UnverifiedGeneration")
    return result, questions


def _summary(runs: list[dict]) -> dict:
    timings = sorted(run["seconds"] for run in runs)
    reserved_usd = [run["reserved_usd"] for run in runs]
    return {
        "runs": len(runs),
        **{status: sum(run["status"] == status for run in runs)
           for status in ("complete", "partial", "failed", "withheld")},
        "deterministic_passes": sum(run["passed"] for run in runs),
        "p50_seconds": round(statistics.median(timings), 6) if timings else None,
        "p95_seconds": timings[math.ceil(0.95 * len(timings)) - 1] if timings else None,
        "llm_attempts": sum(run["llm_attempts"] for run in runs),
        "reserved_tokens": sum(run["reserved_tokens"] for run in runs),
        "reserved_usd": round(sum(reserved_usd), 6) if reserved_usd and None not in reserved_usd else None,
        "rejected_candidates": sum(run["rejected"] for run in runs),
        "unverified_candidates": sum(run["unverified"] for run in runs),
        "refusals": sum(run["refusals"] for run in runs),
        "generation_errors": sum(run["errors"] for run in runs),
    }


def compare_cases(cases: list[dict], *, repeats: int, generate: Generator, mode: str,
                  model: str | None, review_rows: list[dict] | None = None) -> dict:
    """Run each pair once per repeat, alternating which arm goes first."""
    pairs = []
    for case in cases:
        evidence = case["document"][:config.CONTEXT_MAX_CHARS]
        fingerprint = hashlib.sha256(evidence.encode("utf-8")).hexdigest()
        for repeat in range(1, repeats + 1):
            order = _STRATEGIES if len(pairs) % 2 == 0 else _STRATEGIES[::-1]
            arms = {}
            for strategy in order:
                result, questions = _run_arm(case, evidence, strategy, generate, mode)
                arms[strategy] = result
                if review_rows is not None:
                    review_rows.append({
                        "case_id": case["id"], "strategy": strategy, "repeat": repeat,
                        "document": case["document"], "context": evidence, "topic": case.get("topic"),
                        "difficulty": case["difficulty"], "requested_count": case["num_questions"],
                        "questions": questions,
                    })
            pairs.append({
                "id": case["id"], "repeat": repeat, "order": list(order),
                "evidence_sha256": fingerprint, "evidence_characters": len(evidence),
                "difficulty": case["difficulty"], "arms": arms,
                "fallback_minus_primary_seconds": round(arms["fallback"]["seconds"] - arms["primary"]["seconds"], 6),
            })

    return {
        "mode": mode,
        "model": model,
        "repeats": repeats,
        "measurement_scope": "Candidate generation and shared validation on identical bounded source text.",
        "quality_assessment": "unmeasured; deterministic checks and separate blinded review",
        "latency_percentiles": "p50 median and p95 nearest rank; all attempts included",
        "cost_basis": "Budget reservations, not billed usage. USD is null when a cost cap is disabled.",
        "summary": {strategy: _summary([pair["arms"][strategy] for pair in pairs]) for strategy in _STRATEGIES},
        "pairs": pairs,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Use offline fixtures to exercise reporting.")
    mode.add_argument("--live", action="store_true", help="Allow cost-bearing calls to the configured LLM.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, help="Use only the first N cases.")
    parser.add_argument("--repeats", type=int, default=2, help="Repeat each pair N times (default 2).")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--review-packet", type=Path, help="Optional blinded review JSON containing source text.")
    parser.add_argument("--review-key", type=Path, help="Write the strategy mapping separately from the review packet.")
    args = parser.parse_args(argv)
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error("repeats and limit must be positive")
    if (args.review_packet is None) != (args.review_key is None):
        parser.error("review-packet and review-key must be specified together")
    paths = [path.resolve() for path in (args.dataset, args.output, args.review_packet, args.review_key) if path is not None]
    if len(set(paths)) != len(paths):
        parser.error("dataset and output paths must be distinct")

    cases = load_cases(args.dataset, args.limit)
    review_rows = [] if args.review_packet else None
    with isolated_evaluation():
        model = None
        if args.live:
            from core.llm import LLMError, get_llm

            if not config.llm_key_present():
                print("Live evaluation requires configured LLM credentials.", file=sys.stderr)
                return 2
            try:
                provider = get_llm()
            except LLMError:
                print("Could not configure the live evaluation provider.", file=sys.stderr)
                return 2
            model = f"{getattr(provider, 'provider_id', config.LLM_PROVIDER)}/{provider.model}"
        report = compare_cases(cases, repeats=args.repeats, mode="live" if args.live else "stub",
                               model=model, generate=_live_candidates if args.live else _stub_candidates,
                               review_rows=review_rows)

    _write_json(args.output, report)
    if review_rows is not None:
        from evals.review import build_review_packets

        packet, key = build_review_packets(review_rows)
        packet["mode"] = report["mode"]
        _write_json(args.review_packet, packet)
        _write_json(args.review_key, key)
    print(f"Recorded {len(report['pairs'])} pairs ({report['mode']}) in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
