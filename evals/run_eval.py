"""MCQ evaluation harness.

Runs the real generation pipeline over a small JSONL dataset and scores the
output with deterministic metrics, then writes machine-readable and Markdown
reports under ``evals/results/``. It is manual by design: a meaningful run calls
a real LLM (cost + credentials), so it is not part of CI.

Usage
-----
    # Real provider configured via the usual LLM_* variables:
    python evals/run_eval.py --live

    # Deterministic stub -- verifies the harness end-to-end without a key:
    python evals/run_eval.py --dry-run

    # Limit cases / choose output paths:
    python evals/run_eval.py --live --limit 3 --output evals/results/latest.json

Metrics per case (all in [0, 1] unless noted):
  count_ratio        generated / requested
  validity           fraction of questions that satisfy the MCQ contract
  grounding_contract_rate   fraction with the required explanation and quote
  source_quote_rate          fraction whose quote appears in supplied evidence
  answer_text_match_rate    fraction whose answer appears verbatim in evidence
  keyword_coverage          expected terms matched in stems and answers
  duplicate_rate            duplicate normalized stems / generated
  passed                    deterministic contract checks only

Substrings and keyword matches cannot establish factual correctness,
difficulty, ambiguity, or explanation quality. Use the human review packet for
those judgments.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from core.models.mcq_contract import MCQ, GroundedMCQ, normalized_text, question_key  # noqa: E402

DEFAULT_DATASET = ROOT / "evals" / "dataset.jsonl"
DEFAULT_OUTPUT = ROOT / "evals" / "results" / "latest.json"
DEFAULT_MARKDOWN = ROOT / "evals" / "results" / "latest.md"

MIN_COUNT_RATIO = 0.8
MIN_VALIDITY = 1.0
MIN_KEYWORD_COVERAGE = 0.6


def _norm(text: str) -> str:
    return normalized_text(text) if isinstance(text, str) else ""


def load_cases(path: Path, limit: int | None = None) -> list[dict]:
    cases = []
    case_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON ({exc})")
            if not isinstance(case, dict):
                raise SystemExit(f"{path}:{line_number}: each case must be an object")
            case.setdefault("num_questions", 5)
            case.setdefault("difficulty", "medium")
            case.setdefault("topic", None)
            case.setdefault("expected_keywords", [])
            case.setdefault("id", f"case-{line_number}")
            case.setdefault("expected_outcome", "generate")
            case_id, document = case["id"], case.get("document")
            requested, keywords = case["num_questions"], case["expected_keywords"]
            if (not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids
                    or not isinstance(document, str) or not document.strip()
                    or type(requested) is not int or not 1 <= requested <= 30
                    or case["difficulty"] not in ("easy", "medium", "hard")
                    or not isinstance(case["topic"], (str, type(None)))
                    or not isinstance(keywords, list) or any(not _norm(term) for term in keywords)
                    or case["expected_outcome"] not in ("generate", "withhold")):
                raise SystemExit(f"{path}:{line_number}: invalid case fields or duplicate case ID")
            case_ids.add(case_id)
            cases.append(case)
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise SystemExit(f"No cases found in {path}")
    return cases


def _validate(schema, candidate):
    try:
        return schema.model_validate(candidate)
    except ValidationError:
        return None


def score_case(case: dict, mcqs: list[dict], elapsed_seconds: float, *,
               evidence: str | None = None, withheld: bool = False) -> dict:
    """Deterministic per-case scoring. Pure; used by the harness and its tests."""
    requested = int(case.get("num_questions") or 5)
    source = _norm(case.get("document") if evidence is None else evidence)
    keywords = [k for k in case.get("expected_keywords", []) if _norm(k)]
    expected_outcome = case.get("expected_outcome", "generate")

    generated = len(mcqs)
    count_ratio = generated / requested
    valid = [item for raw in mcqs if (item := _validate(MCQ, raw)) is not None]
    grounded = [item for raw in mcqs if (item := _validate(GroundedMCQ, raw)) is not None]
    validity = len(valid) / generated if generated else 0.0
    grounding_contract_rate = len(grounded) / generated if generated else 0.0
    answer_matches = sum(_norm(item.answer_text) in source for item in valid)
    quotes_found = sum(_norm(item.source_hint) in source for item in grounded)
    answer_text_match_rate = answer_matches / generated if generated else 0.0
    source_quote_rate = quotes_found / generated if generated else 0.0

    terms = [_norm(field) for item in valid for field in (item.question, item.answer_text)]
    matched_keywords = [term for term in keywords if any(_norm(term) in text for text in terms)]
    keyword_coverage = (len(matched_keywords) / len(keywords)) if keywords else 1.0

    questions = [question_key(item.question) for item in valid]
    duplicate_rate = (len(questions) - len(set(questions))) / generated if generated else 0.0

    passed = (
        MIN_COUNT_RATIO <= count_ratio <= 1.0
        and validity >= MIN_VALIDITY
        and grounding_contract_rate == 1.0
        and source_quote_rate == 1.0
        and duplicate_rate == 0.0
        and keyword_coverage >= (MIN_KEYWORD_COVERAGE if keywords else 0.0)
    )
    if expected_outcome == "withhold":
        passed = withheld and generated == 0
    status = ("complete" if generated >= requested else "partial" if generated else
              "withheld" if withheld else "failed")

    return {
        "id": case["id"],
        "requested": requested,
        "generated": generated,
        "count_ratio": round(count_ratio, 3),
        "validity": round(validity, 3),
        # Keep the historical machine-readable key as an alias for lexical matching.
        "grounded": round(answer_text_match_rate, 3),
        "answer_text_match_rate": round(answer_text_match_rate, 3),
        "source_quote_rate": round(source_quote_rate, 3),
        "grounding_contract_rate": round(grounding_contract_rate, 3),
        "keyword_coverage": round(keyword_coverage, 3),
        "matched_keywords": matched_keywords,
        "duplicate_rate": round(duplicate_rate, 3),
        "passed": bool(passed),
        "expected_outcome": expected_outcome,
        "status": status,
        "seconds": round(elapsed_seconds, 2),
    }


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def generate_live(case: dict) -> list[dict]:
    """Run the real pipeline (RAG with the configured provider)."""
    from core.services.mcq_pipeline import generate_mcqs_rag

    return generate_mcqs_rag(
        case["document"],
        num_questions=int(case["num_questions"]),
        difficulty=case["difficulty"],
        topic=case.get("topic"),
        owner="eval",
        title=str(case["id"]),
    )


def _sentences(document: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", document) if part.strip()]


def _stub_options(sentence: str, others: list[str], index: int) -> tuple[list[dict], str]:
    """Four distinct options where the correct one is a verbatim source span."""
    correct = sentence[:160]
    fillers = [
        "None of the statements above is supported by the source.",
        "The source explicitly rejects this statement.",
        "This claim is not mentioned in the source.",
        "The opposite of the statement is true.",
    ]
    seen = {_norm(correct)}
    texts = []
    for candidate in [s[:160] for s in others] + fillers:
        key = _norm(candidate)
        if key not in seen:
            texts.append(candidate)
            seen.add(key)
        if len(texts) == 3:
            break
    texts.insert(index % 4, correct)  # rotate the answer position A-D
    labels = ["A", "B", "C", "D"]
    options = [{"label": labels[i], "text": text} for i, text in enumerate(texts)]
    return options, correct


def generate_stub(case: dict) -> list[dict]:
    """Deterministic, offline stand-in. NOT a measure of generation quality."""
    if case.get("expected_outcome") == "withhold":
        return []
    sentences = _sentences(case["document"]) or [case["document"]]
    keywords = case.get("expected_keywords") or [case.get("topic") or "the source"]
    wanted = int(case["num_questions"])
    mcqs = []
    for index in range(wanted):
        sentence = sentences[index % len(sentences)]
        others = [s for i, s in enumerate(sentences) if i != index % len(sentences)]
        options, answer = _stub_options(sentence, others, index)
        keyword = keywords[index % len(keywords)]
        mcqs.append({
            "question": f"Which statement about {keyword} is supported by the source? ({index + 1})",
            "options": options,
            "answer_text": answer,
            "source_hint": answer,
            "explanation": "The marked statement quotes the supplied text.",
        })
    return mcqs


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def summarise(results: list[dict]) -> dict:
    def mean(key: str) -> float:
        return round(statistics.fmean(r[key] for r in results), 3) if results else 0.0

    passed = sum(1 for r in results if r["passed"])
    return {
        "cases": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
        "mean_count_ratio": mean("count_ratio"),
        "mean_validity": mean("validity"),
        "mean_grounded": mean("grounded"),
        "mean_answer_text_match_rate": mean("answer_text_match_rate"),
        "mean_source_quote_rate": mean("source_quote_rate"),
        "mean_grounding_contract_rate": mean("grounding_contract_rate"),
        "mean_keyword_coverage": mean("keyword_coverage"),
        "mean_duplicate_rate": mean("duplicate_rate"),
        "total_seconds": round(sum(r["seconds"] for r in results), 2),
    }


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# MCQ evaluation report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Mode: **{report['mode']}**",
        f"- Model: `{report['model']}`",
        f"- Dataset: `{report['dataset']}`",
        f"- Cases: {summary['cases']} | Passed: {summary['passed']} "
        f"({summary['pass_rate'] * 100:.0f}%)",
        f"- Mean validity: {summary['mean_validity']:.2f} | "
        f"Source quote match: {summary['mean_source_quote_rate']:.2f} | "
        f"Answer text match: {summary['mean_answer_text_match_rate']:.2f} | "
        f"Keyword coverage: {summary['mean_keyword_coverage']:.2f} | "
        f"Duplicates: {summary['mean_duplicate_rate']:.2f}",
        "",
        "| Case | Requested | Generated | Status | Valid | Source quote | Answer match | Keywords | Dups | Pass | Secs |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for r in report["cases"]:
        lines.append(
            f"| {r['id']} | {r['requested']} | {r['generated']} | {r['status']} | "
            f"{r['validity']:.2f} | {r['source_quote_rate']:.2f} | {r['answer_text_match_rate']:.2f} | "
            f"{r['keyword_coverage']:.2f} | {r['duplicate_rate']:.2f} | "
            f"{'yes' if r['passed'] else 'no'} | {r['seconds']:.2f} |"
        )
    lines.extend(["", "> Contract and lexical checks only. Factual correctness, relevance, "
                       "difficulty, distractors, and source coverage require human review."])
    if report["mode"] == "stub":
        lines += [
            "",
            "> Stub mode: questions come from a deterministic local generator, so",
            "> this report verifies the harness and scoring only. It is **not** a",
            "> measurement of model quality.",
        ]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from evals.isolation import isolated_evaluation
    with isolated_evaluation():
        return _main(argv)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate MCQ generation.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N cases.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use the deterministic stub generator; no LLM/credentials.")
    parser.add_argument("--live", action="store_true", help="Explicitly allow cost-bearing provider calls.")
    args = parser.parse_args(argv)
    if args.live == args.dry_run:
        print("Choose exactly one of --dry-run or --live.", file=sys.stderr)
        return 2

    cases = load_cases(args.dataset, args.limit)

    # The harness ingests documents, so point it at a throwaway database: an
    # evaluation run must never add rows to the real library.
    from core.services.generation_metrics import track_generation
    from fastapi_app.database import init_db

    init_db()

    model = "stub"
    if not args.dry_run:
        if not config.llm_key_present():
            print(f"No LLM credentials: {config.missing_key_message()}", file=sys.stderr)
            print("Set a provider key, or use --dry-run to verify the harness.", file=sys.stderr)
            return 2
        from core.llm import get_llm

        llm = get_llm()
        model = f"{getattr(llm, 'provider_id', config.LLM_PROVIDER)}/{llm.model}"

    generator = generate_stub if args.dry_run else generate_live
    results = []
    for case in cases:
        print(f"[eval] {case['id']} ({case['num_questions']} x {case['difficulty']}) ...",
              flush=True)
        started = time.perf_counter()
        with track_generation() as metrics:
            try:
                mcqs = generator(case)
                if not isinstance(mcqs, list):
                    raise TypeError("Generator output is not an MCQ list")
                error = None
            except Exception as exc:  # a failed case is data, not a crashed run
                mcqs = []
                error = type(exc).__name__
        counters = metrics.snapshot()
        elapsed = time.perf_counter() - started
        withheld = (case["expected_outcome"] == "withhold" and not mcqs
                    and ((args.dry_run and error is None) or (not counters["errors"] and not counters["unverified"]
                         and bool(counters["refusals"] or counters["rejected"]))))
        result = score_case(case, mcqs, elapsed, withheld=withheld)
        if error and not withheld:
            result["error"] = error
            result["passed"] = False
        results.append(result)
        print(f"[eval]   -> {result['generated']}/{result['requested']} "
              f"valid={result['validity']:.2f} passed={result['passed']}", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "stub" if args.dry_run else "live",
        "model": model,
        "dataset": args.dataset.resolve().relative_to(ROOT).as_posix()
        if args.dataset.resolve().is_relative_to(ROOT) else str(args.dataset),
        "summary": summarise(results),
        "cases": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")

    summary = report["summary"]
    print(f"[eval] {summary['passed']}/{summary['cases']} cases passed; "
          f"wrote {args.output} and {args.markdown}")
    return 0 if all(result['passed'] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
