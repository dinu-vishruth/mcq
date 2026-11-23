"""MCQ evaluation harness.

Runs the real generation pipeline over a small JSONL dataset and scores the
output with deterministic metrics, then writes machine-readable and Markdown
reports under ``evals/results/``. It is manual by design: a meaningful run calls
a real LLM (cost + credentials), so it is not part of CI.

Usage
-----
    # Real provider configured via the usual LLM_* variables:
    python evals/run_eval.py

    # Deterministic stub -- verifies the harness end-to-end without a key:
    python evals/run_eval.py --dry-run

    # Limit cases / choose output paths:
    python evals/run_eval.py --limit 3 --output evals/results/latest.json

Metrics per case (all in [0, 1] unless noted):
  count_ratio        generated / requested
  validity           fraction of questions that satisfy the MCQ contract
  grounded           fraction whose answer text appears in the source document
  keyword_coverage   fraction of expected_keywords found in question/options
  duplicate_rate     duplicate question texts / generated (lower is better)
  passed             count_ratio >= 0.8, validity == 1.0, no duplicates, and
                     keyword_coverage >= 0.6 when the case defines keywords
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Offline-friendly defaults, set before the app modules read config. An operator
# can override either variable to evaluate the production retrieval stack.
os.environ.setdefault("EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("VECTOR_STORE", "sqlite")

import config  # noqa: E402

DEFAULT_DATASET = ROOT / "evals" / "dataset.jsonl"
DEFAULT_OUTPUT = ROOT / "evals" / "results" / "latest.json"
DEFAULT_MARKDOWN = ROOT / "evals" / "results" / "latest.md"

MIN_COUNT_RATIO = 0.8
MIN_VALIDITY = 1.0
MIN_KEYWORD_COVERAGE = 0.6


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def load_cases(path: Path, limit: int | None = None) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON ({exc})")
            case.setdefault("num_questions", 5)
            case.setdefault("difficulty", "medium")
            case.setdefault("topic", None)
            case.setdefault("expected_keywords", [])
            case.setdefault("id", f"case-{line_number}")
            cases.append(case)
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise SystemExit(f"No cases found in {path}")
    return cases


def _is_valid(mcq: dict) -> bool:
    """Mirror of the structural MCQ contract (QualityAssuranceAgent)."""
    question = (mcq.get("question") or "").strip()
    options = mcq.get("options")
    if not question or not isinstance(options, list) or len(options) != 4:
        return False
    texts = []
    for option in options:
        if not isinstance(option, dict):
            return False
        text = (option.get("text") or "").strip()
        if not text:
            return False
        texts.append(text)
    if len(set(texts)) != 4:
        return False
    answer = (mcq.get("answer_text") or "").strip()
    return answer in texts


def score_case(case: dict, mcqs: list[dict], elapsed_seconds: float) -> dict:
    """Deterministic per-case scoring. Pure; used by the harness and its tests."""
    requested = int(case.get("num_questions") or 5)
    document = _norm(case.get("document") or "")
    keywords = [k for k in (case.get("expected_keywords") or []) if k]

    generated = len(mcqs)
    count_ratio = min(1.0, generated / requested) if requested else 1.0

    validity = (sum(1 for m in mcqs if _is_valid(m)) / generated) if generated else 0.0
    grounded_hits = 0
    for mcq in mcqs:
        answer = _norm(mcq.get("answer_text") or "")
        if answer and answer in document:
            grounded_hits += 1
    grounded = (grounded_hits / generated) if generated else 0.0

    haystack = _norm(" ".join(
        [str(m.get("question") or "") for m in mcqs]
        + [str(o.get("text") or "") for m in mcqs for o in (m.get("options") or [])]
        + [str(m.get("answer_text") or "") for m in mcqs]
    ))
    matched_keywords = [k for k in keywords if _norm(k) in haystack]
    keyword_coverage = (len(matched_keywords) / len(keywords)) if keywords else 1.0

    questions = [_norm(m.get("question") or "") for m in mcqs]
    duplicate_rate = 0.0
    if questions:
        duplicate_rate = 1.0 - (len(set(questions)) / len(questions))

    passed = (
        count_ratio >= MIN_COUNT_RATIO
        and validity >= MIN_VALIDITY
        and duplicate_rate == 0.0
        and keyword_coverage >= (MIN_KEYWORD_COVERAGE if keywords else 0.0)
    )

    return {
        "id": case["id"],
        "requested": requested,
        "generated": generated,
        "count_ratio": round(count_ratio, 3),
        "validity": round(validity, 3),
        "grounded": round(grounded, 3),
        "keyword_coverage": round(keyword_coverage, 3),
        "matched_keywords": matched_keywords,
        "duplicate_rate": round(duplicate_rate, 3),
        "passed": bool(passed),
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
        owner="eval",
        title=str(case["id"]),
    )


def _sentences(document: str) -> list[str]:
    parts = [p.strip() for p in document.replace("\n", " ").split(". ")]
    return [p if p.endswith(".") else p + "." for p in parts if p.strip()]


def _stub_options(sentence: str, others: list[str], index: int) -> tuple[list[dict], str]:
    """Four distinct options where the correct one is a verbatim source span."""
    correct = sentence[:160]
    candidates = [s[:160] for s in others if s[:160] != correct]
    distractors = candidates[:3]
    fillers = [
        "None of the statements above is supported by the source.",
        "The source explicitly rejects this statement.",
        "This claim is not mentioned in the source.",
        "The opposite of the statement is true.",
    ]
    while len(distractors) < 3:
        filler = fillers[len(distractors)]
        if filler != correct and filler not in distractors:
            distractors.append(filler)
    texts = list(distractors[:3])
    texts.insert(index % 4, correct)  # rotate the answer position A-D
    labels = ["A", "B", "C", "D"]
    options = [{"label": labels[i], "text": text} for i, text in enumerate(texts)]
    return options, correct


def generate_stub(case: dict) -> list[dict]:
    """Deterministic, offline stand-in. NOT a measure of generation quality."""
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
        f"Grounded: {summary['mean_grounded']:.2f} | "
        f"Keyword coverage: {summary['mean_keyword_coverage']:.2f} | "
        f"Duplicates: {summary['mean_duplicate_rate']:.2f}",
        "",
        "| Case | Requested | Generated | Valid | Grounded | Keywords | Dups | Pass | Secs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for r in report["cases"]:
        lines.append(
            f"| {r['id']} | {r['requested']} | {r['generated']} | "
            f"{r['validity']:.2f} | {r['grounded']:.2f} | "
            f"{r['keyword_coverage']:.2f} | {r['duplicate_rate']:.2f} | "
            f"{'yes' if r['passed'] else 'no'} | {r['seconds']:.2f} |"
        )
    if report["mode"] == "stub":
        lines += [
            "",
            "> Stub mode: questions come from a deterministic local generator, so",
            "> this report verifies the harness and scoring only. It is **not** a",
            "> measurement of model quality. Run without `--dry-run` for that.",
        ]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate MCQ generation.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N cases.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use the deterministic stub generator; no LLM/credentials.")
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset, args.limit)

    # The harness ingests documents, so point it at a throwaway database: an
    # evaluation run must never add rows to the real library.
    config.DB_PATH = os.path.join(tempfile.mkdtemp(prefix="mcq_eval_"), "eval.db")
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
        try:
            mcqs = generator(case)
            error = None
        except Exception as exc:  # a failed case is data, not a crashed run
            mcqs = []
            error = str(exc)
        elapsed = time.perf_counter() - started
        result = score_case(case, mcqs, elapsed)
        if error:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
