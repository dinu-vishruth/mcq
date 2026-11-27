"""Blind human MCQ review. Score a filled packet with `python -m evals.review`."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

QUESTION_RUBRIC = {
    "factual_correctness": "The marked answer is factually correct under the supplied facts.",
    "evidence_support": "The supplied evidence supports the answer and the cited quote is relevant.",
    "topic_relevance": "The question tests the requested topic or a central idea in the source.",
    "difficulty_fit": "The reasoning needed matches the requested difficulty.",
    "distractor_quality": "The distractors are plausible mistakes and clearly incorrect.",
    "unambiguous": "The stem is clear and exactly one option is defensible.",
    "explanation_quality": "The explanation accurately explains why the marked answer is correct.",
}
BATCH_DIMENSION = "source_coverage"
BATCH_RUBRIC = "The set covers the important source ideas for the topic and count without repeating them."
RATINGS = {"0": "Fails", "1": "Partly meets", "2": "Meets", "null": "Unrated"}
VERSION = 1


def _review_question(question: dict) -> dict:
    # Whitelist fields; receipts, internal scores and generation metadata would
    # compromise blinding and are irrelevant to human review.
    review = {field: question.get(field) for field in ("question", "answer_text", "source_hint", "explanation")}
    options = question.get("options")
    review["options"] = [{"label": option.get("label"), "text": option.get("text")} if isinstance(option, dict) else None
                         for option in options] if isinstance(options, list) else []
    return review


def _digest(item: dict) -> str:
    body = {k: v for k, v in item.items() if k not in ("coverage_rating", "questions")}
    body["questions"] = [{k: v for k, v in question.items() if k != "ratings"} for question in item["questions"]]
    packed = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def build_review_packets(runs: list[dict], *, seed: int | None = None) -> tuple[dict, dict]:
    """Return a shuffled packet and a separate key. Pass only the packet to reviewers."""
    rng = random.SystemRandom() if seed is None else random.Random(seed)
    items, key = [], {}
    for run in runs:
        if not run["questions"]:
            continue  # Failures and withholds stay in the automated result report.
        review_id = f"{rng.getrandbits(128):032x}"
        item = {
            "review_id": review_id,
            "source": run["document"],
            "supplied_evidence": run.get("context", run["document"]),
            "topic": run.get("topic"),
            "difficulty": run["difficulty"],
            "requested_count": run.get("requested_count", len(run["questions"])),
            "coverage_rating": None,
            "questions": [
                {**_review_question(question), "ratings": dict.fromkeys(QUESTION_RUBRIC)}
                for question in run["questions"]
            ],
        }
        items.append(item)
        key[review_id] = {field: run[field] for field in ("case_id", "strategy", "repeat")}
        key[review_id]["digest"] = _digest(item)
    rng.shuffle(items)
    return (
        {"schema_version": VERSION, "ratings": RATINGS, "question_rubric": QUESTION_RUBRIC,
         "batch_rubric": BATCH_RUBRIC, "items": items},
        {"schema_version": VERSION, "runs": key},
    )


def _record(stats: dict, dimension: str, value) -> None:
    row = stats.setdefault(dimension, {"expected": 0, "counts": {str(i): 0 for i in range(3)}})
    row["expected"] += 1
    if value is None:
        return
    if type(value) is not int or value not in (0, 1, 2):
        raise ValueError(f"{dimension} ratings must be 0, 1, 2, or null")
    row["counts"][str(value)] += 1


def score_review_packet(packet: dict, key: dict) -> dict:
    """Aggregate submitted ratings by strategy without imputing missing judgments."""
    if not isinstance(packet, dict) or not isinstance(key, dict):
        raise ValueError("Review packet and key must be JSON objects")
    if packet.get("schema_version") != VERSION or key.get("schema_version") != VERSION:
        raise ValueError("Unsupported review schema version")
    if not isinstance(packet.get("items"), list) or not isinstance(key.get("runs"), dict):
        raise ValueError("Malformed review packet or key")

    grouped, seen = {}, set()
    for item in packet["items"]:
        review_id = item.get("review_id") if isinstance(item, dict) else None
        if not isinstance(review_id, str) or review_id in seen or review_id not in key["runs"]:
            raise ValueError("Unknown or duplicate review ID")
        seen.add(review_id)
        row = key["runs"][review_id]
        if not isinstance(row, dict) or not isinstance(row.get("strategy"), str):
            raise ValueError("Malformed strategy key")
        try:
            matches = _digest(item) == row["digest"]
        except (KeyError, TypeError, AttributeError):
            raise ValueError("Malformed review item") from None
        if not matches:
            raise ValueError("Review content changed. Edit only ratings and the batch coverage rating.")
        stats = grouped.setdefault(row["strategy"], {"quizzes": 0, "questions": 0, "dimensions": {}})
        stats["quizzes"] += 1
        stats["questions"] += len(item["questions"])
        _record(stats["dimensions"], BATCH_DIMENSION, item.get("coverage_rating"))
        for question in item["questions"]:
            ratings = question.get("ratings")
            if not isinstance(ratings, dict) or set(ratings) != set(QUESTION_RUBRIC):
                raise ValueError("Each question must include all rubric fields. Leave unrated fields as null.")
            for dimension, value in ratings.items():
                _record(stats["dimensions"], dimension, value)
    if seen != key["runs"].keys():
        raise ValueError("Review packet is missing one or more quizzes. Leave unrated quizzes in the packet.")

    for strategy in grouped.values():
        for scores in strategy["dimensions"].values():
            counts = scores["counts"]
            total = sum(counts.values())
            scores["rated"] = total
            scores["mean"] = round(sum(int(i) * n for i, n in counts.items()) / total, 3) if total else None
    return {"schema_version": VERSION, "mode": packet.get("mode", "unspecified"),
            "note": "Human ratings of reviewed output only. Stub packets do not measure model quality.",
            "strategies": grouped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score human ratings from a blind MCQ review packet.")
    parser.add_argument("packet", type=Path, help="Fill question ratings and coverage_rating in a copy of this JSON file.")
    parser.add_argument("key", type=Path, help="Separate key generated with the packet; hide it from reviewers.")
    parser.add_argument("--output", type=Path, help="Write report JSON instead of printing it.")
    parser.add_argument("--require-quality", action="store_true", help="Fail unless a complete live review meets the release criteria.")
    args = parser.parse_args(argv)
    try:
        report = score_review_packet(json.loads(args.packet.read_text(encoding="utf-8")),
                                     json.loads(args.key.read_text(encoding="utf-8")))
        report['release_quality'] = release_quality(report)
        output = json.dumps(report, indent=2) + "\n"
        if args.output:
            if args.output.resolve() in (args.packet.resolve(), args.key.resolve()):
                raise ValueError("Report output must differ from the packet and key paths")
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Review could not be scored: {exc}\n")
    return 1 if args.require_quality and not report['release_quality']['passed'] else 0


def release_quality(report: dict) -> dict:
    """Conservative review criteria; automated or incomplete reviews cannot pass."""
    failures = []
    if report.get('mode') != 'live':
        failures.append('A live provider run and independent human review are required.')
    strategies = report.get('strategies', {})
    if not strategies:
        failures.append('No reviewed strategies.')
    for name, strategy in strategies.items():
        if strategy['questions'] < 20 or strategy['quizzes'] < 3:
            failures.append(f'{name}: review at least 20 questions across 3 quizzes.')
        for dimension in (*QUESTION_RUBRIC, BATCH_DIMENSION):
            score = strategy['dimensions'].get(dimension)
            if not score or score['rated'] != score['expected'] or not score['rated']:
                failures.append(f'{name}: {dimension} is incomplete.')
                continue
            critical = dimension in ('factual_correctness', 'evidence_support', 'unambiguous')
            if (critical and score['counts']['2'] != score['rated']) or (not critical and score['mean'] < 1.6):
                failures.append(f'{name}: {dimension} is below the release threshold.')
    return {'passed': not failures, 'failures': failures,
            'scope': 'Human review only; also require automated contract, duplicate and source quote checks.'}


if __name__ == "__main__":
    raise SystemExit(main())
