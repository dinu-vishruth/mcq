"""Blind review export and strict human scoring."""
import json

import pytest

from evals.review import QUESTION_RUBRIC, build_review_packets, main, score_review_packet


def test_quality_gate_rejects_stub_incomplete_small_and_incorrect_reviews():
    import copy

    from evals.review import release_quality
    dimensions = {name: {'expected': 20, 'rated': 20, 'counts': {'0': 0, '1': 0, '2': 20}, 'mean': 2.0}
                  for name in (*QUESTION_RUBRIC, 'source_coverage')}
    report = {'mode': 'live', 'strategies': {'primary': {'questions': 20, 'quizzes': 3, 'dimensions': dimensions}}}
    assert release_quality(report)['passed']
    for mutate in ('stub', 'incomplete', 'small', 'incorrect'):
        bad = copy.deepcopy(report)
        strategy = bad['strategies']['primary']
        if mutate == 'stub':
            bad['mode'] = 'stub'
        elif mutate == 'small':
            strategy['questions'] = 19
        elif mutate == 'incomplete':
            strategy['dimensions']['unambiguous']['rated'] = 19
        else:
            strategy['dimensions']['factual_correctness']['counts'] = {'0': 0, '1': 1, '2': 19}
        assert not release_quality(bad)['passed']


def _runs():
    question = {
        "question": "Which rule applies to reservations?",
        "options": [{"label": "A", "text": "Every visitor must reserve a place.", "model": "private-model"}],
        "answer_text": "Every visitor must reserve a place.",
        "source_hint": "Every visitor must reserve a place.",
        "explanation": "The museum requires reservations for all visitors.",
        "source_evidence": {"receipt": "internal-acceptance-receipt"},
        "verification_status": "verified",
    }
    return [{"case_id": "private-case-id", "strategy": strategy, "repeat": 1, "difficulty": "easy",
             "document": "Every visitor must reserve a place. Closed on Sunday.",
             "context": "Every visitor must reserve a place.", "questions": [question]}
            for strategy in ("primary", "fallback")]


def test_packet_hides_strategy_key_and_internal_verdicts():
    packet, key = build_review_packets(_runs(), seed=42)
    raw = json.dumps(packet)
    for hidden in ("primary", "fallback", "private-case-id", "internal-acceptance-receipt", "verification_status", "private-model"):
        assert hidden not in raw
    assert len(packet["items"]) == 2
    assert {row["strategy"] for row in key["runs"].values()} == {"primary", "fallback"}
    assert packet["items"][0]["supplied_evidence"] == "Every visitor must reserve a place."
    assert packet["items"][0]["questions"][0]["ratings"] == dict.fromkeys(QUESTION_RUBRIC)


def test_score_distinguishes_missing_rating_from_zero(tmp_path):
    packet, key = build_review_packets(_runs(), seed=42)
    packet["mode"] = "stub"
    for item in packet["items"]:
        strategy = key["runs"][item["review_id"]]["strategy"]
        item["coverage_rating"] = 1
        item["questions"][0]["ratings"]["factual_correctness"] = 0 if strategy == "primary" else 2

    packet_path, key_path, output = [tmp_path / f"{name}.json" for name in ("packet", "key", "output")]
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    key_path.write_text(json.dumps(key), encoding="utf-8")
    assert main([str(packet_path), str(key_path), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "stub"
    assert report["strategies"]["primary"]["dimensions"]["factual_correctness"] == {
        "expected": 1, "counts": {"0": 1, "1": 0, "2": 0}, "rated": 1, "mean": 0,
    }
    assert report["strategies"]["fallback"]["dimensions"]["factual_correctness"]["mean"] == 2
    assert report["strategies"]["primary"]["dimensions"]["difficulty_fit"]["mean"] is None


@pytest.mark.parametrize("value", [True, "2", 3, -1, 1.0])
def test_rejects_invalid_ratings(value):
    packet, key = build_review_packets(_runs(), seed=42)
    packet["items"][0]["coverage_rating"] = value
    with pytest.raises(ValueError, match="ratings must be"):
        score_review_packet(packet, key)


def test_rejects_an_edited_or_incomplete_packet():
    packet, key = build_review_packets(_runs(), seed=42)
    packet["items"][0]["questions"][0]["question"] = "A different question"
    with pytest.raises(ValueError, match="content changed"):
        score_review_packet(packet, key)
    packet["items"].pop(0)
    with pytest.raises(ValueError, match="missing one or more quizzes"):
        score_review_packet(packet, key)
