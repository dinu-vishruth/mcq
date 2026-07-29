"""
Evaluation Agent: analyse student performance, detect weak concepts, recommend
next study topics.

Given a graded submission (the same `details` list the submit flow builds), it
labels each question with a topic, aggregates wrong/total per topic, persists to
learning_history + weak_topics, and can produce study recommendations. All
persistence is best-effort so it can never break the quiz submit path.
"""
from __future__ import annotations

from collections import defaultdict

from core.agents.base import Agent
from core.llm import get_llm, LLMError
from core.prompts import evaluation as prompts
from core.repositories import learning_repo


class EvaluationAgent(Agent):
    name = "evaluation"

    def run(self, details: list[dict], *, user_id=None, session_key="", difficulty="medium") -> dict:
        """Return {"topic_stats", "events", "recommendations"}."""
        if not details:
            return {"topic_stats": {}, "events": [], "recommendations": []}

        topics = self._label_topics(details)

        events, topic_stats = [], defaultdict(lambda: {"wrong": 0, "total": 0})
        for i, d in enumerate(details):
            topic = topics.get(i, "General")
            is_correct = bool(d.get("is_correct"))
            events.append({"topic": topic, "question": d.get("question", ""), "is_correct": is_correct})
            topic_stats[topic]["total"] += 1
            if not is_correct:
                topic_stats[topic]["wrong"] += 1

        topic_stats = dict(topic_stats)

        # Best-effort persistence.
        try:
            learning_repo.record_events(user_id, session_key, difficulty, events)
            learning_repo.upsert_weak_topics(user_id, topic_stats)
        except Exception as e:
            self._log(f"persistence skipped: {e}")

        return {"topic_stats": topic_stats, "events": events, "recommendations": []}

    def recommend(self, weak_topics: list[dict]) -> list[str]:
        if not weak_topics:
            return []
        try:
            data = get_llm().complete_json([
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": prompts.recommendation_prompt(weak_topics)},
            ])
            recs = data.get("recommendations", []) if isinstance(data, dict) else []
            return [str(r) for r in recs][:4]
        except (LLMError, Exception) as e:
            self._log(f"recommendation skipped: {e}")
            return []

    def _label_topics(self, details) -> dict:
        """Map question index -> short topic label via one batched LLM call."""
        if not details:
            return {}
        payload = [{"index": i, "question": d.get("question", "")} for i, d in enumerate(details)]
        try:
            data = get_llm().complete_json([
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": prompts.topic_label_prompt(payload)},
            ])
            out = {}
            for t in (data.get("topics", []) if isinstance(data, dict) else []):
                try:
                    out[int(t["index"])] = (t.get("topic") or "General").strip() or "General"
                except (TypeError, ValueError, KeyError):
                    continue
            return out
        except (LLMError, Exception) as e:
            self._log(f"topic labelling skipped: {e}")
            return {}
