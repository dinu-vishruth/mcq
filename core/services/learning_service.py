"""
Learning service: post-submission analysis and dashboard data.

Kept entirely separate from scoring. app.submit() calls analyse_submission()
in a try/except so a failure here never affects the student's score or result
page. Reads for the dashboard are simple repo passthroughs.
"""
from __future__ import annotations

import config
from core.agents.evaluation import EvaluationAgent
from core.repositories import learning_repo


def analyse_submission(details, *, user_id=None, session_key="", difficulty="medium") -> dict:
    """Run evaluation + persist weak topics. Returns the evaluation result.

    Safe no-op when there's no user or the LLM is unavailable -- topic labelling
    degrades to 'General' and nothing breaks.
    """
    if not user_id:
        return {"topic_stats": {}, "events": [], "recommendations": []}
    return EvaluationAgent().run(details, user_id=user_id, session_key=session_key,
                                 difficulty=difficulty)


def get_weak_topics(user_id, limit=5) -> list[dict]:
    try:
        return learning_repo.top_weak_topics(user_id, limit=limit)
    except Exception:
        return []


def get_recommendations(user_id, limit=5) -> list[str]:
    weak = get_weak_topics(user_id, limit=limit)
    if not weak or not config.LLM_API_KEY:
        return []
    try:
        return EvaluationAgent().recommend(weak)
    except Exception:
        return []
