"""Request-local learning preferences: never shared across concurrent users."""
from contextlib import contextmanager
from contextvars import ContextVar

from core.repositories import prefs_repo

_instructions = ContextVar("learning_instructions", default="")

GOALS = {
    "college": "college exams", "placement": "job placement tests",
    "interview": "job interviews", "competitive": "competitive exams",
    "certification": "certification exams", "personal": "personal learning",
}
STYLES = {
    "practice": "Emphasize applied practice and learning by doing.",
    "revision": "Use concise explanations that reinforce recall and revision.",
    "concepts": "Explain underlying concepts and relationships clearly.",
    "mixed": "Balance concept understanding, recall and applied practice.",
}


@contextmanager
def learning_profile(user_id):
    prefs = prefs_repo.load(user_id) or {}
    parts = []
    if prefs.get("goal") in GOALS:
        parts.append(f"The learner is preparing for {GOALS[prefs['goal']]}.")
    if prefs.get("style") in STYLES:
        parts.append(STYLES[prefs["style"]])
    if prefs.get("daily_minutes"):
        parts.append(f"Their daily study budget is {int(prefs['daily_minutes'])} minutes. "
                     "Keep guidance manageable within that budget.")
    if parts:
        parts.append("Respect the requested question count, difficulty and source grounding.")
    token = _instructions.set("\n".join(parts))
    try:
        yield
    finally:
        _instructions.reset(token)


def system_prompt(base: str) -> str:
    instructions = _instructions.get()
    return base + ("\nLearner preferences:\n" + instructions if instructions else "")
