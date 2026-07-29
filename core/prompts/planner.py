"""
Planner prompts. The PlannerAgent maps a free-form user request to a structured
intent + parameters. Keeping the taxonomy here (not in code) lets Phase 7 add
new intents (flashcards, summary, etc.) by editing this list only.
"""

SUPPORTED_INTENTS = [
    "generate_mcqs",     # default
    "generate_flashcards",
    "summarize",
    "explain_topic",
    "compare_chapters",
    "revision_notes",
    "interview_questions",
    "coding_questions",
]

SYSTEM = "You are a planning assistant that only outputs valid JSON objects."


def plan_prompt(user_request: str) -> str:
    intents = ", ".join(SUPPORTED_INTENTS)
    return f"""Classify the user's request into one supported intent and extract parameters.
Supported intents: {intents}.

Respond with a JSON object exactly like:
{{"intent": "generate_mcqs", "num_items": 10, "difficulty": "medium", "topic": "optional focus topic or null"}}

Rules:
- num_items: integer if the user names a count, else null.
- difficulty: one of easy|medium|hard if stated, else null.
- topic: a short focus phrase if the user names one, else null.

USER REQUEST:
\"\"\"{user_request}\"\"\"
"""
