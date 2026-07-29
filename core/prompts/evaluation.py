"""
Evaluation prompts. The EvaluationAgent labels each answered question with a
concise topic/concept so weak areas can be aggregated, then recommends next
study topics. Topic labelling is a cheap classification the LLM does well.
"""

SYSTEM = "You are a learning analytics assistant that only outputs valid JSON objects."


def topic_label_prompt(questions: list[dict]) -> str:
    """questions: [{"index": int, "question": str}]. Return a short topic each."""
    lines = "\n".join(f'{q["index"]}. {q["question"]}' for q in questions)
    return f"""For each question, assign a short (1-4 word) topic/concept label naming what it tests.

Respond with a JSON object exactly like:
{{"topics": [{{"index": 0, "topic": "Photosynthesis"}}, ...]}}

QUESTIONS:
{lines}
"""


def recommendation_prompt(weak_topics: list[dict]) -> str:
    listing = "\n".join(f'- {w["topic"]} ({w["wrong"]}/{w["total"]} wrong)' for w in weak_topics)
    return f"""A student is weakest in the following topics:
{listing}

Recommend what to study next. Respond with a JSON object exactly like:
{{"recommendations": ["short actionable study tip", ...]}}
Keep each tip to one sentence, at most 4 tips, ordered by priority.
"""
