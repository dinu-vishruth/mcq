"""
Prompts for the extensible content generators (Phase 7). Each returns a
retrieval-grounded prompt that instructs the model to use ONLY the supplied
context. Adding a new content type = add a prompt here + register a generator.
"""

SYSTEM = "You are an expert study-content creator that only outputs valid JSON objects."


def flashcards_prompt(context: str, num_items: int, topic: str | None) -> str:
    focus = f" focused on {topic}" if topic else ""
    return f"""Using ONLY the CONTEXT, create {num_items} study flashcards{focus}.
Respond with JSON exactly like:
{{"flashcards": [{{"front": "term or question", "back": "definition or answer"}}]}}
Do not invent facts beyond the CONTEXT.

CONTEXT:
\"\"\"{context}\"\"\"
"""


def summary_prompt(context: str, topic: str | None) -> str:
    focus = f" about {topic}" if topic else ""
    return f"""Using ONLY the CONTEXT, write a structured summary{focus}.
Respond with JSON exactly like:
{{"summary": {{"overview": "2-3 sentence overview", "key_points": ["point 1", "point 2", "point 3"]}}}}

CONTEXT:
\"\"\"{context}\"\"\"
"""


def interview_prompt(context: str, num_items: int, topic: str | None) -> str:
    focus = f" on {topic}" if topic else ""
    return f"""Using ONLY the CONTEXT, create {num_items} interview questions{focus} with model answers.
Respond with JSON exactly like:
{{"questions": [{{"question": "...", "answer": "concise strong answer"}}]}}

CONTEXT:
\"\"\"{context}\"\"\"
"""


def coding_prompt(context: str, num_items: int, topic: str | None) -> str:
    focus = f" related to {topic}" if topic else ""
    return f"""Using ONLY the CONTEXT, create {num_items} coding practice problems{focus}.
Respond with JSON exactly like:
{{"problems": [{{"title": "...", "prompt": "problem statement", "hint": "one hint"}}]}}

CONTEXT:
\"\"\"{context}\"\"\"
"""


def explain_topic_prompt(context: str, topic: str | None) -> str:
    subject = topic or "the main concept in the context"
    return f"""Using ONLY the CONTEXT, explain {subject} clearly for a student.
Respond with JSON exactly like:
{{"explanation": {{"topic": "...", "body": "clear multi-sentence explanation", "example": "a concrete example"}}}}

CONTEXT:
\"\"\"{context}\"\"\"
"""
