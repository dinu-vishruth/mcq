"""
Difficulty validation prompts. The DifficultyAgent asks the model to grade each
question's true cognitive difficulty against the requested level, so a set
labelled "hard" doesn't silently contain recall questions.
"""

from core.services.ai_safety import POLICY

SYSTEM = POLICY + "\n" + "You are an assessment-difficulty grader that only outputs valid JSON objects."


def grade_prompt(questions: list[dict], requested: str) -> str:
    """questions: [{"index": int, "question": str}]. Grade each vs requested level."""
    lines = "\n".join(
        f'{q["index"]}. {q["question"]}\nOptions: {q.get("options", [])}\n'
        f'Answer: {q.get("answer_text", "")}' for q in questions
    )
    return f"""Grade the cognitive difficulty of each question below as easy, medium, or hard,
using Bloom's Taxonomy (easy=Remember/Understand, medium=Apply/Analyze,
hard=Analyze/Evaluate/Create). The requested level for this set is: {requested.upper()}.
Grade the actual reasoning, regardless of the requested label or the subject's
technical vocabulary. Recognizing a definition copied from the source is EASY,
even for database transactions or advanced science. A negated recall question
is still EASY. MEDIUM requires applying a rule to a new scenario or analyzing
a relationship. HARD requires evaluating alternatives or multiple linked
reasoning steps. If the evidence cannot support the requested difficulty,
mark matches_requested=false. Never promote recall merely to fill a hard set.

Respond with a JSON object exactly like:
{{"grades": [{{"index": 0, "level": "medium", "matches_requested": true}}, ...]}}

QUESTIONS:
{lines}
"""
