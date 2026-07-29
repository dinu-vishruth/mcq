"""
Difficulty validation prompts. The DifficultyAgent asks the model to grade each
question's true cognitive difficulty against the requested level, so a set
labelled "hard" doesn't silently contain recall questions.
"""

SYSTEM = "You are an assessment-difficulty grader that only outputs valid JSON objects."


def grade_prompt(questions: list[dict], requested: str) -> str:
    """questions: [{"index": int, "question": str}]. Grade each vs requested level."""
    lines = "\n".join(f'{q["index"]}. {q["question"]}' for q in questions)
    return f"""Grade the cognitive difficulty of each question below as easy, medium, or hard,
using Bloom's Taxonomy (easy=Remember/Understand, medium=Apply/Analyze,
hard=Analyze/Evaluate/Create). The requested level for this set is: {requested.upper()}.

Respond with a JSON object exactly like:
{{"grades": [{{"index": 0, "level": "medium", "matches_requested": true}}, ...]}}

QUESTIONS:
{lines}
"""
