"""
Context-validation prompt. Before the generator runs, the ContextValidationAgent
asks the model whether the retrieved CONTEXT actually contains enough substantive
information to write grounded MCQs at the requested difficulty. This is the
guardrail that turns "invent something plausible" into an explicit
INSUFFICIENT_CONTEXT signal.
"""

SYSTEM = "You are a strict context-sufficiency checker that only outputs valid JSON objects."


def validate_prompt(context: str, num_questions: int, difficulty: str) -> str:
    return f"""Decide whether the CONTEXT below contains enough substantive, factual information to write {num_questions} {difficulty.upper()} multiple-choice questions that are ANSWERABLE FROM THE CONTEXT ALONE.

Sufficient means: the context states concrete facts, definitions, relationships, or processes — not just headings, a table of contents, boilerplate, or a few disconnected sentences.

Judge ONLY what is present in the CONTEXT. Do not consider outside knowledge.

Respond with a JSON object exactly like:
{{"sufficient": true, "confidence": 0.0-1.0, "reason": "one short sentence"}}

CONTEXT:
\"\"\"
{context}
\"\"\"
"""
