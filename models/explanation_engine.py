# models/explanation_engine.py
"""
Answer-explanation entry point.

Keeps its historical public API — `explain_answers(details) -> list[str]`,
one entry per question, positionally aligned — consumed by result.html as
`explanations[loop.index0]`. Delegates to core.agents.explanation.ExplanationAgent,
which grounds explanations in retrieved context when a document_id is supplied
and otherwise reproduces the original ungrounded behaviour (including all the
"no key" / "no wrong answers" / API-error fallbacks).
"""
from core.agents.explanation import ExplanationAgent


def explain_answers(details, document_id=None):
    """details: list of {question, selected, correct, is_correct}.
    Returns a list of explanation strings, one per detail, in order."""
    return ExplanationAgent().run(details, document_id=document_id)
