# models/explanation_engine.py
"""Return saved verified explanations, positionally aligned with submitted answers."""
from core.agents.explanation import ExplanationAgent


def explain_answers(details, document_id=None):
    """details: list of {question, selected, correct, is_correct}.
    Returns a list of explanation strings, one per detail, in order."""
    return ExplanationAgent().run(details, document_id=document_id)
