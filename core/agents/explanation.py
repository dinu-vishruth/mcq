"""Display already verified source-linked explanations; never invent replacements."""
from core.agents.base import Agent
from core.services.ai_safety import candidate_allowed
from core.services.source_evidence import authentic


class ExplanationAgent(Agent):
    name = "explanation"

    def run(self, details: list[dict], *, document_id: int | None = None) -> list[str]:
        # document_id remains accepted for API compatibility. Historical quizzes
        # lacking an acceptance receipt receive explicit availability feedback.
        explanations = []
        for detail in details:
            question = detail.get("source_question")
            if isinstance(question, dict) and authentic(question) and candidate_allowed(question):
                explanations.append(f"{question['explanation']} Source: {question['source_hint']}")
            else:
                state = "Correct" if detail.get("is_correct") else "Not answered" if not detail.get("selected") else "Incorrect"
                explanations.append(f"{state}. The recorded answer is '{detail.get('correct', '')}'. "
                                    "A source-grounded explanation is unavailable for this question.")
        return explanations
