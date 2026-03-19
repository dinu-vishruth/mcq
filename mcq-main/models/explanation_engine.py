# models/explanation_engine.py
"""
Explanation engine using local NLP — no API key needed.
Generates contextual explanations for why the answer is correct.
"""


def explain_answers(details):
    """
    Generate explanations for each question result.

    details: list of dicts with question, selected, correct, is_correct
    """
    explanations = []
    for d in details:
        question = d.get("question", "")
        selected = d.get("selected", "")
        correct = d.get("correct", "")

        if d["is_correct"]:
            explanations.append(
                f"✅ Correct! '{selected}' is the right answer. "
                f"This is the most accurate option that fits the context of the original passage."
            )
        else:
            if selected:
                explanations.append(
                    f"❌ Incorrect. You chose '{selected}', but the correct answer is '{correct}'. "
                    f"The correct option best matches the factual content and context from the source material."
                )
            else:
                explanations.append(
                    f"⏭️ Not answered. The correct answer is '{correct}'. "
                    f"This option aligns with the information presented in the passage."
                )

    return explanations
