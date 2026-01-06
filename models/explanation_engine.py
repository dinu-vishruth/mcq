# models/explanation_engine.py
def explain_answers(details):
    # details: list of dicts with question, selected, correct, is_correct
    explanations = []
    for d in details:
        if d["is_correct"]:
            explanations.append(f"Correct. '{d['selected']}' is the correct answer because it matches the key concept in the sentence.")
        else:
            explanations.append(f"Wrong. Correct answer: '{d['correct']}'. The correct option best matches the factual content of the sentence.")
    return explanations
