# utils/difficulty_classifier.py
def classify_difficulty(text):
    # simple heuristic by average sentence length
    sents = [s.strip() for s in text.split(".") if s.strip()]
    if not sents:
        return "medium"
    avg = sum(len(s.split()) for s in sents) / len(sents)
    if avg < 8:
        return "easy"
    if avg < 18:
        return "medium"
    return "hard"
