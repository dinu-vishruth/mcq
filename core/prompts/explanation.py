"""
Explanation prompts. `legacy_prompt` reproduces the current wording used by
models/explanation_engine.py so the AI-explanation output is unchanged when the
legacy path is active. `grounded_prompt` (Phase 5) adds retrieved context so
explanations cite the source material instead of relying on model memory.
"""

SYSTEM = "You are a helpful teacher that only outputs valid JSON objects."

_TAIL = ('You MUST respond with a JSON object containing an "explanations" key, which holds a '
         'flat array of strings, in the exact same order as the questions above. '
         'Format exactly like this: {"explanations": ["Explanation for Q1", "Explanation for Q2"]}')


def legacy_prompt(wrong_items: list[dict]) -> str:
    """wrong_items: list of {"question","selected","correct"} for wrong answers."""
    prompt = ("You are a helpful teacher. Briefly explain WHY the student's selected wrong answer "
              "is incorrect, and WHY the correct answer is right. Keep it to exactly 1 or 2 clear "
              "sentences per question.\n\n")
    for d in wrong_items:
        prompt += (f"Question: {d['question']}\nStudent Answer: {d['selected']}\n"
                   f"Correct Answer: {d['correct']}\n\n")
    prompt += _TAIL
    return prompt


def grounded_prompt(wrong_items: list[dict], context: str) -> str:
    """Same shape, but explanations must draw on the retrieved CONTEXT."""
    prompt = ("You are a helpful teacher. Using the CONTEXT for factual grounding, briefly explain "
              "WHY each student's selected answer is incorrect and WHY the correct answer is right. "
              "Exactly 1 or 2 clear sentences per question.\n\n")
    prompt += f'CONTEXT:\n"""\n{context}\n"""\n\n'
    for d in wrong_items:
        prompt += (f"Question: {d['question']}\nStudent Answer: {d['selected']}\n"
                   f"Correct Answer: {d['correct']}\n\n")
    prompt += _TAIL
    return prompt
