"""
Explanation prompts. `legacy_prompt` reproduces the current wording used by
models/explanation_engine.py so the AI-explanation output is unchanged when the
legacy path is active. `grounded_prompt` (Phase 5) adds retrieved context so
explanations cite the source material instead of relying on model memory.
"""

SYSTEM = "You are a helpful exam coach that only outputs valid JSON objects."

_TAIL = ('You MUST respond with a JSON object containing an "explanations" key, which holds a '
         'flat array of strings, in the exact same order as the questions above. '
         'Format exactly like this: {"explanations": ["Explanation for Q1", "Explanation for Q2"]}')


# The teaching instruction shared by both prompts. Explanations should TEACH,
# not just grade: name the misconception, explain the right idea and WHY, and
# give a short memory hook. Kept concise so result.html stays readable.
_COACH = ("You are a supportive exam coach speaking directly to the learner. For each question, in "
          "2-3 short sentences: (1) name the likely misconception behind their answer, (2) explain "
          "the correct idea and WHY it's right, (3) add a brief tip to remember it. Address the "
          "learner in the second person ('your answer'), never as 'the student'. Be encouraging and "
          "specific — never just say 'Wrong' or restate the correct option.\n\n")


def legacy_prompt(wrong_items: list[dict]) -> str:
    """wrong_items: list of {"question","selected","correct"} for wrong answers."""
    prompt = _COACH
    for d in wrong_items:
        prompt += (f"Question: {d['question']}\nYour Answer: {d['selected']}\n"
                   f"Correct Answer: {d['correct']}\n\n")
    prompt += _TAIL
    return prompt


def grounded_prompt(wrong_items: list[dict], context: str) -> str:
    """Same shape, but explanations must draw on the retrieved CONTEXT."""
    prompt = _COACH + "Ground every explanation in the CONTEXT below; do not invent facts beyond it.\n\n"
    prompt += f'CONTEXT:\n"""\n{context}\n"""\n\n'
    for d in wrong_items:
        prompt += (f"Question: {d['question']}\nYour Answer: {d['selected']}\n"
                   f"Correct Answer: {d['correct']}\n\n")
    prompt += _TAIL
    return prompt
