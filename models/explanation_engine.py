# models/explanation_engine.py
"""
Answer-explanation entry point.

Keeps its historical public API — `explain_answers(details) -> list[str]`,
one entry per question, positionally aligned — consumed by result.html as
`explanations[loop.index0]`. Internally delegates to the provider-agnostic
LLM layer and the versioned prompt in core.prompts.explanation.

Fallback behaviour (no key, no wrong answers, or any API error) is identical
to the original implementation, so results never regress to a crash.
"""
import config
from core.llm import get_llm, LLMError
from core.prompts import explanation as prompts


def _static_fallback(details):
    return [
        "✅ Correct!" if d["is_correct"] else f"❌ Incorrect. The answer is '{d['correct']}'."
        for d in details
    ]


def explain_answers(details):
    """
    details: list of {question, selected, correct, is_correct}.
    Returns a list of explanation strings, one per detail, in order.
    """
    if not config.LLM_API_KEY:
        return _static_fallback(details)

    # Only ask the model about questions the student actually got wrong.
    wrong = [d for d in details if not d["is_correct"] and d["selected"]]
    if not wrong:
        return [
            f"✅ Correct! '{d['selected']}' is exactly right." if d["is_correct"]
            else f"⏭️ Not answered. The correct answer is '{d['correct']}'."
            for d in details
        ]

    messages = [
        {"role": "system", "content": prompts.SYSTEM},
        {"role": "user", "content": prompts.legacy_prompt(wrong)},
    ]

    try:
        data = get_llm().complete_json(messages)
        if isinstance(data, dict) and "explanations" in data:
            ai = data["explanations"]
        elif isinstance(data, list):
            ai = data
        else:
            ai = []

        merged, ai_idx = [], 0
        for d in details:
            if d["is_correct"]:
                merged.append(f"✅ Correct! '{d['selected']}' is exactly right.")
            elif not d["selected"]:
                merged.append(f"⏭️ Not answered. The correct answer is '{d['correct']}'.")
            elif ai_idx < len(ai):
                merged.append(f"❌ Incorrect. {ai[ai_idx]}")
                ai_idx += 1
            else:
                merged.append(f"❌ Incorrect. The correct answer is '{d['correct']}'.")
        return merged
    except (LLMError, Exception) as e:
        print(f"[ERROR] Failed to generate AI explanations: {e}")
        return _static_fallback(details)
