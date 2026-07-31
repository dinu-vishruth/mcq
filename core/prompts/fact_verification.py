"""
Fact-verification prompt. After generation, the FactVerificationAgent checks
each MCQ against the retrieved CONTEXT in one batched call. It is deliberately
skeptical: its job is to REJECT questions that aren't fully grounded, have more
than one defensible answer, are ambiguous, or whose distractors could be argued
correct — the failure modes that produce "wrong" MCQs like the WHERE/HAVING mixup.
"""

SYSTEM = "You are a meticulous fact-checker for exam questions. You only output valid JSON objects."


def verify_prompt(mcqs: list[dict], context: str) -> str:
    """mcqs: [{index, question, options:[text...], answer_text, explanation?}]."""
    blocks = []
    for m in mcqs:
        opts = "\n".join(f"    - {o}" for o in m["options"])
        block = (
            f'#{m["index"]}\n'
            f'  Question: {m["question"]}\n'
            f'  Options:\n{opts}\n'
            f'  Marked correct: {m["answer_text"]}'
        )
        if m.get("explanation"):
            block += f'\n  Explanation: {m["explanation"]}'
        blocks.append(block)
    listing = "\n\n".join(blocks)

    return f"""Verify each multiple-choice question STRICTLY against the CONTEXT. Use ONLY the CONTEXT as ground truth; ignore any outside knowledge.

For each question, check ALL of the following:
1. supported: the question is answerable from the CONTEXT.
2. answer_correct: the option marked correct is actually correct per the CONTEXT.
3. distractors_wrong: the other three options are each clearly INCORRECT per the CONTEXT (none is also correct or partially correct).
4. single_answer: exactly one option is correct — not zero, not two.
5. unambiguous: the wording has one clear interpretation.
6. explanation_matches: if an explanation is given, it agrees with the CONTEXT.

A question PASSES only if 1-6 are all true. If any check fails, it FAILS.
Also return a confidence in [0,1] for your verdict.

Respond with a JSON object exactly like:
{{"results": [{{"index": 0, "passed": true, "confidence": 0.95, "issue": ""}}, ...]}}
Set "issue" to a short reason ONLY when passed is false.

CONTEXT:
\"\"\"
{context}
\"\"\"

QUESTIONS:
{listing}
"""
