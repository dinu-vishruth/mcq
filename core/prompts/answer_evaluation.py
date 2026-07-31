"""
Answer-evaluation prompt: conceptual grading of a free-text student answer.

This is the opposite of keyword matching. The grader compares the student's
answer to the reference answer AND the retrieved context by MEANING: paraphrases,
synonyms, and alternative-but-correct explanations are all accepted; only genuine
conceptual misunderstandings are marked wrong. Extra correct knowledge is
rewarded, never penalised.
"""

SYSTEM = "You are an expert tutor grading understanding, not wording. You only output valid JSON objects."


def evaluate_prompt(question: str, reference_answer: str, student_answer: str,
                    context: str = "", *, reevaluate: bool = False) -> str:
    ctx_block = f'CONTEXT (ground truth):\n"""\n{context}\n"""\n\n' if (context or "").strip() else ""
    nudge = (
        "\nYou were previously unsure. Re-read carefully and commit to a well-justified verdict.\n"
        if reevaluate else ""
    )
    return f"""Grade the STUDENT ANSWER by conceptual understanding, NOT by wording.

Rules:
- Accept paraphrasing, synonyms, and alternative correct explanations.
- Never require exact phrasing and never compare exact words.
- Reward additional correct information; do NOT penalise correct extra detail.
- Mark incorrect ONLY for a genuine conceptual misunderstanding or a false claim.
- If the answer is partially correct, identify exactly which points are missing.
- "score" is 0-100 (conceptual completeness). "confidence" is your certainty in [0,1].
- "model_answer" is a concise ideal answer grounded in the reference/context.
{nudge}
{ctx_block}QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

STUDENT ANSWER:
{student_answer}

Respond with a JSON object exactly like:
{{
  "is_correct": true,
  "score": 92,
  "confidence": 0.97,
  "concept_match": true,
  "strengths": ["..."],
  "missing_points": ["..."],
  "incorrect_points": ["..."],
  "feedback": "constructive, teaching feedback addressed to the learner in the second person",
  "model_answer": "concise ideal answer"
}}
"""
