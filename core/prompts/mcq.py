"""
MCQ generation prompts.

`legacy_prompt` reproduces the exact wording the current single-shot generator
uses, so behaviour is identical when AI_PIPELINE=legacy. `rag_prompt` is the
retrieval-grounded variant used by the QuestionAgent in Phase 4: it forbids the
model from inventing facts beyond the supplied context chunks.
"""

SYSTEM = "You are an expert exam creator that only outputs valid JSON objects."

_STRUCTURE = """You MUST respond with a JSON object containing a "questions" key, which holds an array of exactly {num_questions} MCQ objects.
Each MCQ object must have exactly the following structure:
{{
  "question": "The question text",
  "options": [
    {{"label": "A", "text": "Option A text"}},
    {{"label": "B", "text": "Option B text"}},
    {{"label": "C", "text": "Option C text"}},
    {{"label": "D", "text": "Option D text"}}
  ],
  "answer_text": "The exact text of the correct option (must match one of the options' text)"
}}"""


def legacy_prompt(text: str, num_questions: int, difficulty: str) -> str:
    """Byte-for-byte equivalent of the original models/mcq_generator.py prompt."""
    return f"""You are an expert exam creator. Create {num_questions} Multiple Choice Questions from the text below.
The difficulty must be exactly: {difficulty.upper()}.
Make the distractors highly plausible and challenging (unless difficulty is 'easy').

{_STRUCTURE.format(num_questions=num_questions)}

TEXT:
\"\"\"
{text}
\"\"\"
"""


# Bloom's Taxonomy guidance per difficulty band, used by the RAG QuestionAgent.
BLOOM_GUIDANCE = {
    "easy": "Bloom levels: Remember, Understand. Test recall of definitions and stated facts.",
    "medium": "Bloom levels: Apply, Analyze. Require applying a concept or interpreting a relationship.",
    "hard": "Bloom levels: Analyze, Evaluate, Create. Require multi-step reasoning, comparison, or judgement.",
}


_RAG_STRUCTURE = """You MUST respond with a JSON object containing a "questions" key, which holds an array of exactly {num_questions} MCQ objects.
Each MCQ object must have exactly the following structure:
{{
  "question": "The question text",
  "options": [
    {{"label": "A", "text": "Option A text"}},
    {{"label": "B", "text": "Option B text"}},
    {{"label": "C", "text": "Option C text"}},
    {{"label": "D", "text": "Option D text"}}
  ],
  "answer_text": "The exact text of the correct option (must match one of the options' text)",
  "explanation": "1-2 sentences, grounded in the CONTEXT, explaining why the answer is correct",
  "bloom": "The Bloom level (e.g. Apply)",
  "source_hint": "A short quote (<=8 words) from the CONTEXT that supports the answer"
}}"""


def rag_prompt(context: str, num_questions: int, difficulty: str) -> str:
    """Retrieval-grounded prompt. The model may only use the provided context.

    Enforces the brief's generation contract: draw ONLY on the context, never
    invent facts, obey strict distractor rules, ground the explanation, and
    self-verify before returning. If the context cannot support the request,
    the model returns a single INSUFFICIENT_CONTEXT marker instead of guessing.
    """
    bloom = BLOOM_GUIDANCE.get(difficulty, BLOOM_GUIDANCE["medium"])
    return f"""You are an expert exam creator. Using ONLY the CONTEXT below, create {num_questions} Multiple Choice Questions.
The difficulty must be exactly: {difficulty.upper()}. {bloom}

GROUNDING RULES (critical):
- Use ONLY facts stated in the CONTEXT. Do NOT use outside knowledge. Do NOT infer facts the context does not state.
- Every question must be answerable purely from the CONTEXT.
- The correct answer must be directly supported by the CONTEXT.

DISTRACTOR RULES (the 3 wrong options):
- Each distractor must be factually INCORRECT according to the CONTEXT.
- Never write a distractor that is also correct or partially correct.
- Never contradict the CONTEXT with a claim the CONTEXT actually supports elsewhere.
- Never invent unrelated concepts; keep distractors on the same subject as the CONTEXT.
- All four options must have similar length, style, and specificity so the answer isn't guessable from format.

For each question also provide:
- "explanation": 1-2 sentences, grounded in the CONTEXT, saying why the correct answer is right.
- "bloom": the Bloom level.
- "source_hint": a short quote (<=8 words) from the CONTEXT supporting the answer.

SELF-VERIFICATION (do this silently before you output):
For every question confirm: (1) the question is supported by the CONTEXT, (2) the correct answer is supported, (3) exactly ONE option is correct and the other three are wrong, (4) the explanation matches the CONTEXT. If any check fails, rewrite that question before returning. Only output questions that pass all four checks.

If the CONTEXT does not contain enough information to write {num_questions} grounded questions, respond with exactly this JSON and nothing else: {{"status": "INSUFFICIENT_CONTEXT"}}

{_RAG_STRUCTURE.format(num_questions=num_questions)}

CONTEXT:
\"\"\"
{context}
\"\"\"
"""
