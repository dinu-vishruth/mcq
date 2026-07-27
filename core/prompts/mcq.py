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


def rag_prompt(context: str, num_questions: int, difficulty: str) -> str:
    """Retrieval-grounded prompt. The model may only use the provided context."""
    bloom = BLOOM_GUIDANCE.get(difficulty, BLOOM_GUIDANCE["medium"])
    return f"""You are an expert exam creator. Using ONLY the CONTEXT below, create {num_questions} Multiple Choice Questions.
The difficulty must be exactly: {difficulty.upper()}. {bloom}
Do NOT use any knowledge beyond the CONTEXT. If the context is insufficient for a question, base it on what the context does say.
Make distractors plausible and grounded in the same subject matter (unless difficulty is 'easy').
Add a "bloom" field naming the Bloom level, and a "source_hint" field with a short quote (<=8 words) from the context that supports the answer.

{_STRUCTURE.format(num_questions=num_questions)}

CONTEXT:
\"\"\"
{context}
\"\"\"
"""
