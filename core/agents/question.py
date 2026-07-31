"""
Question Agent: generate MCQs from retrieved context, following Bloom's Taxonomy
for the requested difficulty. Uses the retrieval-grounded prompt so the model
draws only on the supplied context, not its own memory.

Returns raw MCQ dicts in the canonical shape
    {question, options:[{label,text}*4], answer_text, [bloom], [source_hint]}
The extra bloom/source_hint keys are harmless to templates (which read only the
three core keys) and are used by the Difficulty/QA agents.
"""
from __future__ import annotations

import config
from core.agents.base import Agent
from core.llm import get_llm, LLMError
from core.prompts import mcq as mcq_prompts


class InsufficientContextError(LLMError):
    """The model reported the retrieved context can't support grounded MCQs."""


class QuestionAgent(Agent):
    name = "question"

    def run(self, context: str, num_questions: int, difficulty: str) -> list[dict]:
        prompt = mcq_prompts.rag_prompt(context, num_questions, difficulty)
        messages = [
            {"role": "system", "content": mcq_prompts.SYSTEM},
            {"role": "user", "content": prompt},
        ]
        # Low temperature keeps questions faithful to the context, not the
        # model's own memory (the source of hallucinated / off-context options).
        data = get_llm().complete_json(
            messages,
            max_tokens=config.mcq_token_budget(num_questions),
            temperature=config.LLM_GENERATION_TEMPERATURE,
        )
        if isinstance(data, dict) and str(data.get("status", "")).upper() == "INSUFFICIENT_CONTEXT":
            self._log("model reported INSUFFICIENT_CONTEXT")
            raise InsufficientContextError("Retrieved context is insufficient to generate grounded questions.")
        if isinstance(data, dict) and "questions" in data:
            mcqs = data["questions"]
        elif isinstance(data, list):
            mcqs = data
        else:
            raise LLMError("JSON structure is missing 'questions' array key.")
        self._log(f"generated {len(mcqs)} raw questions ({difficulty})")
        return mcqs
