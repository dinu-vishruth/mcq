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

from core.agents.base import Agent
from core.llm import get_llm, LLMError
from core.prompts import mcq as mcq_prompts


class QuestionAgent(Agent):
    name = "question"

    def run(self, context: str, num_questions: int, difficulty: str) -> list[dict]:
        prompt = mcq_prompts.rag_prompt(context, num_questions, difficulty)
        messages = [
            {"role": "system", "content": mcq_prompts.SYSTEM},
            {"role": "user", "content": prompt},
        ]
        data = get_llm().complete_json(messages)
        if isinstance(data, dict) and "questions" in data:
            mcqs = data["questions"]
        elif isinstance(data, list):
            mcqs = data
        else:
            raise LLMError("JSON structure is missing 'questions' array key.")
        self._log(f"generated {len(mcqs)} raw questions ({difficulty})")
        return mcqs
