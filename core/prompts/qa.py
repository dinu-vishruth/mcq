"""
Quality-assurance prompts. Most QA checks are deterministic (structure, exactly
one correct answer, no duplicate options), done in code. The LLM QA pass is only
used to regenerate questions that fail structural checks, reusing the RAG prompt.
"""

SYSTEM = "You are an expert exam creator that only outputs valid JSON objects."


def regenerate_prompt(context: str, num_questions: int, difficulty: str, reason: str) -> str:
    from core.prompts.mcq import rag_prompt
    base = rag_prompt(context, num_questions, difficulty)
    return f"{base}\n\nIMPORTANT: A previous attempt was rejected because: {reason}. Fix this."
