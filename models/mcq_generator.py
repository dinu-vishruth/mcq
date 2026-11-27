# models/mcq_generator.py
"""Public MCQ API; every strategy publishes through shared quality guardrails.

Legacy candidate generation is explicit and nonrecursive. Raw candidate batches
are never acceptable quiz results until the common acceptance gate passes them.
"""
import config
from core.llm import get_llm
from core.prompts import mcq as mcq_prompts
from core.services.personalization import system_prompt
from utils.text_cleaner import clean_text


class MCQGenerationError(Exception):
    """Custom exception raised when MCQ generation fails."""
    pass


def _structurally_valid(mcqs):
    """Compatibility helper; generation also requires shared semantic acceptance."""
    from core.agents.quality_assurance import QualityAssuranceAgent
    return QualityAssuranceAgent().run(mcqs)[0]


def _validate(mcqs, num_questions):
    """Shared structural validation for compatibility callers."""
    valid = _structurally_valid(mcqs)
    if not valid:
        raise MCQGenerationError("No valid questions could be structured from the AI response. Please retry.")
    return valid[:num_questions]


def _extract_questions(data):
    """Pull the question array out of either accepted response shape."""
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    if isinstance(data, list):
        return data
    raise MCQGenerationError("JSON structure is missing 'questions' array key.")


def generate_mcqs(text, num_questions=5, difficulty="medium"):
    """
    Generate high-quality MCQs. Difficulty: easy / medium / hard.
    Returns a list of MCQ dicts: {question, options:[{label,text}*4], answer_text}.
    """
    text = clean_text(text)
    if not config.llm_key_present():
        # Full diagnostic (variable names, .env path) to the log for whoever runs
        # the server; a plain message to the learner, who can't act on config.
        print(f"[mcq] generation blocked: {config.missing_key_message()}")
        raise MCQGenerationError(config.USER_FACING_AI_UNAVAILABLE)
    if not text:
        raise MCQGenerationError("No text found. Please upload a valid document or provide some text.")

    # RAG path (Phase 4): delegate to the agent pipeline when enabled.
    if config.AI_PIPELINE == "rag":
        try:
            from core.services.mcq_pipeline import generate_mcqs_rag
        except ImportError:
            pass  # Pipeline not present yet; fall through to legacy.
        else:
            return generate_mcqs_rag(text, num_questions=num_questions, difficulty=difficulty)

    from core.services.mcq_pipeline import _legacy_from_context
    return _legacy_from_context(text[:150000], num_questions, difficulty)


def generate_legacy_candidates(text, num_questions=5, difficulty="medium", *, topic=None, excluded=(), retry_feedback=()):
    """One raw candidate batch, never directly published and never re-enters RAG."""
    prompt = mcq_prompts.legacy_prompt(text, num_questions, difficulty, topic=topic, excluded=excluded, retry_feedback=retry_feedback)
    data = get_llm().complete_json([
        {"role": "system", "content": system_prompt(mcq_prompts.SYSTEM)},
        {"role": "user", "content": prompt},
    ], max_tokens=config.mcq_token_budget(num_questions))
    if isinstance(data, dict) and data.get("status") == "INSUFFICIENT_CONTEXT":
        from core.agents.question import InsufficientContextError
        raise InsufficientContextError("The source is insufficient to generate grounded questions.")
    questions = _extract_questions(data)
    if not isinstance(questions, list):
        raise MCQGenerationError("The AI service did not return a questions array.")
    return questions
