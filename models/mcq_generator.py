# models/mcq_generator.py
"""
MCQ generation entry point.

This module keeps its historical public API — `generate_mcqs(text, num_questions,
difficulty)` and `MCQGenerationError` — so app.py and test_mcq.py are unchanged.
Internally it now delegates to the provider-agnostic LLM layer (core.llm) and the
versioned prompt in core.prompts.mcq, which also fixes the old api.xai.com typo.

When config.AI_PIPELINE == "rag" it routes to the retrieval pipeline. Both
paths enforce the same typed MCQ contract and use source-grounded prompts.
"""
import config
from core.llm import LLMError, get_llm
from core.llm.budget import BudgetExceeded, generation_budget, remaining
from core.models.questions import attach_evidence, normalized, validate_questions
from core.prompts import mcq as mcq_prompts
from core.services.personalization import system_prompt
from utils.text_cleaner import clean_text


class MCQGenerationError(Exception):
    """Custom exception raised when MCQ generation fails."""
    pass


def _structurally_valid(mcqs):
    """Apply the shared deterministic MCQ contract."""
    return validate_questions(mcqs)[0]


def _validate(mcqs, num_questions):
    """Validate using the same contract as RAG and persistence."""
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


@generation_budget
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

    return generate_mcqs_legacy(text, num_questions, difficulty)


@generation_budget
def generate_mcqs_legacy(text, num_questions=5, difficulty="medium"):
    text = clean_text(text)
    if not text:
        raise MCQGenerationError("No readable source text was provided.")
    # The direct generation path keeps its input ceiling and shares RAG's prompt.
    max_chars = 150000
    if len(text) > max_chars:
        text = text[:max_chars]

    llm = get_llm()
    collected: list[dict] = []
    seen: set[str] = set()
    # Ask for the shortfall repeatedly: a single call can under-deliver (models
    # routinely return fewer items than asked on large batches, and long outputs
    # can be truncated), which is why an explicit "25" used to yield a handful.
    # Scale the attempt budget with the request so large sets can fill, and stop
    # early once a round adds nothing new (a stalled model won't improve).
    max_attempts = 3 if num_questions <= 10 else 6
    for attempt in range(max_attempts):
        need = num_questions - len(collected)
        if need <= 0:
            break

        try:
            remaining()
        except BudgetExceeded:
            if collected:
                break
            raise
        prompt = mcq_prompts.rag_prompt(text, need, difficulty)
        if collected:
            prompt += "\nDo not repeat these accepted questions:\n" + "\n".join(q["question"] for q in collected)
        messages = [
            {"role": "system", "content": system_prompt(mcq_prompts.SYSTEM)},
            {"role": "user", "content": prompt},
        ]
        try:
            data = llm.complete_json(messages, max_tokens=config.mcq_token_budget(need))
        except LLMError as e:
            if not collected:
                raise MCQGenerationError(str(e))
            break  # Keep what we already have rather than failing outright.
        except Exception as e:
            if not collected:
                raise MCQGenerationError(f"Unexpected error: {e}")
            break

        before = len(collected)
        try:
            questions = _extract_questions(data)
        except MCQGenerationError:
            if collected:
                break
            raise
        for q in _structurally_valid(questions):
            key = normalized(q["question"])
            if key and key not in seen:
                seen.add(key)
                collected.append(q)

        if len(collected) == before:
            break  # No new unique questions this round; another call won't help.

    if not collected:
        raise MCQGenerationError("No valid questions could be structured from the AI response. Please retry.")

    return attach_evidence(collected[:num_questions], text)
