"""
MCQ generation pipeline (RAG). Orchestrates:

  Planner -> (ingest if needed) -> Retriever -> ContextBuilder
          -> Question -> QA -> Difficulty -> (regenerate failures) -> final MCQs

Public entry point `generate_mcqs_rag(text, num_questions, difficulty)` matches
the legacy generate_mcqs signature and RETURN SHAPE exactly (list of
{question, options:[{label,text}*4], answer_text}) so app.py and every template
are unchanged. Extra keys (bloom, source_hint) may be present and are ignored by
the existing UI.

Never sends the whole document to the LLM: only retrieved, budgeted context.
Falls back to the legacy single-shot generator if anything in the RAG path
fails, so enabling AI_PIPELINE=rag can never make generation worse than before.
"""
from __future__ import annotations

import time

import config
from core.agents.planner import PlannerAgent
from core.agents.retriever import RetrieverAgent
from core.agents.context_builder import ContextBuilderAgent
from core.agents.question import QuestionAgent
from core.agents.difficulty import DifficultyAgent
from core.agents.quality_assurance import QualityAssuranceAgent
from core.services.ingestion_service import ingest_document


def _max_attempts(num_questions: int) -> int:
    """Retry budget for filling a batch.

    Models under-deliver on large requests and the Difficulty agent discards
    off-level questions, so a fixed 2-3 attempts left big sets short. Scale with
    the request; the wall-clock deadline below is the real stop condition.
    """
    base = 2 if config.IS_VERCEL else 3
    return base if num_questions <= 10 else base + 2


def _retrieval_query(difficulty: str, topic: str | None) -> str:
    if topic:
        return f"Key concepts and facts about {topic} suitable for {difficulty} exam questions."
    return f"The most important concepts, facts and relationships for {difficulty} exam questions."


def generate_mcqs_rag(text: str, num_questions: int = 5, difficulty: str = "medium",
                      *, owner: str = "", title: str = "", user_request: str = "") -> list[dict]:
    # 1. Planner: for the standard upload flow, params are explicit; a free-form
    #    user_request (chat/API) can override count/difficulty/topic.
    plan = PlannerAgent().run(user_request) if user_request else None
    if plan:
        num_questions = plan.num_items or num_questions
        difficulty = plan.difficulty or difficulty
    topic = plan.topic if plan else None

    # 2. Ingest (idempotent, cached). Gives us a document_id to retrieve against.
    ingest = ingest_document(text, owner=owner, title=title or "uploaded document",
                             source_type="paste")
    document_id = ingest["document_id"]

    # 3. Retrieve diverse, document-spread context. Scale breadth to question count.
    top_k = max(config.RETRIEVAL_TOP_K, min(num_questions * 2, 40))
    query = _retrieval_query(difficulty, topic)
    hits = RetrieverAgent().run(document_id, query, top_k, spread=True)
    if not hits:
        # No retrievable context (e.g. embedding store empty) -> legacy path.
        return _legacy_fallback(text, num_questions, difficulty)

    # 4. Build token-budgeted context (retrieved chunks only).
    context = ContextBuilderAgent().run(hits)

    # 5. Generate -> QA (structural) -> Difficulty (level match), regenerating
    #    only what's missing, up to a small number of attempts.
    qa = QualityAssuranceAgent()
    question_agent = QuestionAgent()
    difficulty_agent = DifficultyAgent()

    collected: list[dict] = []
    attempts = 0
    max_attempts = _max_attempts(num_questions)
    # Stop retrying before the serverless function limit so we never get killed
    # mid-generation. Leave headroom below Vercel's 60s cap for the final response.
    deadline = time.monotonic() + (config.PIPELINE_DEADLINE_SECONDS - config.LLM_TIMEOUT)
    while len(collected) < num_questions and attempts < max_attempts:
        if attempts >= 1 and time.monotonic() >= deadline:
            break  # out of time budget; return what we have so far
        attempts += 1
        need = num_questions - len(collected)
        try:
            raw = question_agent.run(context, need, difficulty)
        except Exception as e:
            if attempts == 1:
                return _legacy_fallback(text, num_questions, difficulty, reason=str(e))
            break

        valid, _rejected = qa.run(raw)
        graded = difficulty_agent.run(valid, difficulty)
        collected.extend(graded["matched"])
        # De-duplicate by question text across attempts.
        seen, deduped = set(), []
        for q in collected:
            key = q["question"].strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(q)
        collected = deduped

    if not collected:
        return _legacy_fallback(text, num_questions, difficulty)

    return collected[:num_questions]


def generate_from_document(document_id: int, num_questions: int = 5,
                           difficulty: str = "medium", *, topic: str | None = None) -> list[dict]:
    """Generate MCQs from an ALREADY-INGESTED document (Practice from Knowledge).

    Skips ingestion — the document is already chunked and embedded — and runs the
    same Retriever -> ContextBuilder -> Question -> QA -> Difficulty loop. Returns
    the same MCQ shape as generate_mcqs_rag. Raises ValueError if the document has
    no retrievable context so the caller can surface a clear message.
    """
    top_k = max(config.RETRIEVAL_TOP_K, min(num_questions * 2, 40))
    query = _retrieval_query(difficulty, topic)
    hits = RetrieverAgent().run(document_id, query, top_k, spread=True)
    if not hits:
        raise ValueError("This knowledge source has no indexed content to practice from yet.")

    context = ContextBuilderAgent().run(hits)

    qa = QualityAssuranceAgent()
    question_agent = QuestionAgent()
    difficulty_agent = DifficultyAgent()

    collected: list[dict] = []
    attempts = 0
    max_attempts = _max_attempts(num_questions)
    deadline = time.monotonic() + (config.PIPELINE_DEADLINE_SECONDS - config.LLM_TIMEOUT)
    while len(collected) < num_questions and attempts < max_attempts:
        if attempts >= 1 and time.monotonic() >= deadline:
            break
        attempts += 1
        need = num_questions - len(collected)
        try:
            raw = question_agent.run(context, need, difficulty)
        except Exception:
            break
        valid, _rejected = qa.run(raw)
        graded = difficulty_agent.run(valid, difficulty)
        collected.extend(graded["matched"])
        seen, deduped = set(), []
        for q in collected:
            k = q["question"].strip().lower()
            if k not in seen:
                seen.add(k)
                deduped.append(q)
        collected = deduped

    if not collected:
        raise ValueError("Could not generate questions from this source. Try a different difficulty or topic.")
    return collected[:num_questions]


def _legacy_fallback(text, num_questions, difficulty, reason: str = "") -> list[dict]:
    """Use the original single-shot generator. Import locally to avoid a cycle."""
    if reason:
        print(f"[mcq_pipeline] RAG fallback to legacy generator: {reason}")
    # Temporarily force legacy to avoid recursion back into this pipeline.
    prev = config.AI_PIPELINE
    config.AI_PIPELINE = "legacy"
    try:
        from models.mcq_generator import generate_mcqs
        return generate_mcqs(text, num_questions=num_questions, difficulty=difficulty)
    finally:
        config.AI_PIPELINE = prev
