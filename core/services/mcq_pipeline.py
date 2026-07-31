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
from core.agents.context_validation import ContextValidationAgent
from core.agents.question import QuestionAgent, InsufficientContextError
from core.agents.difficulty import DifficultyAgent
from core.agents.fact_verification import FactVerificationAgent
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


def _collect_mcqs(context: str, num_questions: int, difficulty: str) -> list[dict]:
    """Generate -> QA -> Fact-verify -> Difficulty, regenerating only the
    shortfall, until we have `num_questions` or the retry/time budget runs out.

    Shared by both public entry points. Emits the structured per-stage log from
    the brief. Raises InsufficientContextError only if the FIRST attempt reports
    it (so the caller can fall back to the fuller-text path); later attempts that
    hit it just stop the loop with whatever was collected.
    """
    qa = QualityAssuranceAgent()
    question_agent = QuestionAgent()
    verifier = FactVerificationAgent()
    difficulty_agent = DifficultyAgent()

    collected: list[dict] = []
    attempts = 0
    max_attempts = _max_attempts(num_questions)
    deadline = time.monotonic() + (config.PIPELINE_DEADLINE_SECONDS - config.LLM_TIMEOUT)

    while len(collected) < num_questions and attempts < max_attempts:
        if attempts >= 1 and time.monotonic() >= deadline:
            break  # out of time budget; return what we have
        attempts += 1
        need = num_questions - len(collected)

        raw = question_agent.run(context, need, difficulty)  # may raise InsufficientContextError

        # Structural QA (deterministic contract guard).
        valid, _rejected = qa.run(raw)

        # Fact verification (the new grounding gate) before we accept anything.
        if config.FACT_VERIFICATION_ENABLED and valid:
            verified, rejected = verifier.run(valid, context)
            print(f"[mcq_pipeline] Fact Checker: {len(verified)} PASS, {len(rejected)} rejected (attempt {attempts})")
            valid = verified
        else:
            print(f"[mcq_pipeline] Fact Checker: skipped (attempt {attempts})")

        # Difficulty match (discard off-level questions).
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

    print(f"[mcq_pipeline] Collected {len(collected)}/{num_questions} MCQs in {attempts} attempt(s)")
    return collected[:num_questions]


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
    retriever = RetrieverAgent()
    hits = retriever.run(document_id, query, top_k, spread=True)
    if not hits:
        # No retrievable context (e.g. embedding store empty) -> legacy path.
        return _legacy_fallback(text, num_questions, difficulty)

    stats = retriever.assess(hits)
    print(f"[mcq_pipeline] Retriever: {stats['count']} chunks, avg similarity "
          f"{stats['avg_score']:.3f}{' (low confidence)' if stats['low_confidence'] else ''}")

    # 4. Build token-budgeted context (retrieved chunks only).
    context = ContextBuilderAgent().run(hits)

    # 4b. Context validation: if the context confidently can't support grounded
    #     questions, fall back to the fuller-text legacy path instead of letting
    #     the generator invent facts.
    if config.CONTEXT_VALIDATION_ENABLED:
        verdict = ContextValidationAgent().run(context, num_questions, difficulty)
        print(f"[mcq_pipeline] Context Validator: {'PASS' if verdict['sufficient'] else 'INSUFFICIENT'}")
        if not verdict["sufficient"]:
            return _legacy_fallback(text, num_questions, difficulty,
                                    reason=f"insufficient context: {verdict.get('reason', '')}")

    # 5. Generate -> QA -> Fact-verify -> Difficulty, filling the shortfall.
    try:
        collected = _collect_mcqs(context, num_questions, difficulty)
    except InsufficientContextError as e:
        return _legacy_fallback(text, num_questions, difficulty, reason=str(e))
    except Exception as e:
        return _legacy_fallback(text, num_questions, difficulty, reason=str(e))

    if not collected:
        return _legacy_fallback(text, num_questions, difficulty)

    return collected


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
    retriever = RetrieverAgent()
    hits = retriever.run(document_id, query, top_k, spread=True)
    if not hits:
        raise ValueError("This knowledge source has no indexed content to practice from yet.")

    stats = retriever.assess(hits)
    print(f"[mcq_pipeline] Retriever: {stats['count']} chunks, avg similarity "
          f"{stats['avg_score']:.3f}{' (low confidence)' if stats['low_confidence'] else ''}")

    context = ContextBuilderAgent().run(hits)

    # In legacy mode, match the instant-quiz path: a single-shot generation from
    # the retrieved context. The full RAG verification loop (context validation +
    # fact verification + difficulty grading) fires several sequential LLM calls,
    # which — especially under 429 backoff — makes this path slow enough to look
    # hung. Only run that heavier pipeline when RAG is explicitly enabled.
    if config.AI_PIPELINE != "rag":
        try:
            mcqs = _legacy_from_context(context, num_questions, difficulty)
        except Exception as e:
            raise ValueError(f"Could not generate questions from this source: {e}")
        if not mcqs:
            raise ValueError("Could not generate questions from this source. Try a different difficulty or topic.")
        return mcqs[:num_questions]

    # There's no fuller-text fallback here (the raw document text isn't in hand),
    # so a confident INSUFFICIENT verdict surfaces as a clear, actionable error
    # rather than silently producing hallucinated questions.
    if config.CONTEXT_VALIDATION_ENABLED:
        verdict = ContextValidationAgent().run(context, num_questions, difficulty)
        print(f"[mcq_pipeline] Context Validator: {'PASS' if verdict['sufficient'] else 'INSUFFICIENT'}")
        if not verdict["sufficient"]:
            raise ValueError("This source doesn't have enough indexed content to generate grounded "
                             "questions. Try a broader topic or add more material.")

    try:
        collected = _collect_mcqs(context, num_questions, difficulty)
    except InsufficientContextError:
        raise ValueError("This source doesn't have enough indexed content to generate grounded "
                         "questions. Try a broader topic or add more material.")

    if not collected:
        raise ValueError("Could not generate questions from this source. Try a different difficulty or topic.")
    return collected


def _legacy_from_context(context, num_questions, difficulty) -> list[dict]:
    """Single-shot generation using the retrieved CONTEXT as the source text.

    Used by generate_from_document in legacy mode so making a quiz from a saved
    resource is as fast as the instant-upload path (one LLM call, not the full
    RAG verification loop). The context is already retrieved + budgeted, so we
    feed it straight to the legacy generator.
    """
    print(f"[mcq_pipeline] legacy single-shot from context ({len(context)} chars)")
    return _legacy_fallback(context, num_questions, difficulty)


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
