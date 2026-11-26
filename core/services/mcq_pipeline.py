"""Generation strategies share fail-closed quality and request-local time budgets."""
from __future__ import annotations

import time
from typing import Literal

import config
from core.agents.context_builder import ContextBuilderAgent
from core.agents.context_validation import ContextValidationAgent
from core.agents.planner import PlannerAgent
from core.agents.question import InsufficientContextError, QuestionAgent
from core.agents.retriever import RetrieverAgent
from core.observability import log_event, timed
from core.services.generation_budget import check, generation_scope, remaining
from core.services.generation_metrics import current_metrics
from core.services.ingestion_service import ingest_document
from core.services.mcq_quality import accept_candidates
from core.services.source_evidence import source_locations
from models.mcq_generator import MCQGenerationError


class SourceRejectedError(MCQGenerationError):
    """A source was explicitly rejected, or was empty."""


def _max_attempts(num_questions: int) -> int:
    base = 2 if config.IS_VERCEL else 3
    return base if num_questions <= 10 else base + 2


def _retrieval_query(difficulty: str, topic: str | None) -> str:
    if topic:
        return f"Key concepts and facts about {topic} suitable for {difficulty} exam questions."
    return f"The most important concepts, facts and relationships for {difficulty} exam questions."


@generation_scope
def _collect_mcqs(context: str, num_questions: int, difficulty: str,
                  *, deadline_seconds: int | None = None, on_progress=None,
                  topic: str | None = None, candidate_generator=None,
                  accepted=None, max_attempts=None, recovery: bool = False) -> list[dict]:
    """Accumulate accepted candidates only; later failures preserve earlier work.

    ``accepted`` is an internal handoff of already checked questions from the
    primary loop to recovery. It is never populated from HTTP/model input.
    Rejected candidates can return only by passing all checks on a fresh attempt.
    """
    collected = list(accepted or [])
    generate = candidate_generator or QuestionAgent().run
    rounds = max_attempts if max_attempts is not None else _max_attempts(num_questions)
    # The outer request owns the deadline; the local argument only creates it at entry.
    started = time.perf_counter()
    attempts = 0
    metrics = current_metrics()
    for attempt in range(rounds):
        need = num_questions - len(collected)
        if need <= 0 or (remaining(reserve=4) is not None and remaining(reserve=4) <= 0):
            break
        attempts += 1
        try:
            check()
            if metrics is not None:
                metrics.attempts += 1
                metrics.fallback_attempted |= recovery
            with timed("question", attempt=attempt + 1, requested=need) as stage:
                raw = generate(context, need, difficulty, topic=topic, excluded=tuple(collected))
                stage["mcqs"] = len(raw) if isinstance(raw, list) else 0
            with timed("acceptance", attempt=attempt + 1) as stage:
                result = accept_candidates(raw, context, difficulty, topic=topic, excluded=collected)
                stage.update(accepted=len(result.accepted), rejected=len(result.rejected),
                             unverified=len(result.unverified))
            if metrics is not None:
                metrics.rejected += len(result.rejected)
                metrics.unverified += len(result.unverified)
            check()
            accepted_batch = result.accepted[:need]
            collected.extend(accepted_batch)
            if metrics is not None and recovery:
                metrics.fallback_count += len(accepted_batch)
        except InsufficientContextError:
            if metrics is not None:
                metrics.refusals += 1
            log_event("pipeline.insufficient_context", collected=len(collected), recovery=recovery)
            break
        except Exception as exc:
            if metrics is not None:
                metrics.errors += 1
            log_event("pipeline.attempt_failed", error_type=type(exc).__name__, collected=len(collected))
            break
        if on_progress is not None:
            try:
                on_progress(len(collected))
            except Exception:
                pass
    log_event("pipeline.complete", requested=num_questions, collected=len(collected), attempts=attempts,
              difficulty=difficulty, duration_ms=round((time.perf_counter() - started) * 1000, 2))
    return collected[:num_questions]


def _require_source(context, num_questions, difficulty):
    from core.services.ai_safety import UnsafeSourceError, source_block

    try:
        source_block(context)
        if not context.strip():
            raise SourceRejectedError("No readable source material is available.")
        if not config.FACT_VERIFICATION_ENABLED:
            raise MCQGenerationError("Verified quiz generation is unavailable while fact checking is disabled.")
        if config.CONTEXT_VALIDATION_ENABLED:
            verdict = ContextValidationAgent().run(context, num_questions, difficulty)
            log_event("context.validated", sufficient=verdict["sufficient"], status=verdict.get("status"))
            if not verdict["sufficient"]:
                error = SourceRejectedError if verdict.get("status") == "rejected" else MCQGenerationError
                raise error("The source could not be verified as sufficient. Try fewer questions or different material.")
    except (SourceRejectedError, UnsafeSourceError):
        metrics = current_metrics()
        if metrics is not None:
            metrics.refusals += 1
        raise


@generation_scope
def _generate_checked(context, num_questions, difficulty, *, topic=None,
                      deadline_seconds=None, on_progress=None):
    _require_source(context, num_questions, difficulty)
    collected = _collect_mcqs(context, num_questions, difficulty, topic=topic,
                              deadline_seconds=deadline_seconds, on_progress=on_progress)
    if len(collected) < num_questions and (remaining(reserve=4) is None or remaining(reserve=4) > 0):
        collected = _legacy_fallback(context, num_questions, difficulty, topic=topic,
                                     accepted=collected, on_progress=on_progress,
                                     deadline_seconds=deadline_seconds, source_checked=True)
    if not collected:
        raise MCQGenerationError("No verified questions completed within the generation budget.")
    return collected


@generation_scope
def generate_mcqs_rag(text: str, num_questions: int = 5, difficulty: str = "medium",
                     *, owner: str = "", title: str = "", user_request: str = "",
                     topic: str | None = None) -> list[dict]:
    plan = PlannerAgent().run(user_request) if user_request else None
    if plan:
        num_questions = plan.num_items or num_questions
        difficulty = plan.difficulty or difficulty
        topic = plan.topic or topic
    ingest = ingest_document(text, owner=owner, title=title or "uploaded document", source_type="paste")
    top_k = max(config.RETRIEVAL_TOP_K, min(num_questions * 2, 40))
    retriever = RetrieverAgent()
    hits = retriever.run(ingest["document_id"], _retrieval_query(difficulty, topic), top_k, spread=True)
    if not hits:
        return _legacy_fallback(text[:150000], num_questions, difficulty, topic=topic)
    log_event("retrieval.complete", document_id=ingest["document_id"], **retriever.assess(hits))
    context = ContextBuilderAgent().run(hits)
    # Recovery keeps this exact evidence context, including constraints, rather
    # than letting previously rejected questions escape into a weaker contract.
    with source_locations(hits):
        return _generate_checked(context, num_questions, difficulty, topic=topic)


@generation_scope
def generate_from_text(text: str, num_questions: int = 5, difficulty: str = "medium",
                       *, topic: str | None = None) -> list[dict]:
    context = text[:config.CONTEXT_MAX_CHARS]
    # Topic instructions are separate from evidence and never count as source.
    if config.AI_PIPELINE == "rag":
        return _generate_checked(context, num_questions, difficulty, topic=topic)
    return _legacy_from_context(context, num_questions, difficulty, topic=topic)


@generation_scope
def generate_from_document(document_id: int, num_questions: int = 5,
                           difficulty: str = "medium", *, topic: str | None = None,
                           deadline_seconds: int | None = None, on_progress=None) -> list[dict]:
    top_k = max(config.RETRIEVAL_TOP_K, min(num_questions * 2, 40))
    retriever = RetrieverAgent()
    hits = retriever.run(document_id, _retrieval_query(difficulty, topic), top_k, spread=True)
    if not hits:
        raise ValueError("This knowledge source has no indexed content to practice from yet.")
    log_event("retrieval.complete", document_id=document_id, **retriever.assess(hits))
    context = ContextBuilderAgent().run(hits)
    with source_locations(hits):
        if config.AI_PIPELINE != "rag":
            return _legacy_from_context(context, num_questions, difficulty, topic=topic,
                                        deadline_seconds=deadline_seconds, on_progress=on_progress)
        return _generate_checked(context, num_questions, difficulty, topic=topic,
                                 deadline_seconds=deadline_seconds, on_progress=on_progress)



@generation_scope
def _legacy_from_context(context, num_questions, difficulty, *, topic=None,
                         deadline_seconds=None, on_progress=None) -> list[dict]:
    return _legacy_fallback(context, num_questions, difficulty, topic=topic,
                            deadline_seconds=deadline_seconds, on_progress=on_progress, recovery=False)


@generation_scope
def _legacy_fallback(text, num_questions, difficulty, reason: str = "", *, topic=None,
                     accepted=None, deadline_seconds=None, on_progress=None,
                     source_checked=False, recovery=True, allow_empty=False) -> list[dict]:
    """Request the shortfall through an explicit candidate generator and shared gate.

    No public-generator recursion or global pipeline-mode mutation is necessary.
    Partial results survive failure. No acceptable result is a recoverable error.
    """
    from models.mcq_generator import generate_legacy_candidates
    if not source_checked:
        _require_source(text, num_questions, difficulty)
    if reason:
        log_event("pipeline.legacy_fallback", reason=reason)
    result = _collect_mcqs(text, num_questions, difficulty, topic=topic,
                           candidate_generator=generate_legacy_candidates, accepted=accepted,
                           max_attempts=3 if num_questions <= 10 else 6,
                           deadline_seconds=deadline_seconds, on_progress=on_progress, recovery=recovery)
    if not result and not allow_empty:
        raise MCQGenerationError("No questions passed the quality checks. Please retry or use different material.")
    return result


@generation_scope
def evaluate_candidates(context: str, num_questions: int, difficulty: str, *,
                        topic: str | None = None, strategy: Literal["primary", "fallback"]) -> list[dict]:
    """Isolate one producer for evals while keeping the production acceptance gate."""
    from core.services.ai_safety import UnsafeSourceError

    if strategy not in ("primary", "fallback"):
        raise ValueError("Unknown evaluation strategy.")
    try:
        _require_source(context, num_questions, difficulty)
    except (SourceRejectedError, UnsafeSourceError):
        return []
    if strategy == "fallback":
        return _legacy_fallback(context, num_questions, difficulty, topic=topic,
                               source_checked=True, allow_empty=True)
    return _collect_mcqs(context, num_questions, difficulty, topic=topic)
