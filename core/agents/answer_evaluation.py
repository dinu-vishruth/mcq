"""
Answer Evaluation Agent: conceptual grading of a free-text student answer.

Replaces keyword/exact-match grading for open-ended answers. Given the question,
a reference answer, the student's answer, and (optionally) retrieved context, it
evaluates MEANING — accepting paraphrase, synonyms, and alternative correct
explanations, rewarding extra correct knowledge, and rejecting only genuine
conceptual errors.

Output is the exact JSON contract from the brief:
    {is_correct, score, confidence, concept_match, strengths, missing_points,
     incorrect_points, feedback, model_answer}

Per the failure-handling rule, if the model's confidence is below
ANSWER_EVAL_REEVAL_CONFIDENCE the agent asks it to re-evaluate once.

COMPONENT ONLY: the app is MCQ-only today, so nothing wires a free-text answer
into this agent yet. It's a tested, ready building block for a future
short-answer mode. It is never called from the current quiz/submit path.
"""
from __future__ import annotations

import config
from core.agents.base import Agent
from core.llm import get_llm, LLMError
from core.prompts import answer_evaluation as prompts


class AnswerEvaluationAgent(Agent):
    name = "answer_evaluation"

    def run(self, question: str, reference_answer: str, student_answer: str,
            *, context: str = "", document_id: int | None = None) -> dict:
        """Return the brief's evaluation JSON. Safe, structured fallback on error."""
        if not (student_answer or "").strip():
            return self._blank(reference_answer, "No answer was provided.")

        if not context and document_id:
            context = self._retrieve_context(question, document_id)

        result = self._evaluate(question, reference_answer, student_answer, context, reevaluate=False)
        if result is None:
            return self._blank(reference_answer, "Evaluation is temporarily unavailable.")

        # Failure handling: a low-confidence verdict gets one re-evaluation pass.
        try:
            confidence = float(result.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        if confidence < config.ANSWER_EVAL_REEVAL_CONFIDENCE:
            self._log(f"confidence {confidence:.2f} < threshold; re-evaluating once")
            second = self._evaluate(question, reference_answer, student_answer, context, reevaluate=True)
            if second is not None:
                result = second

        return self._normalise(result, reference_answer)

    # -- internals ---------------------------------------------------------
    def _evaluate(self, question, reference_answer, student_answer, context, *, reevaluate):
        try:
            data = get_llm().complete_json([
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": prompts.evaluate_prompt(
                    question, reference_answer, student_answer, context, reevaluate=reevaluate)},
            ], temperature=config.LLM_GENERATION_TEMPERATURE)
            self._log(f"evaluated (concept_match={data.get('concept_match')}, "
                      f"score={data.get('score')})" if isinstance(data, dict) else "evaluated")
            return data if isinstance(data, dict) else None
        except (LLMError, Exception) as e:
            self._log(f"evaluation error: {e}")
            return None

    def _retrieve_context(self, question, document_id):
        try:
            from core.agents.retriever import RetrieverAgent
            from core.agents.context_builder import ContextBuilderAgent
            hits = RetrieverAgent().run(document_id, question, config.RETRIEVAL_TOP_K, spread=False)
            return ContextBuilderAgent().run(hits, max_chars=4000) if hits else ""
        except Exception as e:
            self._log(f"context retrieval skipped: {e}")
            return ""

    @staticmethod
    def _as_list(v):
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        if v is None or v == "":
            return []
        return [str(v)]

    def _normalise(self, data: dict, reference_answer: str) -> dict:
        try:
            score = int(round(float(data.get("score", 0))))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(100, score))
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return {
            "is_correct": bool(data.get("is_correct", False)),
            "score": score,
            "confidence": round(confidence, 2),
            "concept_match": bool(data.get("concept_match", data.get("is_correct", False))),
            "strengths": self._as_list(data.get("strengths")),
            "missing_points": self._as_list(data.get("missing_points")),
            "incorrect_points": self._as_list(data.get("incorrect_points")),
            "feedback": str(data.get("feedback", "") or ""),
            "model_answer": str(data.get("model_answer", "") or reference_answer or ""),
        }

    @staticmethod
    def _blank(reference_answer: str, feedback: str) -> dict:
        return {
            "is_correct": False, "score": 0, "confidence": 0.0, "concept_match": False,
            "strengths": [], "missing_points": [], "incorrect_points": [],
            "feedback": feedback, "model_answer": reference_answer or "",
        }
