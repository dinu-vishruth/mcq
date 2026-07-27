"""
Explanation Agent: educational explanations for answers.

Preserves the historical explain_answers(details) -> list[str] contract exactly
(one entry per question, in order), so result.html is unchanged. When a
document_id is supplied it retrieves relevant context and grounds explanations
in the source material; otherwise it uses the legacy ungrounded prompt.

The module-level explain_answers() keeps the old import path working; the
class form is what the pipeline/services use.
"""
from __future__ import annotations

import config
from core.agents.base import Agent
from core.llm import get_llm, LLMError
from core.prompts import explanation as prompts


class ExplanationAgent(Agent):
    name = "explanation"

    def run(self, details: list[dict], *, document_id: int | None = None) -> list[str]:
        if not config.LLM_API_KEY:
            return self._static(details)

        wrong = [d for d in details if not d["is_correct"] and d["selected"]]
        if not wrong:
            return [
                f"✅ Correct! '{d['selected']}' is exactly right." if d["is_correct"]
                else f"⏭️ Not answered. The correct answer is '{d['correct']}'."
                for d in details
            ]

        context = self._maybe_context(wrong, document_id)
        if context:
            user = prompts.grounded_prompt(wrong, context)
        else:
            user = prompts.legacy_prompt(wrong)

        try:
            data = get_llm().complete_json([
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": user},
            ])
            ai = data.get("explanations", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            return self._merge(details, ai)
        except (LLMError, Exception) as e:
            self._log(f"explanation fallback: {e}")
            return self._static(details)

    def _maybe_context(self, wrong, document_id):
        if not document_id:
            return ""
        try:
            from core.agents.retriever import RetrieverAgent
            from core.agents.context_builder import ContextBuilderAgent
            query = " ".join(d["question"] for d in wrong[:5])
            hits = RetrieverAgent().run(document_id, query, config.RETRIEVAL_TOP_K, spread=False)
            return ContextBuilderAgent().run(hits, max_chars=4000) if hits else ""
        except Exception as e:
            self._log(f"context retrieval skipped: {e}")
            return ""

    @staticmethod
    def _static(details):
        return [
            "✅ Correct!" if d["is_correct"] else f"❌ Incorrect. The answer is '{d['correct']}'."
            for d in details
        ]

    @staticmethod
    def _merge(details, ai):
        merged, ai_idx = [], 0
        for d in details:
            if d["is_correct"]:
                merged.append(f"✅ Correct! '{d['selected']}' is exactly right.")
            elif not d["selected"]:
                merged.append(f"⏭️ Not answered. The correct answer is '{d['correct']}'.")
            elif ai_idx < len(ai):
                merged.append(f"❌ Incorrect. {ai[ai_idx]}")
                ai_idx += 1
            else:
                merged.append(f"❌ Incorrect. The correct answer is '{d['correct']}'.")
        return merged
