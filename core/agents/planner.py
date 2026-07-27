"""
Planner Agent: understand user intent and extract parameters.

For MCQ generation from the existing UI (which passes explicit num_questions and
difficulty), the planner is a cheap pass-through -- we already know the intent.
The LLM classification path exists for free-form requests (Phase 7 features and
a future chat box) and degrades gracefully to generate_mcqs on any failure.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.agents.base import Agent
from core.llm import get_llm, LLMError
from core.prompts import planner as prompts


@dataclass
class Plan:
    intent: str = "generate_mcqs"
    num_items: int | None = None
    difficulty: str | None = None
    topic: str | None = None


class PlannerAgent(Agent):
    name = "planner"

    def run(self, user_request: str) -> Plan:
        if not user_request or not user_request.strip():
            return Plan()
        try:
            data = get_llm().complete_json([
                {"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": prompts.plan_prompt(user_request)},
            ])
            intent = data.get("intent", "generate_mcqs")
            if intent not in prompts.SUPPORTED_INTENTS:
                intent = "generate_mcqs"
            diff = data.get("difficulty")
            if diff not in ("easy", "medium", "hard"):
                diff = None
            num = data.get("num_items")
            num = int(num) if isinstance(num, (int, float)) else None
            topic = data.get("topic") or None
            return Plan(intent=intent, num_items=num, difficulty=diff, topic=topic)
        except (LLMError, Exception) as e:
            self._log(f"planner fallback to generate_mcqs: {e}")
            return Plan()
