"""
Extensible content generation (Phase 7).

All content types (flashcards, summaries, interview/coding questions, topic
explanations) share ONE retrieval-grounded spine:

    ingest -> retrieve (MMR + spread) -> build context -> prompt -> JSON

A generator is just (prompt_fn, result_key). Registering a new content type is a
two-line addition to GENERATORS below plus a prompt in core/prompts/content.py --
no pipeline surgery, satisfying the "design for future features without major
refactoring" requirement. The PlannerAgent's intents map directly onto these
keys, so a free-form request routes here automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import config
from core.agents.retriever import RetrieverAgent
from core.agents.context_builder import ContextBuilderAgent
from core.llm import get_llm, LLMError
from core.services.ingestion_service import ingest_document
from core.prompts import content as prompts


@dataclass
class Generator:
    prompt_fn: Callable          # (context, num_items, topic) -> str  OR (context, topic) -> str
    result_key: str              # top-level JSON key holding the items
    needs_count: bool = True     # whether the prompt takes num_items


# intent -> Generator. Extend this map to add a content type.
GENERATORS: dict[str, Generator] = {
    "generate_flashcards": Generator(prompts.flashcards_prompt, "flashcards"),
    "summarize":           Generator(lambda c, n, t: prompts.summary_prompt(c, t), "summary", needs_count=False),
    "interview_questions": Generator(prompts.interview_prompt, "questions"),
    "coding_questions":    Generator(prompts.coding_prompt, "problems"),
    "explain_topic":       Generator(lambda c, n, t: prompts.explain_topic_prompt(c, t), "explanation", needs_count=False),
}


def supported_intents() -> list[str]:
    return list(GENERATORS.keys())


def generate_content(intent: str, text: str, *, num_items: int = 10, topic: str | None = None,
                     owner: str = "", title: str = "") -> dict:
    """Return {"intent", "items", "document_id"} for any registered content type.

    Raises ValueError for an unknown intent so callers can fall back explicitly.
    """
    gen = GENERATORS.get(intent)
    if gen is None:
        raise ValueError(f"Unsupported content intent: {intent}")

    ingest = ingest_document(text, owner=owner, title=title or "uploaded document", source_type="paste")
    document_id = ingest["document_id"]

    top_k = max(config.RETRIEVAL_TOP_K, min(num_items * 2, 40))
    query = f"Key material about {topic}" if topic else "The most important concepts and details"
    hits = RetrieverAgent().run(document_id, query, top_k, spread=True)
    context = ContextBuilderAgent().run(hits)

    if gen.needs_count:
        user = gen.prompt_fn(context, num_items, topic)
    else:
        user = gen.prompt_fn(context, num_items, topic)  # lambdas ignore num_items

    try:
        data = get_llm().complete_json([
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": user},
        ])
        items = data.get(gen.result_key) if isinstance(data, dict) else data
    except LLMError as e:
        raise ValueError(str(e))

    return {"intent": intent, "items": items, "document_id": document_id}
