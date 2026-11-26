"""Check local AI configuration without spending an API request or exposing keys."""
import logging

import config


def generation_available() -> bool:
    from core.services.admin_control import enabled
    if not enabled("generation"):
        return False
    if not config.llm_key_present():
        return False
    try:
        from core.llm import get_llm
        return bool(get_llm().api_key)
    except Exception:
        logging.getLogger(__name__).exception("AI provider setup failed")
        return False


def require_generation() -> None:
    from fastapi import HTTPException

    from core.services.admin_control import enabled
    if not enabled("generation"):
        raise HTTPException(503, "Quiz generation is currently paused by the site owner. Please try again later.")
    if not generation_available():
        logging.getLogger(__name__).error(
            "Generation unavailable. Check credentials for LLM_PROVIDER=%s. %s",
            config.LLM_PROVIDER, config.missing_key_message(),
        )
        raise HTTPException(status_code=503, detail=config.USER_FACING_AI_UNAVAILABLE)
