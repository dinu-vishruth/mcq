"""Check local AI configuration without spending an API request or exposing keys."""
import logging

import config


def generation_available() -> bool:
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
    if not generation_available():
        logging.getLogger(__name__).error(
            "Generation unavailable. Check credentials for LLM_PROVIDER=%s. %s",
            config.LLM_PROVIDER, config.missing_key_message(),
        )
        raise HTTPException(status_code=503, detail=config.USER_FACING_AI_UNAVAILABLE)
