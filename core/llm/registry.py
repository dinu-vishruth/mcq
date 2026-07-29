"""
Provider selection. `get_llm()` returns a cached LLMProvider chosen from config.

Resolution order:
  1. If LLM_PROVIDER is explicit (groq/xai/openai/gemini/anthropic), use it.
  2. If "auto" (default), reproduce the legacy behaviour exactly:
       key starts with "gsk_"  -> Groq  (llama-3.3-70b-versatile unless a
                                          llama model is already configured)
       otherwise               -> xAI (Grok) at the corrected api.x.ai host
This means an unchanged .env keeps working identically, minus the old
api.xai.com typo which previously broke every non-Groq key.
"""
from __future__ import annotations

import config
from core.llm.base import LLMProvider, LLMError
from core.llm.openai_compatible import OpenAICompatibleProvider
from core.llm.gemini import GeminiProvider
from core.llm.anthropic import AnthropicProvider

_cached: LLMProvider | None = None


def _resolve_auto(api_key: str, model: str) -> tuple[str, str]:
    """Return (provider_id, model) using the legacy prefix heuristic."""
    if api_key.startswith("gsk_"):
        chosen = model if "llama" in model.lower() else "llama-3.3-70b-versatile"
        return "groq", chosen
    return "xai", model


def build_llm() -> LLMProvider:
    """Construct a provider from current config (uncached)."""
    provider = config.LLM_PROVIDER
    key = config.LLM_API_KEY
    model = config.LLM_MODEL
    timeout = config.LLM_TIMEOUT
    temp = config.LLM_TEMPERATURE

    if provider == "auto":
        provider, model = _resolve_auto(key, model)

    if provider in ("groq", "xai", "openai", "together"):
        return OpenAICompatibleProvider(key, model, timeout, temp, provider_id=provider)
    if provider == "gemini":
        return GeminiProvider(config.GEMINI_API_KEY or key, model, timeout, temp)
    if provider == "anthropic":
        return AnthropicProvider(config.ANTHROPIC_API_KEY or key, model, timeout, temp)

    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. "
                   "Use one of: auto, groq, xai, openai, gemini, anthropic.")


def get_llm() -> LLMProvider:
    """Return the process-wide cached provider, building it on first use."""
    global _cached
    if _cached is None:
        _cached = build_llm()
    return _cached


def reset_llm() -> None:
    """Drop the cache (used by tests when they patch config)."""
    global _cached
    _cached = None
