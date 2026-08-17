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


# Default model per provider, used when auto-detection picks a provider that the
# configured model name clearly doesn't belong to (e.g. a llama model name left
# over in LLM_MODEL while the only key available is an Anthropic one).
_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "xai": "grok-2-1212",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "anthropic": "claude-sonnet-5",
}

#: Substring that should appear in a model name for each provider. Used only to
#: decide whether the configured LLM_MODEL is plausible for the chosen provider.
_MODEL_HINTS = {
    "groq": ("llama", "mixtral", "gemma", "qwen", "deepseek", "kimi", "gpt-oss", "compound", "allam"),
    "xai": ("grok",),
    "openai": ("gpt", "o1", "o3", "o4"),
    "gemini": ("gemini",),
    "anthropic": ("claude",),
}

# Decommissioned/deprecated models from providers that should automatically map to the active default
_DEPRECATED_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
}


def _model_for(provider_id: str, model: str) -> str:
    """Keep `model` when it plausibly belongs to `provider_id`, else the default."""
    if model in _DEPRECATED_MODELS:
        return _DEFAULT_MODELS.get(provider_id, model)
    hints = _MODEL_HINTS.get(provider_id, ())
    if model and any(h in model.lower() for h in hints):
        return model
    return _DEFAULT_MODELS.get(provider_id, model)


def _resolve_auto(api_key: str, model: str) -> tuple[str, str]:
    """Return (provider_id, model) for LLM_PROVIDER=auto.

    Key-prefix detection first (the historical behaviour), then a fallback to
    whichever per-provider key is actually configured. Without that fallback a
    deployment holding only, say, ANTHROPIC_API_KEY resolved to xAI with an empty
    key and failed at request time with a misleading "invalid key" error.
    """
    if api_key.startswith("gsk_"):
        return "groq", _model_for("groq", model)
    if api_key.startswith("xai-"):
        return "xai", _model_for("xai", model)
    if api_key.startswith("sk-ant-"):
        return "anthropic", _model_for("anthropic", model)
    if api_key.startswith("sk-"):
        return "openai", _model_for("openai", model)
    if api_key:
        # Unrecognised prefix: preserve the legacy default of treating it as xAI.
        return "xai", model
    # No unified key at all -- fall back to any provider-specific key present.
    for provider_id, key in (
        ("anthropic", config.ANTHROPIC_API_KEY),
        ("gemini", config.GEMINI_API_KEY),
        ("openai", config.OPENAI_API_KEY),
    ):
        if key:
            return provider_id, _model_for(provider_id, model)
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
