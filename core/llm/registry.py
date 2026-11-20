"""
Provider selection. `get_llm()` returns a cached LLMProvider chosen from config.

Resolution order:
  1. If LLM_PROVIDER is explicit (groq/xai/openai/gemini/anthropic), use it.
  2. If "auto" (default), reproduce the legacy behaviour exactly:
       key starts with "gsk_"  -> Groq (with a current Groq model)
       otherwise               -> xAI (Grok) at the corrected api.x.ai host
This means an unchanged .env keeps working identically, minus the old
api.xai.com typo which previously broke every non-Groq key.
"""
from __future__ import annotations

import config
from core.llm.anthropic import AnthropicProvider
from core.llm.base import LLMError, LLMProvider
from core.llm.gemini import GeminiProvider
from core.llm.openai_compatible import OpenAICompatibleProvider

_cached: LLMProvider | None = None


# Default model per provider, used when auto-detection picks a provider that the
# configured model name clearly doesn't belong to (e.g. a llama model name left
# over in LLM_MODEL while the only key available is an Anthropic one). Sourced
# from config so there is exactly one table of defaults.
_DEFAULT_MODELS = config.PROVIDER_DEFAULT_MODELS

#: Substring that should appear in a model name for each provider. Used only to
#: decide whether the configured LLM_MODEL is plausible for the chosen provider.
_MODEL_HINTS = {
    "groq": ("llama", "mixtral", "gemma", "qwen", "deepseek", "kimi", "compound", "allam", "gpt-oss"),
    "xai": ("grok",),
    "openai": ("gpt", "o1", "o3", "o4"),
    "gemini": ("gemini",),
    "anthropic": ("claude",),
}

#: Markers that disqualify a model for a provider even though a broad hint above
#: would match. "openai/gpt-oss-20b" contains "gpt" but is served by Groq, not
#: OpenAI; this table is why provider selection can't be a loose substring test.
_FOREIGN_MARKERS = {
    "openai": ("gpt-oss", "llama", "mixtral", "gemma", "qwen", "deepseek", "kimi", "grok"),
    "groq": ("grok", "claude", "gemini"),
    "xai": ("gpt-oss", "llama", "claude", "gemini"),
    "gemini": ("gpt-oss", "llama", "claude", "grok"),
    "anthropic": ("gpt-oss", "llama", "gemini", "grok"),
}

# Retired Groq models map to the current default. Active GPT-OSS models must
# remain selectable; a rate limit is not a model deprecation.
_DEPRECATED_MODELS = {
    # Compound systems were decommissioned on 2026-09-21:
    # https://console.groq.com/docs/changelog/compound
    "groq/compound",
    "groq/compound-mini",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
}


def _belongs(provider_id: str, model: str) -> bool:
    """True when `model` plausibly belongs to `provider_id`."""
    if not model:
        return False
    lowered = model.lower()
    if any(marker in lowered for marker in _FOREIGN_MARKERS.get(provider_id, ())):
        return False
    return any(h in lowered for h in _MODEL_HINTS.get(provider_id, ()))


def _model_for(provider_id: str, model: str) -> str:
    """Keep `model` when it plausibly belongs to `provider_id`, else the default."""
    if provider_id == "groq" and model in _DEPRECATED_MODELS:
        return _DEFAULT_MODELS.get(provider_id, model)
    if _belongs(provider_id, model):
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
        # Unrecognised prefix: preserve the legacy default of treating it as xAI,
        # but still migrate a model name that obviously belongs elsewhere.
        return "xai", _model_for("xai", model)
    # No unified key at all -- fall back to any provider-specific key present.
    for provider_id, key in (
        ("anthropic", config.ANTHROPIC_API_KEY),
        ("gemini", config.GEMINI_API_KEY),
        ("openai", config.OPENAI_API_KEY),
    ):
        if key:
            return provider_id, _model_for(provider_id, model)
    return "xai", _model_for("xai", model)


def validate_provider_model(provider_id: str, model: str) -> None:
    """Raise LLMError when `model` cannot belong to `provider_id`.

    Every provider has its own default, so `_model_for` should already have
    replaced an implausible model. This is the fail-fast guard for the cases it
    cannot repair (unknown provider, missing default), so a misconfiguration
    stops boot with one clear line instead of failing per-request later with a
    provider 404 or "invalid model".
    """
    default = _DEFAULT_MODELS.get(provider_id, "")
    if not default:
        raise LLMError(
            f"No default model configured for LLM_PROVIDER '{provider_id}'. "
            f"Set LLM_MODEL explicitly or add a default to config.PROVIDER_DEFAULT_MODELS."
        )
    if model and _belongs(provider_id, model):
        return
    raise LLMError(
        f"Resolved model '{model}' does not belong to provider '{provider_id}'. "
        f"Set LLM_MODEL to a {provider_id} model (default: '{default}')."
    )


def build_llm() -> LLMProvider:
    """Construct a provider from current config (uncached).

    Every provider gets its own default model, and a configured model is only
    kept when it plausibly belongs to the resolved provider. The final pair is
    validated before construction so a mismatched provider/model fails with one
    clear error instead of a confusing provider-side 404 later.
    """
    provider = config.LLM_PROVIDER
    key = config.LLM_API_KEY
    model = config.LLM_MODEL
    timeout = config.LLM_TIMEOUT
    temp = config.LLM_TEMPERATURE

    if provider == "auto":
        provider, model = _resolve_auto(key, model)
    else:
        # Explicit selection applies the same migration rules as auto: a stale
        # or foreign model name is replaced with this provider's default.
        model = _model_for(provider, model)

    validate_provider_model(provider, model)

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
