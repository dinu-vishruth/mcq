"""LLM provider abstraction. Import `get_llm()` to obtain the configured client."""
from core.llm.base import LLMError, LLMProvider
from core.llm.registry import get_llm, reset_llm

__all__ = ["get_llm", "reset_llm", "LLMProvider", "LLMError"]
