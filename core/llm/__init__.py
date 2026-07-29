"""LLM provider abstraction. Import `get_llm()` to obtain the configured client."""
from core.llm.registry import get_llm, reset_llm
from core.llm.base import LLMProvider, LLMError

__all__ = ["get_llm", "reset_llm", "LLMProvider", "LLMError"]
