"""
Adapter for any provider exposing the OpenAI /chat/completions shape:
Groq, xAI (Grok), OpenAI, Together, etc. They differ only by base URL,
so one class covers all of them.

NOTE: fixes the long-standing typo in the legacy code, which used
`https://api.xai.com` (nonexistent host). The correct xAI host is
`https://api.x.ai`.
"""
from __future__ import annotations

import requests

from core.llm.base import LLMProvider, LLMError

# Canonical base URLs per provider id.
BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "xai": "https://api.x.ai/v1",        # fixed: was api.xai.com in legacy code
    "openai": "https://api.openai.com/v1",
    "together": "https://api.together.xyz/v1",
}


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, api_key, model, timeout=45, temperature=0.3,
                 base_url: str | None = None, provider_id: str = "groq"):
        super().__init__(api_key, model, timeout, temperature)
        self.provider_id = provider_id
        self.base_url = (base_url or BASE_URLS.get(provider_id, BASE_URLS["groq"])).rstrip("/")

    def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False) -> str:
        if not self.api_key:
            raise LLMError("API Key is missing. Please set LLM_API_KEY (or GROK_API_KEY) in your .env file.")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if json_mode:
            # Supported by Groq, OpenAI and xAI; harmless hint elsewhere.
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(f"{self.base_url}/chat/completions",
                                 headers=headers, json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise LLMError("The request to the AI API timed out. Please try again with a shorter text.")
        except requests.exceptions.RequestException as re:
            raise LLMError(f"Network error when connecting to the AI API: {re}")

        if resp.status_code == 401:
            raise LLMError("Invalid API key! Please check your LLM_API_KEY / GROK_API_KEY.")
        if resp.status_code == 429:
            raise LLMError("API rate limit exceeded! Please wait a moment and try again.")
        if resp.status_code != 200:
            raise LLMError(f"API request failed with status code {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"Unexpected AI API response shape: {e}")
