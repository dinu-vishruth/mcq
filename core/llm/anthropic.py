"""
Anthropic Claude adapter via the Messages REST endpoint.

Plain HTTP, no SDK dependency. Anthropic requires the system prompt as a
top-level `system` field (not a message) and an explicit `max_tokens`.
JSON mode is achieved by instruction + prefill rather than a response_format
flag, so complete_json relies on the tolerant parser in the base class.
"""
from __future__ import annotations

import requests

from core.llm.base import LLMProvider, LLMError

URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(LLMProvider):
    def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False) -> str:
        if not self.api_key:
            raise LLMError("API Key is missing. Please set ANTHROPIC_API_KEY in your .env file.")

        system_parts, turns = [], []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                turns.append({"role": "assistant" if role == "assistant" else "user",
                              "content": content})

        if json_mode:
            system_parts.append("You must respond with a single valid JSON value and nothing else.")

        payload = {
            "model": self.model,
            "messages": turns,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(URL, headers=headers, json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise LLMError("The request to the AI API timed out. Please try again with a shorter text.")
        except requests.exceptions.RequestException as re:
            raise LLMError(f"Network error when connecting to the AI API: {re}")

        if resp.status_code == 401:
            raise LLMError("Invalid API key! Please check your ANTHROPIC_API_KEY.")
        if resp.status_code == 429:
            raise LLMError("API rate limit exceeded! Please wait a moment and try again.")
        if resp.status_code != 200:
            raise LLMError(f"API request failed with status code {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
            return "".join(block.get("text", "") for block in data["content"]).strip()
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"Unexpected Anthropic response shape: {e}")
