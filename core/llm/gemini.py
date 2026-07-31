"""
Google Gemini adapter via the REST generateContent endpoint.

Uses plain HTTP (no extra SDK dependency, keeps the Vercel bundle small).
Gemini's schema differs from OpenAI's: system prompt goes in
`system_instruction`, turns go in `contents` with role "user"/"model".
"""
from __future__ import annotations

import requests

from core.llm.base import LLMProvider, LLMError, RateLimitError

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(LLMProvider):
    def _complete_once(self, messages, *, temperature=None, max_tokens=None, json_mode=False) -> str:
        if not self.api_key:
            raise LLMError("API Key is missing. Please set GEMINI_API_KEY in your .env file.")

        system_parts, contents = [], []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                gem_role = "model" if role == "assistant" else "user"
                contents.append({"role": gem_role, "parts": [{"text": content}]})

        gen_config = {"temperature": self.temperature if temperature is None else temperature}
        if max_tokens:
            gen_config["maxOutputTokens"] = max_tokens
        if json_mode:
            gen_config["responseMimeType"] = "application/json"

        payload = {"contents": contents, "generationConfig": gen_config}
        if system_parts:
            payload["system_instruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        url = f"{BASE_URL}/{self.model}:generateContent?key={self.api_key}"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise LLMError("The request to the AI API timed out. Please try again with a shorter text.")
        except requests.exceptions.RequestException as re:
            raise LLMError(f"Network error when connecting to the AI API: {re}")

        if resp.status_code == 401 or resp.status_code == 403:
            raise LLMError("Invalid API key! Please check your GEMINI_API_KEY.")
        if resp.status_code == 429:
            raise RateLimitError("API rate limit exceeded! Please wait a moment and try again.",
                                 retry_after=self._parse_retry_after(resp))
        if resp.status_code != 200:
            raise LLMError(f"API request failed with status code {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"Unexpected Gemini response shape: {e}")
