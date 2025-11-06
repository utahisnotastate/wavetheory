"""
Chatbot backend for Wave Theory Chatbot.

Uses Google Gemini exclusively via GOOGLE_API_KEY.
Default model can be set with GEMINI_MODEL (defaults to gemini-2.5-pro).
"""

from __future__ import annotations

import os
from typing import Callable


class GeminiChat:
    """Thin wrapper around google-generativeai to generate text responses."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro") -> None:
        import google.generativeai as genai

        self._genai = genai
        self._genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self._model.generate_content(prompt, **kwargs)
            # The SDK returns candidates; prefer the first text part
            if hasattr(response, "text") and response.text:
                return response.text
            # Fallback: serialize
            return str(response)
        except Exception as exc:  # noqa: BLE001
            return f"[Gemini error] {exc}"


def get_chatbot() -> Callable[[str], str]:
    """
    Return a callable that accepts a prompt and returns a response string.

    Uses Google Gemini via GOOGLE_API_KEY (model set by GEMINI_MODEL or default).
    """
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        def _missing_key(_: str) -> str:
            return "Chat backend not configured. Please set GOOGLE_API_KEY for Gemini."
        return _missing_key

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
    gem = GeminiChat(api_key=google_api_key, model_name=model_name)

    def _gemini_generate(prompt: str) -> str:
        return gem.generate(prompt)

    return _gemini_generate


