"""
Chatbot backends for Wave Theory Chatbot.

Provides a Gemini-backed generator when GOOGLE_API_KEY is available,
falling back to a small Hugging Face text-generation pipeline otherwise.
"""

from __future__ import annotations

import os
from typing import Callable, Optional


class GeminiChat:
    """Thin wrapper around google-generativeai to generate text responses."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash") -> None:
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


def _build_hf_generator() -> Optional[Callable[[str], str]]:
    """Create a lightweight Hugging Face text generator without importing transformers.pipelines.

    We avoid importing `transformers.pipelines` entirely because it pulls in vision utils
    that depend on torchvision and can crash in CPU-only or mismatched Torch/Torchvision setups.
    """
    try:
        # Import only the pieces we need; avoid `transformers.pipelines`
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        model_name = os.environ.get("HF_CHAT_MODEL", "microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Ensure pad token is set to eos if missing (common for small dialog models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        @torch.inference_mode()
        def _generate(prompt: str) -> str:
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                output_ids = model.generate(
                    **inputs,
                    max_length=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                return text
            except Exception as exc:  # noqa: BLE001
                return f"[HF error] {exc}"

        return _generate
    except Exception:
        return None


def get_chatbot() -> Callable[[str], str]:
    """
    Return a callable that accepts a prompt and returns a response string.

    Order of preference:
      1) Gemini via GOOGLE_API_KEY (model set by GEMINI_MODEL or default)
      2) Hugging Face small local fallback using AutoModelForCausalLM
    """
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        gem = GeminiChat(api_key=google_api_key, model_name=model_name)

        def _gemini_generate(prompt: str) -> str:
            return gem.generate(prompt)

        return _gemini_generate

    hf = _build_hf_generator()
    if hf is not None:
        return hf

    def _noop(prompt: str) -> str:
        return "Chat backend not configured. Set GOOGLE_API_KEY or install transformers models."

    return _noop


