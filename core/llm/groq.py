"""
groq.py — Groq LLM adapter (Step 4d) — DEFAULT provider

Uses the openai SDK pointed at Groq's base URL.
Imports nothing from kashif_core — hard isolation rule.

Usage (resolved by cli/main.py at runtime):
    llm = GroqLLM(model="llama-3.3-70b-versatile", api_key_env="GROQ_API_KEY")
    response = llm.complete(prompt)
"""

from __future__ import annotations

import os

from core.llm.base import BaseLLM, LLMError

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


class GroqLLM(BaseLLM):
    """
    LLM adapter for Groq using the OpenAI-compatible SDK.

    Parameters
    ----------
    model : str
        Groq model ID, e.g. 'llama-3.3-70b-versatile'.
    api_key_env : str
        Name of the environment variable that holds the Groq API key.
        Default: 'GROQ_API_KEY'.
    temperature : float
        Sampling temperature. Default 0.2.
    max_tokens : int
        Max tokens per completion. Default 4096.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key_env: str = "GROQ_API_KEY",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_key_env = api_key_env
        self._client = None  # lazy — built on first call

    def _get_client(self):
        """Lazy-init the OpenAI client pointed at Groq's base URL."""
        if self._client is not None:
            return self._client

        if OpenAI is None:
            raise LLMError("openai package is not installed. Run: uv add openai")

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise LLMError(
                f"Environment variable '{self.api_key_env}' is not set. "
                "Export your Groq API key before running Kashif."
            )

        self._client = OpenAI(api_key=api_key, base_url=_GROQ_BASE_URL)
        return self._client

    def complete(self, prompt: str) -> str:
        """Send *prompt* to Groq and return the response text."""
        return self._chat([{"role": "user", "content": prompt}])

    def complete_with_system(self, system: str, user: str) -> str:
        """Send a system + user message pair — uses Groq's native system role."""
        return self._chat([
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ])

    def _chat(self, messages: list) -> str:
        """Internal: call the Groq chat completions endpoint."""
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if not response.choices:
                raise LLMError("Groq returned an empty choices list.")
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(f"Groq API error: {exc}") from exc
