"""
anthropic.py — Anthropic LLM adapter (Step 4d) — secondary provider

Uses the anthropic SDK directly.
Imports nothing from kashif_core — hard isolation rule.

Usage (resolved by cli/main.py at runtime):
    llm = AnthropicLLM(model="claude-sonnet-4-6", api_key_env="ANTHROPIC_API_KEY")
    response = llm.complete(prompt)
"""

from __future__ import annotations

import os

from core.llm.base import BaseLLM, LLMError

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


class AnthropicLLM(BaseLLM):
    """
    LLM adapter for Anthropic Claude models.

    Parameters
    ----------
    model : str
        Anthropic model ID, e.g. 'claude-sonnet-4-6'.
    api_key_env : str
        Name of the environment variable that holds the Anthropic API key.
        Default: 'ANTHROPIC_API_KEY'.
    temperature : float
        Sampling temperature. Default 0.2.
    max_tokens : int
        Max tokens per completion. Default 4096.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key_env: str = "ANTHROPIC_API_KEY",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_key_env = api_key_env
        self._client = None  # lazy — built on first call

    def _get_client(self):
        """Lazy-init the Anthropic client."""
        if self._client is not None:
            return self._client

        if anthropic is None:
            raise LLMError("anthropic package is not installed. Run: uv add anthropic")

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise LLMError(
                f"Environment variable '{self.api_key_env}' is not set. "
                "Export your Anthropic API key before running Kashif."
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def complete(self, prompt: str) -> str:
        """Send *prompt* to Anthropic and return the response text."""
        client = self._get_client()
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if not message.content:
                raise LLMError("Anthropic returned an empty content list.")
            block = message.content[0]
            if not hasattr(block, "text"):
                raise LLMError(f"Anthropic returned unexpected content block type: {type(block).__name__}")
            return block.text or ""
        except Exception as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc
