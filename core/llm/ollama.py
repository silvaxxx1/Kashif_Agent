"""
ollama.py — Ollama local LLM adapter

Uses the openai SDK pointed at Ollama's OpenAI-compatible endpoint.
No API key required — Ollama runs locally.
Imports nothing from kashif_core — hard isolation rule.

Usage (resolved by cli/main.py at runtime):
    llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434/v1")
    response = llm.complete(prompt)

Ollama must be running before Kashif is invoked:
    ollama serve
    ollama pull llama3.2
"""

from __future__ import annotations

from core.llm.base import BaseLLM, LLMError

_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


class OllamaLLM(BaseLLM):
    """
    LLM adapter for Ollama local inference using the OpenAI-compatible API.

    No API key is required — Ollama runs entirely on-device.
    The base_url can be overridden for remote or non-default Ollama deployments.

    Parameters
    ----------
    model : str
        Ollama model name, e.g. 'llama3.2', 'mistral', 'codellama'.
        The model must already be pulled: ``ollama pull <model>``.
    base_url : str
        Ollama server URL. Default: 'http://localhost:11434/v1'.
    temperature : float
        Sampling temperature. Default 0.2.
    max_tokens : int
        Max tokens per completion. Default 4096.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = _OLLAMA_DEFAULT_BASE_URL,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.base_url = base_url
        self._client = None  # lazy — built on first call

    def _get_client(self):
        """Lazy-init the OpenAI client pointed at Ollama's base URL."""
        if self._client is not None:
            return self._client

        if OpenAI is None:
            raise LLMError("openai package is not installed. Run: uv add openai")

        # Ollama's OpenAI-compatible endpoint accepts any non-empty string as the key
        self._client = OpenAI(api_key="ollama", base_url=self.base_url)
        return self._client

    def complete(self, prompt: str) -> str:
        """Send *prompt* to Ollama and return the response text."""
        return self._chat([{"role": "user", "content": prompt}])

    def complete_with_system(self, system: str, user: str) -> str:
        """Send a system + user message pair — uses Ollama's native system role."""
        return self._chat([
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ])

    def _chat(self, messages: list) -> str:
        """Internal: call the Ollama chat completions endpoint."""
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if not response.choices:
                raise LLMError("Ollama returned an empty choices list.")
            return response.choices[0].message.content or ""
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama API error: {exc}") from exc

    def __repr__(self) -> str:
        return (
            f"OllamaLLM("
            f"model={self.model!r}, "
            f"base_url={self.base_url!r}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )
