"""
base.py — Abstract LLM base class (Step 4d)

Imports nothing from kashif_core — hard isolation rule.

All provider adapters inherit BaseLLM and implement complete().
fe_agent.py only ever holds a BaseLLM reference — it never imports
a provider SDK directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Minimal contract every LLM adapter must satisfy.

    Parameters
    ----------
    model : str
        Model identifier string (provider-specific).
    temperature : float
        Sampling temperature. Lower = more deterministic. Default 0.2.
    max_tokens : int
        Maximum tokens to generate per call. Default 4096.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """
        Send *prompt* to the LLM and return the response text.

        Parameters
        ----------
        prompt : str
            Full prompt string (fe_agent builds this).

        Returns
        -------
        str
            Raw text response from the model.

        Raises
        ------
        LLMError
            On any provider-side failure (auth, rate limit, timeout, etc.).
        """

    def complete_with_system(self, system: str, user: str) -> str:
        """
        Send a system + user message pair and return the response text.

        Providers that support a dedicated system role (Groq, Anthropic) override
        this to use it properly. The default implementation concatenates them so
        adapters that haven't overridden it still work.

        Parameters
        ----------
        system : str
            System-role content — rules, constraints, output format.
        user : str
            User-role content — data, history, task.

        Raises
        ------
        LLMError
            On any provider-side failure.
        """
        return self.complete(f"{system}\n\n{user}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model!r}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )


class LLMError(Exception):
    """
    Raised by adapters when the provider returns an unrecoverable error.
    fe_agent.py catches this and logs it as a round failure.
    """
