"""
test_llm.py — Step 4d tests for core/llm/

Strategy:
  - No real API calls — providers are tested with mocked SDK clients
  - Covers: BaseLLM contract, GroqLLM, AnthropicLLM
    - lazy client init
    - missing API key raises LLMError
    - missing SDK package raises LLMError
    - provider SDK error wrapped in LLMError
    - complete() returns string
    - repr
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from core.llm.base import BaseLLM, LLMError
from core.llm.groq import GroqLLM
from core.llm.anthropic import AnthropicLLM


# ---------------------------------------------------------------------------
# BaseLLM — abstract contract
# ---------------------------------------------------------------------------

class TestBaseLLM:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseLLM(model="x")  # type: ignore

    def test_concrete_subclass_works(self):
        class DummyLLM(BaseLLM):
            def complete(self, prompt: str) -> str:
                return f"echo: {prompt}"

        llm = DummyLLM(model="dummy", temperature=0.5, max_tokens=100)
        assert llm.model == "dummy"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 100
        assert llm.complete("hello") == "echo: hello"

    def test_repr(self):
        class DummyLLM(BaseLLM):
            def complete(self, prompt: str) -> str:
                return ""

        llm = DummyLLM(model="test-model", temperature=0.1, max_tokens=512)
        r = repr(llm)
        assert "DummyLLM" in r
        assert "test-model" in r
        assert "0.1" in r
        assert "512" in r

    def test_llm_error_is_exception(self):
        err = LLMError("something went wrong")
        assert isinstance(err, Exception)
        assert "something went wrong" in str(err)


# ---------------------------------------------------------------------------
# GroqLLM
# ---------------------------------------------------------------------------

class TestGroqLLM:
    def test_default_params(self):
        llm = GroqLLM()
        assert llm.model == "llama-3.3-70b-versatile"
        assert llm.api_key_env == "GROQ_API_KEY"
        assert llm.temperature == 0.2
        assert llm.max_tokens == 4096

    def test_custom_params(self):
        llm = GroqLLM(model="llama3-8b-8192", temperature=0.5, max_tokens=1024)
        assert llm.model == "llama3-8b-8192"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1024

    def test_is_base_llm(self):
        assert isinstance(GroqLLM(), BaseLLM)

    def test_missing_api_key_raises_llm_error(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        llm = GroqLLM(api_key_env="GROQ_API_KEY")
        with pytest.raises(LLMError, match="GROQ_API_KEY"):
            llm.complete("test")

    def test_missing_openai_package_raises_llm_error(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        llm = GroqLLM()
        # Simulate openai not installed
        with patch.dict("sys.modules", {"openai": None}):
            llm._client = None  # reset lazy cache
            with pytest.raises((LLMError, ImportError)):
                llm.complete("test")

    def test_complete_returns_string(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")

        # Build mock response object matching openai SDK shape
        mock_message = MagicMock()
        mock_message.content = "def engineer_features(df): return df"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        llm = GroqLLM()
        llm._client = mock_client

        result = llm.complete("write some feature engineering code")
        assert isinstance(result, str)
        assert "engineer_features" in result

    def test_complete_calls_correct_model(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")

        mock_message = MagicMock()
        mock_message.content = "response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        llm = GroqLLM(model="llama3-8b-8192", temperature=0.1, max_tokens=512)
        llm._client = mock_client

        llm.complete("prompt")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "llama3-8b-8192"
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 512

    def test_provider_error_wrapped_in_llm_error(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("rate limit")

        llm = GroqLLM()
        llm._client = mock_client

        with pytest.raises(LLMError, match="Groq API error"):
            llm.complete("test")

    def test_empty_response_returns_empty_string(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")

        mock_message = MagicMock()
        mock_message.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        llm = GroqLLM()
        llm._client = mock_client

        assert llm.complete("prompt") == ""

    def test_client_is_lazy(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        llm = GroqLLM()
        assert llm._client is None  # not created until first call

    def test_client_cached_after_first_call(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")

        mock_message = MagicMock()
        mock_message.content = "ok"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("core.llm.groq.OpenAI", return_value=mock_client) as mock_ctor:
            llm = GroqLLM()
            llm.complete("first")
            llm.complete("second")
            # OpenAI() constructor should only be called once
            assert mock_ctor.call_count == 1

    def test_repr(self):
        llm = GroqLLM(model="llama3-8b-8192")
        assert "GroqLLM" in repr(llm)
        assert "llama3-8b-8192" in repr(llm)


# ---------------------------------------------------------------------------
# AnthropicLLM
# ---------------------------------------------------------------------------

class TestAnthropicLLM:
    def test_default_params(self):
        llm = AnthropicLLM()
        assert llm.model == "claude-sonnet-4-6"
        assert llm.api_key_env == "ANTHROPIC_API_KEY"
        assert llm.temperature == 0.2
        assert llm.max_tokens == 4096

    def test_custom_params(self):
        llm = AnthropicLLM(model="claude-haiku-4-5-20251001", temperature=0.0, max_tokens=2048)
        assert llm.model == "claude-haiku-4-5-20251001"
        assert llm.temperature == 0.0
        assert llm.max_tokens == 2048

    def test_is_base_llm(self):
        assert isinstance(AnthropicLLM(), BaseLLM)

    def test_missing_api_key_raises_llm_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        llm = AnthropicLLM(api_key_env="ANTHROPIC_API_KEY")
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            llm.complete("test")

    def test_complete_returns_string(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

        mock_content_block = MagicMock()
        mock_content_block.text = "def engineer_features(df): return df"
        mock_response = MagicMock()
        mock_response.content = [mock_content_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        llm = AnthropicLLM()
        llm._client = mock_client

        result = llm.complete("write feature engineering code")
        assert isinstance(result, str)
        assert "engineer_features" in result

    def test_complete_calls_correct_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

        mock_content_block = MagicMock()
        mock_content_block.text = "response"
        mock_response = MagicMock()
        mock_response.content = [mock_content_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        llm = AnthropicLLM(model="claude-haiku-4-5-20251001", temperature=0.0, max_tokens=1024)
        llm._client = mock_client

        llm.complete("prompt")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 1024

    def test_provider_error_wrapped_in_llm_error(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("overloaded")

        llm = AnthropicLLM()
        llm._client = mock_client

        with pytest.raises(LLMError, match="Anthropic API error"):
            llm.complete("test")

    def test_empty_content_raises_llm_error(self, monkeypatch):
        """Empty content list must raise LLMError — not silently return empty string."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

        mock_response = MagicMock()
        mock_response.content = []

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        llm = AnthropicLLM()
        llm._client = mock_client

        with pytest.raises(LLMError, match="empty content"):
            llm.complete("prompt")

    def test_client_is_lazy(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        llm = AnthropicLLM()
        assert llm._client is None

    def test_client_cached_after_first_call(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

        mock_content_block = MagicMock()
        mock_content_block.text = "ok"
        mock_response = MagicMock()
        mock_response.content = [mock_content_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("core.llm.anthropic.anthropic") as mock_anthropic_module:
            mock_anthropic_module.Anthropic.return_value = mock_client
            llm = AnthropicLLM()
            llm.complete("first")
            llm.complete("second")
            assert mock_anthropic_module.Anthropic.call_count == 1

    def test_repr(self):
        llm = AnthropicLLM(model="claude-sonnet-4-6")
        assert "AnthropicLLM" in repr(llm)
        assert "claude-sonnet-4-6" in repr(llm)


# ---------------------------------------------------------------------------
# complete_with_system — regression tests for Fix C
# ---------------------------------------------------------------------------

class TestCompleteWithSystem:
    """Verify system/user split is wired correctly in both providers."""

    def _mock_groq(self, monkeypatch, response_text: str) -> GroqLLM:
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        mock_message = MagicMock()
        mock_message.content = response_text
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        llm = GroqLLM()
        llm._client = mock_client
        return llm

    def _mock_anthropic(self, monkeypatch, response_text: str) -> AnthropicLLM:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        mock_block = MagicMock()
        mock_block.text = response_text
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        llm = AnthropicLLM()
        llm._client = mock_client
        return llm

    def test_groq_complete_with_system_returns_string(self, monkeypatch):
        llm = self._mock_groq(monkeypatch, "def engineer_features(df): return df")
        result = llm.complete_with_system("you are an expert", "write code")
        assert isinstance(result, str)
        assert "engineer_features" in result

    def test_groq_complete_with_system_sends_two_messages(self, monkeypatch):
        """System role must be a separate message, not concatenated into user."""
        llm = self._mock_groq(monkeypatch, "response")
        llm.complete_with_system(system="system instructions", user="user task")
        call_kwargs = llm._client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[0]["content"] == "system instructions"
        assert messages[1]["content"] == "user task"

    def test_groq_complete_single_message_still_works(self, monkeypatch):
        """complete() (single user message) must still function after refactor."""
        llm = self._mock_groq(monkeypatch, "response")
        llm.complete("just a user prompt")
        call_kwargs = llm._client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_anthropic_complete_with_system_sends_system_param(self, monkeypatch):
        """Anthropic system must be sent via the system= kwarg, not in messages."""
        llm = self._mock_anthropic(monkeypatch, "response")
        llm.complete_with_system(system="rules", user="task")
        call_kwargs = llm._client.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "rules"
        assert call_kwargs["messages"] == [{"role": "user", "content": "task"}]

    def test_anthropic_complete_without_system_omits_system_param(self, monkeypatch):
        """complete() must not pass system= to Anthropic (avoids empty-string errors)."""
        llm = self._mock_anthropic(monkeypatch, "response")
        llm.complete("prompt only")
        call_kwargs = llm._client.messages.create.call_args.kwargs
        assert "system" not in call_kwargs

    def test_base_llm_default_concatenates(self):
        """BaseLLM default complete_with_system falls back to concatenation."""
        class DummyLLM(BaseLLM):
            def complete(self, prompt: str) -> str:
                return f"received: {prompt}"

        llm = DummyLLM(model="dummy")
        result = llm.complete_with_system("SYSTEM", "USER")
        assert "SYSTEM" in result
        assert "USER" in result


# ---------------------------------------------------------------------------
# _call_llm retry and hard-stop — regression tests for Fix A + B
# ---------------------------------------------------------------------------

class TestCallLLMRetry:
    """Verify Fix A (hard stop on LLMError) and Fix B (retry on parse failure)."""

    def _make_agent(self, llm):
        from core.fe_agent import FEAgent
        return FEAgent(llm, config={"max_rounds": 1, "stall_rounds": 1})

    def test_llm_error_propagates_from_call_llm(self):
        """LLMError from the provider must propagate out of _call_llm."""
        from core.fe_agent import FEAgent
        from core.llm.base import LLMError

        mock = MagicMock(spec=BaseLLM)
        mock.complete_with_system.side_effect = LLMError("auth failed")
        agent = FEAgent(mock, config={})
        with pytest.raises(LLMError):
            agent._call_llm("some prompt")

    def test_parse_failure_returns_none_not_raises(self):
        """Parse failure (no def engineer_features) must return None, not raise."""
        from core.fe_agent import FEAgent

        mock = MagicMock(spec=BaseLLM)
        mock.complete_with_system.return_value = "sorry I cannot help"
        agent = FEAgent(mock, config={})
        result = agent._call_llm("prompt")
        assert result is None

    def test_retry_called_on_parse_failure(self):
        """complete_with_system must be called twice when first response is unparseable."""
        from core.fe_agent import FEAgent

        mock = MagicMock(spec=BaseLLM)
        mock.complete_with_system.return_value = "not a function"
        agent = FEAgent(mock, config={})
        agent._call_llm("prompt")
        assert mock.complete_with_system.call_count == 2

    def test_no_retry_when_first_parse_succeeds(self):
        """If first response parses correctly, only one API call is made."""
        from core.fe_agent import FEAgent

        mock = MagicMock(spec=BaseLLM)
        mock.complete_with_system.return_value = "def engineer_features(df):\n    return df"
        agent = FEAgent(mock, config={})
        result = agent._call_llm("prompt")
        assert result is not None
        assert mock.complete_with_system.call_count == 1


# ---------------------------------------------------------------------------
# Cross-provider: both satisfy BaseLLM contract identically
# ---------------------------------------------------------------------------

class TestProviderContract:
    @pytest.mark.parametrize("llm_cls,env_var,mock_path,response_factory", [
        (
            GroqLLM,
            "GROQ_API_KEY",
            "core.llm.groq.OpenAI",
            lambda: _make_groq_mock("contract response"),
        ),
        (
            AnthropicLLM,
            "ANTHROPIC_API_KEY",
            "core.llm.anthropic.anthropic",
            lambda: _make_anthropic_mock("contract response"),
        ),
    ])
    def test_complete_returns_string(self, monkeypatch, llm_cls, env_var, mock_path, response_factory):
        monkeypatch.setenv(env_var, "fake-key")
        mock_client, setup = response_factory()
        with patch(mock_path, **setup):
            llm = llm_cls()
            result = llm.complete("test prompt")
            assert isinstance(result, str)
            assert result == "contract response"


# ---------------------------------------------------------------------------
# Mock factories (shared)
# ---------------------------------------------------------------------------

def _make_groq_mock(text: str):
    mock_message = MagicMock()
    mock_message.content = text
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client, {"return_value": mock_client}


def _make_anthropic_mock(text: str):
    mock_block = MagicMock()
    mock_block.text = text
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_module = MagicMock()
    mock_module.Anthropic.return_value = mock_client
    return mock_client, {"new": mock_module}


# ---------------------------------------------------------------------------
# Audit fixes — new tests for previously uncovered edge cases
# ---------------------------------------------------------------------------

class TestGroqLLMAuditFixes:
    def test_empty_choices_raises_llm_error(self, monkeypatch):
        """Issue #1: IndexError on empty choices list must become LLMError."""
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        llm = GroqLLM()
        llm._client = mock_client
        with pytest.raises(LLMError, match="empty choices"):
            llm.complete("prompt")


class TestAnthropicLLMAuditFixes:
    def test_empty_content_raises_llm_error(self, monkeypatch):
        """Issue #1: empty content list must raise LLMError, not return empty string."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        mock_response = MagicMock()
        mock_response.content = []
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        llm = AnthropicLLM()
        llm._client = mock_client
        with pytest.raises(LLMError, match="empty content"):
            llm.complete("prompt")

    def test_non_text_block_raises_llm_error(self, monkeypatch):
        """Issue #1: content block without .text attr must raise LLMError."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        mock_block = MagicMock(spec=[])   # no attributes at all
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        llm = AnthropicLLM()
        llm._client = mock_client
        with pytest.raises(LLMError, match="unexpected content block"):
            llm.complete("prompt")
