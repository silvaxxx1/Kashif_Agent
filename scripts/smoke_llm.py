"""
smoke_llm.py — core/llm/ smoke test

Tests the LLM layer with real API calls if keys are set,
or with mock clients if they aren't (so it always runs cleanly in CI).

Run from kashif_core/:
    uv run python scripts/smoke_llm.py

To test with a real provider, export the key first:
    export GROQ_API_KEY=gsk_...
    uv run python scripts/smoke_llm.py

    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python scripts/smoke_llm.py
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm.base import BaseLLM, LLMError
from core.llm.groq import GroqLLM
from core.llm.anthropic import AnthropicLLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)

def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")

def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# 1. BaseLLM — abstract contract
# ---------------------------------------------------------------------------

section("1. BaseLLM abstract contract")

class DummyLLM(BaseLLM):
    def complete(self, prompt: str) -> str:
        return f"echo: {prompt[:40]}"

llm = DummyLLM(model="dummy", temperature=0.1, max_tokens=128)
result = llm.complete("test prompt")
ok(f"DummyLLM.complete()  →  {result!r}")
ok(f"repr  →  {repr(llm)}")

try:
    BaseLLM(model="x")
    fail("BaseLLM should be abstract")
except TypeError:
    ok("BaseLLM cannot be instantiated directly")

err = LLMError("something went wrong")
ok(f"LLMError is Exception: {isinstance(err, Exception)}")


# ---------------------------------------------------------------------------
# 2. GroqLLM — structure checks (no API call)
# ---------------------------------------------------------------------------

section("2. GroqLLM — structure checks")

llm = GroqLLM()
ok(f"Default model      →  {llm.model}")
ok(f"Default api_key_env →  {llm.api_key_env}")
ok(f"Is BaseLLM         →  {isinstance(llm, BaseLLM)}")
ok(f"Client is lazy     →  {llm._client is None}")
ok(f"repr               →  {repr(llm)}")

# missing key → LLMError
old_key = os.environ.pop("GROQ_API_KEY", None)
try:
    llm2 = GroqLLM(api_key_env="GROQ_API_KEY")
    llm2._client = None
    llm2.complete("test")
    fail("Should have raised LLMError for missing key")
except LLMError as e:
    ok(f"Missing GROQ_API_KEY raises LLMError: {str(e)[:60]}")
finally:
    if old_key:
        os.environ["GROQ_API_KEY"] = old_key

# provider error → wrapped in LLMError
mock_client = MagicMock()
mock_client.chat.completions.create.side_effect = RuntimeError("rate limited")
llm3 = GroqLLM()
llm3._client = mock_client
try:
    llm3.complete("test")
    fail("Should have raised LLMError")
except LLMError as e:
    ok(f"Provider error wrapped in LLMError: {str(e)[:60]}")


# ---------------------------------------------------------------------------
# 3. AnthropicLLM — structure checks (no API call)
# ---------------------------------------------------------------------------

section("3. AnthropicLLM — structure checks")

llm = AnthropicLLM()
ok(f"Default model       →  {llm.model}")
ok(f"Default api_key_env →  {llm.api_key_env}")
ok(f"Is BaseLLM          →  {isinstance(llm, BaseLLM)}")
ok(f"Client is lazy      →  {llm._client is None}")
ok(f"repr                →  {repr(llm)}")

# missing key → LLMError
old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
try:
    llm2 = AnthropicLLM(api_key_env="ANTHROPIC_API_KEY")
    llm2._client = None
    llm2.complete("test")
    fail("Should have raised LLMError for missing key")
except LLMError as e:
    ok(f"Missing ANTHROPIC_API_KEY raises LLMError: {str(e)[:60]}")
finally:
    if old_key:
        os.environ["ANTHROPIC_API_KEY"] = old_key

# provider error → wrapped in LLMError
mock_client = MagicMock()
mock_client.messages.create.side_effect = RuntimeError("overloaded")
llm3 = AnthropicLLM()
llm3._client = mock_client
try:
    llm3.complete("test")
    fail("Should have raised LLMError")
except LLMError as e:
    ok(f"Provider error wrapped in LLMError: {str(e)[:60]}")


# ---------------------------------------------------------------------------
# 4. Mock complete() — both providers
# ---------------------------------------------------------------------------

section("4. Mock complete() — both providers return correct string")

PROMPT = "Write a one-line Python function that returns 42."

# Groq mock
def _groq_mock_client(text):
    mock_msg = MagicMock(); mock_msg.content = text
    mock_choice = MagicMock(); mock_choice.message = mock_msg
    mock_resp = MagicMock(); mock_resp.choices = [mock_choice]
    client = MagicMock(); client.chat.completions.create.return_value = mock_resp
    return client

# Anthropic mock
def _anthropic_mock_client(text):
    mock_block = MagicMock(); mock_block.text = text
    mock_resp = MagicMock(); mock_resp.content = [mock_block]
    client = MagicMock(); client.messages.create.return_value = mock_resp
    return client

groq_llm = GroqLLM()
groq_llm._client = _groq_mock_client("def f(): return 42")
result = groq_llm.complete(PROMPT)
ok(f"GroqLLM mock response     →  {result!r}")
assert isinstance(result, str) and len(result) > 0

ant_llm = AnthropicLLM()
ant_llm._client = _anthropic_mock_client("def f(): return 42")
result = ant_llm.complete(PROMPT)
ok(f"AnthropicLLM mock response →  {result!r}")
assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# 5. Real API calls (only if keys are present)
# ---------------------------------------------------------------------------

section("5. Real API calls (skipped if keys not set)")

groq_key = os.environ.get("GROQ_API_KEY")
if groq_key:
    try:
        real_groq = GroqLLM(model="llama-3.3-70b-versatile")
        response = real_groq.complete("Reply with exactly: GROQ_OK")
        ok(f"Groq live response  →  {response.strip()[:80]!r}")
    except LLMError as e:
        fail(f"Groq live call failed: {e}")
else:
    print("  [SKIP] GROQ_API_KEY not set — skipping live Groq test")

ant_key = os.environ.get("ANTHROPIC_API_KEY")
if ant_key:
    try:
        real_ant = AnthropicLLM(model="claude-haiku-4-5-20251001")
        response = real_ant.complete("Reply with exactly: ANTHROPIC_OK")
        ok(f"Anthropic live response →  {response.strip()[:80]!r}")
    except LLMError as e:
        fail(f"Anthropic live call failed: {e}")
else:
    print("  [SKIP] ANTHROPIC_API_KEY not set — skipping live Anthropic test")


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

section("DONE")
print("  core/llm/ smoke test complete.")
print("  All [OK] = expected behaviour. [SKIP] = key not set.\n")
