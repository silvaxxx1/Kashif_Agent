"""
core/llm/ — LLM provider adapters for Kashif.

All adapters implement BaseLLM.complete(prompt) -> str.
fe_agent.py only ever holds a BaseLLM reference — providers are
resolved at runtime by cli/main.py from config.yaml.

Providers:
  groq.py      — Groq via openai SDK (default, fastest)
  anthropic.py — Anthropic Claude via anthropic SDK (secondary)
  ollama.py    — Local inference via Ollama's OpenAI-compatible API (no key needed)
"""
