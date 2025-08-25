# src/core/settings.py
from __future__ import annotations

import os


def _get(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


# LLM configs
LLM_PROVIDER = _get("LLM_PROVIDER", "openai")  # "gemini" or "openai"
LLM_MODEL = _get("LLM_MODEL", "gpt-3.5-turbo")  # For OpenAI
# LLM_MODEL would be "gemini-2.5-pro" for Gemini
LLM_TIMEOUT_SEC = float(_get("LLM_TIMEOUT_SEC", "20"))
LLM_RETRIES = int(_get("LLM_RETRIES", "3"))
LLM_RETRY_BASE_DELAY_SEC = float(_get("LLM_RETRY_BASE_DELAY_SEC", "0.6"))
LLM_MAX_TOKENS = int(_get("LLM_MAX_TOKENS", "256"))

# Observability
TRACE_ENABLED = _get("TRACE_ENABLED", "false").lower() == "true"
