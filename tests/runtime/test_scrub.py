import asyncio
from unittest.mock import AsyncMock

from src.copilot.scrub import clamp_tokens, scrub_prompt
from src.core.gemini_pilot import GeminiPilotClient


def test_clamp_tokens():
    text = "one two three four"  # 4 tokens
    assert clamp_tokens(text, 2) == "one two"
    assert clamp_tokens(text, 10) == text
    assert clamp_tokens(text, 0) == ""


def test_scrub_prompt_removes_secrets():
    s = "Here is API_KEY=12345 and another secret token=abcd"
    cleaned = scrub_prompt(s)
    assert "12345" not in cleaned
    assert "abcd" not in cleaned
    assert cleaned.count("[REDACTED]") == 2


async def _run_pilot_decide(contract: str) -> str:
    client = GeminiPilotClient(api_key="test")
    client.load_contract = AsyncMock(return_value=contract)  # type: ignore[assignment]
    captured = {}

    async def fake_call(payload: dict):
        captured.update(payload)
        return {
            "reasoning": "",
            "action": "publish_event",
            "parameters": {},
            "schema_version": 1,
        }

    client._call_gemini = fake_call  # type: ignore[assignment]
    try:
        await client.pilot_decide(goal="g", agent_state="s")
    finally:
        await client.client.aclose()
    return captured["contents"][0]["parts"][0]["text"]


def test_pilot_decide_scrubs_secrets():
    contract = """
meta:
  max_tokens: 10
  schema_version: 1
system_instruction: ""
content: |
  secret API_KEY=12345
"""
    text = asyncio.run(_run_pilot_decide(contract))
    assert "12345" not in text
    assert "[REDACTED]" in text
