import pytest

from src.abilities.gemini_codegen_ability import GeminiCodegenAbility


class DummyBus:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, event: dict) -> None:
        self.events.append(event)

    async def subscribe(self, event_type: str, handler):
        return None


@pytest.mark.asyncio
async def test_local_fallback_emits_proposal(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ability = GeminiCodegenAbility()
    await ability.setup(event_bus=DummyBus(), store=None, config={})
    await ability.start()

    out: dict = {}

    async def capture(evt_type: str, **payload):
        out["evt_type"] = evt_type
        out["payload"] = payload

    ability.emit_event = capture  # type: ignore

    await ability._on_request(
        {
            "event_type": "codegen_request",
            "requirements": "Create a FastAPI /users endpoint with JWT auth",
            "repo_path": ".",
            "context_files": [],
        }
    )

    assert out["evt_type"] == "codegen_implementation_proposed"
    payload = out["payload"]
    assert isinstance(payload.get("proposal_id"), str)
    assert payload.get("diffs")
    assert payload.get("tests") is not None
