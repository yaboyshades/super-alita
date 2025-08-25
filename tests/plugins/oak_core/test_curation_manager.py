import sys
from pathlib import Path
import importlib.util
import pytest

class Bus:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, event_type: str, **kwargs) -> None:
        self.events.append({"event_type": event_type, **kwargs})

    async def subscribe(self, event_type, handler) -> None:  # pragma: no cover
        pass

spec = importlib.util.spec_from_file_location(
    "curation_manager", Path("src/plugins/oak_core/curation_manager.py")
)
curation_manager_mod = importlib.util.module_from_spec(spec)
sys.modules["curation_manager"] = curation_manager_mod
assert spec.loader is not None
spec.loader.exec_module(curation_manager_mod)
CurationManager = curation_manager_mod.CurationManager

class _TestCurationManager(CurationManager):
    async def start(self) -> None:  # pragma: no cover
        pass

@pytest.mark.asyncio
async def test_emits_feature_utility_updated() -> None:
    bus = Bus()
    mgr = _TestCurationManager()
    await mgr.setup(bus, None, {})

    class ResultEvent:
        success = True
        error = ""

    await mgr.handle_tool_result(ResultEvent())
    assert any(evt.get("event_type") == "oak.feature_utility_updated" for evt in bus.events)
