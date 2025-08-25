import importlib.util
import pathlib

import pytest

spec = importlib.util.spec_from_file_location(
    "curation_manager", pathlib.Path("src/plugins/oak_core/curation_manager.py")
)
curation_manager = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(curation_manager)
CurationManager = curation_manager.CurationManager


class DummyEventBus:
    def __init__(self) -> None:
        self.published = []
        self.subscriptions = []

    async def publish(self, event):
        self.published.append(event)

    async def subscribe(self, event_type, handler):
        self.subscriptions.append((event_type, handler))


class FeatureStore:
    def __init__(self) -> None:
        self.features = {}

    def create_feature(self, fid: str) -> None:
        self.features[fid] = {}


@pytest.mark.asyncio
async def test_curation_manager_emits_when_features_present():
    bus = DummyEventBus()
    store = FeatureStore()
    plugin = CurationManager()
    await plugin.setup(bus, store, {})
    # Features should be initialized
    assert {"global_play", "global_planning"} <= set(store.features)

    event = type("E", (), {"success": True})()
    await plugin.handle_tool_result(event)
    assert any(
        e.event_type == "oak.feature_utility_update" and e.feature_id == "global_play"
        for e in bus.published
    )


@pytest.mark.asyncio
async def test_curation_manager_skips_when_features_missing(caplog):
    bus = DummyEventBus()

    class BareStore:
        pass

    store = BareStore()
    plugin = CurationManager()
    await plugin.setup(bus, store, {})
    caplog.set_level("WARNING")

    event = type("E", (), {"success": True})()
    await plugin.handle_tool_result(event)

    assert not any(e.event_type == "oak.feature_utility_update" for e in bus.published)
    assert any("skipping utility update" in r.message for r in caplog.records)
