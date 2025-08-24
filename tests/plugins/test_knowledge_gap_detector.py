import pytest
from unittest.mock import AsyncMock
from src.core.event_bus import EventBus
from src.plugins.knowledge_gap_detector import KnowledgeGapDetector


@pytest.mark.asyncio
async def test_gap_on_low_confidence():
    bus = AsyncMock(spec=EventBus)
    class AllowAll:
        async def should_use_cortex(self, confidence, context):
            return True
    det = KnowledgeGapDetector(event_bus=bus, policy=AllowAll())
    await det.check_reasoning_confidence({"data": {"confidence": 0.1}})
    assert bus.publish.called
    payload = bus.publish.call_args.args[0]
    assert getattr(payload, "event_type", getattr(payload, "type", None)) == "knowledge_gap"
    assert getattr(payload, "gap_type", None) == "low_confidence"


@pytest.mark.asyncio
async def test_gap_on_short_path():
    bus = AsyncMock(spec=EventBus)
    class AllowAll:
        async def should_use_cortex(self, confidence, context):
            return True
    det = KnowledgeGapDetector(event_bus=bus, policy=AllowAll())
    await det.check_navigation_success({"data": {"path_length": 1}})
    assert bus.publish.called
    payload = bus.publish.call_args.args[0]
    assert getattr(payload, "event_type", getattr(payload, "type", None)) == "knowledge_gap"
    assert getattr(payload, "gap_type", None) == "isolated_knowledge"


@pytest.mark.asyncio
async def test_gap_on_uncertainty_pattern():
    bus = AsyncMock(spec=EventBus)
    class AllowAll:
        async def should_use_cortex(self, confidence, context):
            return True
    det = KnowledgeGapDetector(event_bus=bus, policy=AllowAll(), cooldown_seconds=0.5)
    await det.detect_uncertainty_patterns({"data": {"content": "I'm not sure about this"}})
    assert bus.publish.called
    payload = bus.publish.call_args.args[0]
    assert getattr(payload, "event_type", getattr(payload, "type", None)) == "knowledge_gap"
    assert getattr(payload, "gap_type", None) == "uncertainty_expression"
    await det.detect_uncertainty_patterns({"data": {"content": "I'm not sure again"}})
    assert bus.publish.call_count == 1
    await det._maybe_publish_gap(gap_description="x", context={"hop_count": 3}, gap_type="t")
    assert bus.publish.call_count == 1
