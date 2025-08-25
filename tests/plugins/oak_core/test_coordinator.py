import pytest

from src.plugins.oak_core.coordinator import OakCoordinator


class Bus:
    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, object]] = []

    async def emit(self, event_type: str, **kwargs) -> None:  # pragma: no cover
        self.subscriptions.append((event_type, kwargs))

    async def subscribe(self, event_type: str, handler) -> None:
        self.subscriptions.append((event_type, handler))


@pytest.mark.asyncio
async def test_coordinator_setup_propagates_bus() -> None:
    bus = Bus()
    coord = OakCoordinator()
    await coord.setup(bus, None, {})
    assert coord.event_bus is bus
    assert coord.feature_engine.event_bus is bus
    assert coord.option_trainer.event_bus is bus
    assert coord.planning_engine.option_source is coord.option_trainer
