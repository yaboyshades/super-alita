from typing import Any

import pytest
import torch

from src.core.events import BaseEvent
from src.plugins.oak_core.option_trainer import OptionTrainer


class MockBus:
    def __init__(self):
        self.emitted_events = []
        self.handlers = {}

    async def publish(self, event: BaseEvent):
        self.emitted_events.append(event)
        event_type = event.event_type
        if event_type in self.handlers:
            # The real event bus passes the event object itself
            for handler in self.handlers[event_type]:
                await handler(event)

    async def subscribe(self, event_type: str, handler: Any):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

class MockMessageStore:
    def __init__(self):
        self.persisted = []
    async def persist(self, event_type, payload):
        self.persisted.append((event_type, payload))

@pytest.mark.asyncio
async def test_option_trainer_learns() -> None:
    store = MockMessageStore()

    config = {
        "device": "cpu",
        "batch_size": 2,
        "update_frequency": 4,
        "ppo_epochs": 1
    }

    trainer = OptionTrainer()
    bus = MockBus()
    await trainer.setup(bus, store, config)
    await trainer.start()

    # Create an option by publishing an event that the trainer subscribes to
    subproblem_event_data = {"subproblem_id": "sp_1", "feature_id": "f_1", "kappa": 1.0}

    # The plugin's emit_event calls create_event, so the test should do the same
    # to simulate how the event bus would receive the event.
    # However, for this test, we are calling the handler directly, so we pass the data.
    # The handler expects a dictionary.
    await trainer.create_option(subproblem_event_data)

    # Get the option_id from the event emitted by the trainer
    option_created_event = next((e for e in bus.emitted_events if e.event_type == "option_created"), None)
    assert option_created_event is not None
    opt_id = option_created_event.option_id

    # Start the option
    await trainer.handle_option_start({"option_id": opt_id, "state": {}})

    net = trainer.networks[opt_id]
    orig_params = [p.clone().detach() for p in net.parameters()]

    # Simulate transitions
    for i in range(4):
        await trainer.handle_transition({"state": {}, "action": 0, "reward": 1.0, "next_state": {}, "done": False, "features": []})

    # Check if training happened
    assert trainer.step_count == 4

    new_params = [p.clone().detach() for p in net.parameters()]
    params_changed = False
    for p_orig, p_new in zip(orig_params, new_params, strict=False):
        if not torch.equal(p_orig, p_new):
            params_changed = True
            break
    assert params_changed

    # Check for training event
    training_update_event = next((e for e in bus.emitted_events if e.event_type == "option_training_update"), None)
    assert training_update_event is not None
    assert training_update_event.option_id == opt_id
