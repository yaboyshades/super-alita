import importlib.util
from pathlib import Path
import sys

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "option_trainer", Path("src/plugins/oak_core/option_trainer.py")
)
option_trainer = importlib.util.module_from_spec(spec)
sys.modules["option_trainer"] = option_trainer
assert spec.loader is not None
spec.loader.exec_module(option_trainer)
OptionTrainer = option_trainer.OptionTrainer


class Bus:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, event_type: str, **kwargs) -> None:
        self.events.append({"event_type": event_type, **kwargs})

    async def subscribe(self, event_type, handler) -> None:  # pragma: no cover
        pass


class _TestOptionTrainer(OptionTrainer):
    async def start(self) -> None:  # pragma: no cover
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("ppo_epochs,expected_steps", [(1, 2), (3, 6)])
async def test_ppo_epochs_controls_iterations(ppo_epochs: int, expected_steps: int) -> None:
    bus = Bus()
    trainer = _TestOptionTrainer()
    await trainer.setup(
        bus,
        None,
        {"state_dim": 2, "action_dim": 2, "batch_size": 2, "ppo_epochs": ppo_epochs},
    )

    class SubEvent:
        subproblem_id = "1"

    await trainer.handle_subproblem_defined(SubEvent())
    opt_id = "option_1"

    net = trainer.options[opt_id]
    for p in net.parameters():
        torch.nn.init.constant_(p, 0.0)

    for i in range(4):
        class StepEvent:
            pass

        e = StepEvent()
        e.option_id = opt_id
        e.state = [0.0, 0.0]
        e.action = 0
        e.reward = 1.0
        e.next_state = [0.0, 0.0]
        e.done = i == 3
        await trainer.handle_state_transition(e)

    steps = 0
    orig_step = trainer.optim[opt_id].step

    def step_hook(*args, **kwargs):
        nonlocal steps
        steps += 1
        return orig_step(*args, **kwargs)

    trainer.optim[opt_id].step = step_hook  # type: ignore[assignment]

    await trainer.handle_training_tick(object())

    assert steps == expected_steps
