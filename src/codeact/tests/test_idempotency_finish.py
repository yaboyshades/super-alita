import pytest
from hypothesis import given, strategies as st

from codeact.actions import AgentFinish
from codeact.bus_handlers import CodeActStartRequest, handle_start
from codeact.runner import CodeActRunner
from codeact.sandbox import PythonSandbox
from tests.runtime.fakes import FakeEventBus


class FinishPolicy:
    def __call__(self, _):
        return AgentFinish()


@given(st.text(min_size=1, max_size=20))
@pytest.mark.asyncio
async def test_idempotent_atoms(code):
    sandbox = PythonSandbox()
    runner = CodeActRunner(sandbox, FinishPolicy())
    bus = FakeEventBus()
    event = CodeActStartRequest(code=code)
    await handle_start(event, bus, runner)
    first_atoms = [e for e in bus.events if e["event_type"] == "batch_atoms_created"][0]["atoms"]
    first_bonds = [e for e in bus.events if e["event_type"] == "batch_bonds_added"][0]["bonds"]
    bus.events.clear()
    await handle_start(event, bus, runner)
    second_atoms = [e for e in bus.events if e["event_type"] == "batch_atoms_created"][0]["atoms"]
    second_bonds = [e for e in bus.events if e["event_type"] == "batch_bonds_added"][0]["bonds"]
    assert first_atoms == second_atoms
    assert first_bonds == second_bonds
