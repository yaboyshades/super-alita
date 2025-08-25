from unittest.mock import AsyncMock

import pytest

from src.core.execution_flow import REUGExecutionFlow
from src.core.states import TransitionTrigger


class StubPlan:
    def __init__(self):
        self.plan = [{"name": "echo_tool", "args": {"text": "hi"}, "type": "normal"}]
        self.strategy = "SINGLE_BEST"
        self.confidence = 0.9

class StubPolicy:
    async def decide_and_plan(self, _message, _ctx, _budget=None):
        return StubPlan()

@pytest.mark.asyncio
async def test_execution_flow_uses_decision_policy():
    bus = AsyncMock()
    flow = REUGExecutionFlow(event_bus=bus, plugin_registry={})
    # Inject stub policy
    flow.decision_policy = StubPolicy()
    # Set context
    flow.state_machine.context.user_input = "hello world"
    trig = await flow._handle_understand_state()
    assert trig == TransitionTrigger.TOOLS_ROUTED
    tools = flow.state_machine.context.tools_selected
    assert isinstance(tools, list) and tools and tools[0]["name"] == "echo_tool"
