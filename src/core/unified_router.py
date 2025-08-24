from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.core.decision_policy_v1 import DecisionPolicyEngine
from src.core.execution_flow import REUGExecutionFlow


class UnifiedRouter:
    def __init__(self, event_bus=None, plugin_registry: dict[str, Any] | None = None):
        self.decision_policy = DecisionPolicyEngine()
        self.execution_flow = REUGExecutionFlow(
            event_bus=event_bus, plugin_registry=plugin_registry or {}
        )

    async def route_request(self, user_input: str, session_id: str):
        context = {
            "session_id": session_id,
            "user_input": user_input,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        plan = await self.decision_policy.decide_and_plan(user_input, context)
        if hasattr(self.execution_flow, "execute_plan"):
            return await self.execution_flow.execute_plan(plan)
        return plan
