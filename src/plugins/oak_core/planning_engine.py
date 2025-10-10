from __future__ import annotations

from typing import Any

from src.core.plugin_interface import PluginInterface


class PlanningEngine(PluginInterface):
    """Option-aware planning over goals; emits plan proposals."""

    def __init__(self):
        super().__init__()
        self.option_source = None

    @property
    def name(self) -> str:
        return "oak_planning_engine"

    async def setup(
        self, event_bus: Any, store: Any, config: dict[str, Any]
    ) -> None:
        await super().setup(event_bus, store, config)
        self.option_source = self.get_config("option_source")

    async def start(self) -> None:
        await super().start()

    async def shutdown(self) -> None:
        await super().shutdown()

    async def handle_goal(self, event: Any) -> None:
        goal = event.get("goal", "")
        session_id = event.get("session_id")
        candidates: list[str] = []
        if self.option_source and hasattr(self.option_source, "options"):
            candidates = list(self.option_source.options.keys())

        beam_width = self.get_config("beam_width", 3)
        plan = [
            {"option_id": oid, "step": 0} for oid in candidates[:beam_width]
        ]
        await self.emit_event(
            "oak.plan_proposed",
            goal=goal,
            plan=plan,
            options_considered=len(candidates),
            session_id=session_id,
        )
