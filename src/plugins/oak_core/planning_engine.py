from __future__ import annotations

from typing import Any, Dict, List

from src.core.plugin_interface import PluginInterface


class PlanningEngine(PluginInterface):
    """Option-aware planning over goals; emits plan proposals.

    Emits:
      - oak.plan_proposed
    Subscribes:
      - goal_received
    """

    @property
    def name(self) -> str:  # type: ignore[override]
        return "oak_planning_engine"

    def __init__(self, option_source: Any | None = None) -> None:
        super().__init__()
        self.cfg: dict[str, Any] = {"beam_width": 3}
        self.option_source = option_source  # expects OptionTrainer-like with .options

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:  # type: ignore[override]
        await super().setup(event_bus, store, config)
        self.cfg.update(config or {})
        await self.subscribe("goal_received", self.handle_goal)

    async def handle_goal(self, event: Any) -> None:
        goal = getattr(event, "goal", "")
        candidates: List[str] = []
        if self.option_source and hasattr(self.option_source, "options"):
            candidates = list(getattr(self.option_source, "options").keys())
        plan = [{"option_id": oid, "step": 0} for oid in candidates[: int(self.cfg["beam_width"])]]
        await self.emit_event("oak.plan_proposed", goal=goal, plan=plan, options_considered=len(candidates))

