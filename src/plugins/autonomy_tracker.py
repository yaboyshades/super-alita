from __future__ import annotations

"""Track agent autonomy and emit progress events."""

from dataclasses import dataclass
from typing import Any, Dict, List

from src.core.plugin_interface import PluginInterface
from src.core.event_bus import EventBus
from src.core.events import create_event


@dataclass
class AutonomyMetrics:
    tasks_completed: int
    assistance_requests: int

    def autonomy_score(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return 1 - (self.assistance_requests / self.tasks_completed)


class AutonomyTracker(PluginInterface):
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.metrics_history: List[AutonomyMetrics] = []

    @property
    def name(self) -> str:  # type: ignore[override]
        return "autonomy_tracker"

    async def setup(self, event_bus: Any, store: Any, config: Dict[str, Any]) -> None:  # type: ignore[override]
        self.event_bus = event_bus
        self.store = store
        self.config = config

    async def start(self) -> None:  # type: ignore[override]
        self.is_running = True

    async def record_metrics(self, tasks_completed: int, assistance_requests: int) -> None:
        metrics = AutonomyMetrics(tasks_completed, assistance_requests)
        self.metrics_history.append(metrics)
        await self.event_bus.publish(
            create_event(
                "autonomy_update",
                event_version=1,
                source_plugin=self.name,
                current_score=metrics.autonomy_score(),
                cortex_dependency=assistance_requests,
                trend="flat",
                milestone_reached=False,
            )
        )

    async def get_graduation_readiness(self) -> Dict[str, Any]:
        score = self.metrics_history[-1].autonomy_score() if self.metrics_history else 0.0
        ready = score > 0.8
        return {
            "ready": ready,
            "autonomy_score": score,
            "criteria_met": {"score_over_80": ready},
            "recommendation": "graduate" if ready else "continue_training",
        }
