"""
Main integration point for cortex-assisted development.
"""
from __future__ import annotations
from typing import Dict, Any

from src.core.event_bus import EventBus
from src.core.events import create_event
from src.core.temporal_graph import TemporalGraph
from src.core.navigation import NeuralNavigator, NavigationConfig
from src.plugins.cortex_adapter_plugin import CortexAdapterPlugin, GitHubCopilotCortex
from src.plugins.knowledge_gap_detector import KnowledgeGapDetector
from src.plugins.autonomy_tracker import AutonomyTracker
from src.orchestration.cortex_weaning import CortexWeaningOrchestrator


class CortexIntegration:
    """Main class for integrating cortex-assisted development."""

    def __init__(self, event_bus: EventBus, graph: TemporalGraph, navigator: NeuralNavigator):
        self.event_bus = event_bus
        self.graph = graph
        self.navigator = navigator

        # Initialize plugins
        self.cortex_adapter = CortexAdapterPlugin(event_bus, graph, navigator)
        self.gap_detector = KnowledgeGapDetector(event_bus)
        self.autonomy_tracker = AutonomyTracker(event_bus)
        self.weaning_orchestrator = CortexWeaningOrchestrator()

        # Register default cortex provider
        self.cortex_adapter.register_cortex("github_copilot", GitHubCopilotCortex())

        # Subscribe to autonomy updates to manage weaning
        self.event_bus.subscribe("autonomy_update", self.handle_autonomy_update)

    @property
    def name(self) -> str:
        return "cortex_integration"

    async def shutdown(self) -> None:
        pass

    async def handle_autonomy_update(self, event: Dict[str, Any]):
        """Handle autonomy updates to potentially advance phases."""
        data = event.get("data", {}) if isinstance(event, dict) else {}
        autonomy_score = float(data.get("current_score", 0.0))

        advanced = await self.weaning_orchestrator.advance_phase_if_ready(autonomy_score)
        if advanced:
            await self.event_bus.publish(
                create_event(
                    "phase_advanced",
                    event_version=1,
                    source_plugin=self.name,
                    new_phase=self.weaning_orchestrator.current_phase.value,
                    autonomy_score=autonomy_score,
                )
            )

    async def should_use_cortex(self, confidence: float, context: Dict[str, Any] | None = None) -> bool:
        return await self.weaning_orchestrator.should_use_cortex(confidence, context or {})

    async def get_system_status(self) -> Dict[str, Any]:
        learning_stats = await self.cortex_adapter.get_learning_stats()
        autonomy_status = await self.autonomy_tracker.get_graduation_readiness()
        return {
            "current_phase": self.weaning_orchestrator.current_phase.value,
            "learning_stats": learning_stats,
            "autonomy_status": autonomy_status,
            "cortex_providers": list(self.cortex_adapter.cortex_providers.keys()),
        }
