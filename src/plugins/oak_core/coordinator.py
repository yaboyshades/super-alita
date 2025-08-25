from __future__ import annotations

from typing import Any, Dict

from src.core.plugin_interface import PluginInterface

from .feature_discovery import FeatureDiscoveryEngine
from .subproblem_manager import SubproblemManager
from .option_trainer import OptionTrainer
from .prediction_engine import PredictionEngine
from .planning_engine import PlanningEngine
from .curation_manager import CurationManager


class OakCoordinator(PluginInterface):
    """High level orchestrator wiring together OaK core engines."""

    @property
    def name(self) -> str:  # type: ignore[override]
        return "oak_coordinator"

    def __init__(self) -> None:
        super().__init__()
        # Instantiate subcomponents without runtime dependencies
        self.feature_engine = FeatureDiscoveryEngine()
        self.subproblem_manager = SubproblemManager()
        self.option_trainer = OptionTrainer()
        self.prediction_engine = PredictionEngine()
        self.planning_engine = PlanningEngine(option_source=self.option_trainer)
        self.curation_manager = CurationManager()

    async def setup(self, event_bus: Any, store: Any, config: Dict[str, Any]) -> None:  # type: ignore[override]
        await super().setup(event_bus, store, config)
        cfg = config or {}
        await self.feature_engine.setup(event_bus, store, cfg.get("feature_discovery", {}))
        await self.subproblem_manager.setup(event_bus, store, cfg.get("subproblem_manager", {}))
        await self.option_trainer.setup(event_bus, store, cfg.get("option_trainer", {}))
        await self.prediction_engine.setup(event_bus, store, cfg.get("prediction_engine", {}))
        # ensure planning engine sees the trained options
        self.planning_engine.option_source = self.option_trainer
        await self.planning_engine.setup(event_bus, store, cfg.get("planning_engine", {}))
        await self.curation_manager.setup(event_bus, store, cfg.get("curation_manager", {}))

    async def start(self) -> None:  # type: ignore[override]
        await super().start()
        for component in (
            self.feature_engine,
            self.subproblem_manager,
            self.option_trainer,
            self.prediction_engine,
            self.planning_engine,
            self.curation_manager,
        ):
            if hasattr(component, "start"):
                await component.start()

    async def shutdown(self) -> None:  # type: ignore[override]
        for component in (
            self.feature_engine,
            self.subproblem_manager,
            self.option_trainer,
            self.prediction_engine,
            self.planning_engine,
            self.curation_manager,
        ):
            if hasattr(component, "shutdown"):
                await component.shutdown()
        await super().shutdown()
