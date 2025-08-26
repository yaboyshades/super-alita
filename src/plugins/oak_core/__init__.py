"""
OaK Core Plugin package for Super Alita.

Provides cohesive Options-and-Knowledge components aligned with the
repo's PluginInterface and event bus contracts.
"""

from .coordinator import OakCoordinator
from .curation_manager import CurationManager
from .feature_discovery import FeatureDiscoveryEngine
from .option_trainer import OptionTrainer
from .planning_engine import PlanningEngine
from .prediction_engine import PredictionEngine
from .subproblem_manager import SubproblemManager

__all__ = [
    "OakCoordinator",
    "FeatureDiscoveryEngine",
    "SubproblemManager",
    "OptionTrainer",
    "PredictionEngine",
    "PlanningEngine",
    "CurationManager",
]
