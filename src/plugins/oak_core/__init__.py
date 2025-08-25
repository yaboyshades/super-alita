"""
OaK Core Plugin package for Super Alita.

Provides cohesive Options-and-Knowledge components aligned with the
repo's PluginInterface and event bus contracts.
"""

from .coordinator import OakCoordinator
from .feature_discovery import FeatureDiscoveryEngine
from .subproblem_manager import SubproblemManager
from .option_trainer import OptionTrainer
from .prediction_engine import PredictionEngine
from .planning_engine import PlanningEngine
from .curation_manager import CurationManager

__all__ = [
    "OakCoordinator",
    "FeatureDiscoveryEngine",
    "SubproblemManager",
    "OptionTrainer",
    "PredictionEngine",
    "PlanningEngine",
    "CurationManager",
]

