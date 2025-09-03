"""LADDER: Layered Autonomous Decomposition with Dynamic Execution and Reasoning.

A hierarchical task planning system with adaptive tool selection and dependency management.
"""

from .decomposers.base import (
    DefaultLLMDecomposer,
    LadderDecomposer,
    ParallelDecomposer,
    SequentialDecomposer,
)
from .graph.task_graph import (
    ExecutionStrategy,
    GraphValidationError,
    TaskGraph,
    TaskGraphMetrics,
    merge_task_graphs,
)

# Re-enabled integration imports after fixing cortex_adapter.py
from .integration.cortex_adapter import (
    LadderAdapter,
    LadderIntegrationConfig,
    PlanningMode,
)
from .models.task import (
    Task,
    TaskStatus,
    TaskTemplate,
)
from .planner import (
    ExecutionContext,
    ExecutionResult,
    LadderPlanner,
    PlannerConfig,
)
from .policies.bandit import (
    BanditPolicy,
    ContextualBanditPolicy,
    EpsilonGreedyPolicy,
    ThompsonSamplingPolicy,
    UCB1Policy,
)

__version__ = "0.1.0"

__all__ = [
    # Core planner
    "LadderPlanner",
    "PlannerConfig",
    "ExecutionContext",
    "ExecutionResult",
    # Task models
    "Task",
    "TaskStatus",
    "TaskTemplate",
    # Task graph
    "TaskGraph",
    "TaskGraphMetrics",
    "ExecutionStrategy",
    "GraphValidationError",
    "merge_task_graphs",
    # Decomposers
    "LadderDecomposer",
    "DefaultLLMDecomposer",
    "SequentialDecomposer",
    "ParallelDecomposer",
    # Bandit policies
    "BanditPolicy",
    "UCB1Policy",
    "EpsilonGreedyPolicy",
    "ThompsonSamplingPolicy",
    "ContextualBanditPolicy",
    # Integration
    "LadderAdapter",
    "LadderIntegrationConfig",
    "PlanningMode",
]
