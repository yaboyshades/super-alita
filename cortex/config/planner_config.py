"""Configuration management for the Enhanced LADDER Planner."""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class PlannerConfig:
    """Configuration for the Enhanced LADDER Planner."""

    # Execution mode
    mode: str = "shadow"  # "shadow" or "active"

    # Multi-armed bandit parameters
    exploration_rate: float = 0.1  # ε for ε-greedy algorithm
    bandit_decay_factor: float = 0.95  # Decay factor for bandit statistics

    # Task management
    max_tasks: int = 50  # Maximum number of tasks in a plan
    energy_threshold: float = 10.0  # Maximum total energy for auto-execution
    priority_decay: float = 0.9  # Priority decay over time

    # Decomposition strategy
    max_decomposition_depth: int = 3  # Maximum task decomposition depth
    min_task_energy: float = 0.1  # Minimum energy for a task
    max_task_energy: float = 5.0  # Maximum energy for a task

    # Knowledge base
    kb_similarity_threshold: float = 0.3  # Similarity threshold for task matching
    kb_max_entries: int = 1000  # Maximum knowledge base entries
    kb_cleanup_interval: int = 100  # Cleanup interval (in executions)

    # Performance monitoring
    enable_metrics: bool = True  # Enable performance metrics collection
    metrics_retention_days: int = 30  # Metrics retention period

    # Logging
    log_level: str = "INFO"  # Logging level
    log_execution_details: bool = True  # Log detailed execution information

    @classmethod
    def from_env(cls) -> "PlannerConfig":
        """Create configuration from environment variables."""
        return cls(
            mode=os.getenv("LADDER_MODE", "shadow"),
            exploration_rate=float(os.getenv("LADDER_EXPLORATION_RATE", "0.1")),
            bandit_decay_factor=float(os.getenv("LADDER_BANDIT_DECAY", "0.95")),
            max_tasks=int(os.getenv("LADDER_MAX_TASKS", "50")),
            energy_threshold=float(os.getenv("LADDER_ENERGY_THRESHOLD", "10.0")),
            priority_decay=float(os.getenv("LADDER_PRIORITY_DECAY", "0.9")),
            max_decomposition_depth=int(os.getenv("LADDER_MAX_DEPTH", "3")),
            min_task_energy=float(os.getenv("LADDER_MIN_ENERGY", "0.1")),
            max_task_energy=float(os.getenv("LADDER_MAX_ENERGY", "5.0")),
            kb_similarity_threshold=float(os.getenv("LADDER_KB_SIMILARITY", "0.3")),
            kb_max_entries=int(os.getenv("LADDER_KB_MAX_ENTRIES", "1000")),
            kb_cleanup_interval=int(os.getenv("LADDER_KB_CLEANUP", "100")),
            enable_metrics=os.getenv("LADDER_ENABLE_METRICS", "true").lower() == "true",
            metrics_retention_days=int(os.getenv("LADDER_METRICS_RETENTION", "30")),
            log_level=os.getenv("LADDER_LOG_LEVEL", "INFO"),
            log_execution_details=os.getenv("LADDER_LOG_DETAILS", "true").lower()
            == "true",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "mode": self.mode,
            "exploration_rate": self.exploration_rate,
            "bandit_decay_factor": self.bandit_decay_factor,
            "max_tasks": self.max_tasks,
            "energy_threshold": self.energy_threshold,
            "priority_decay": self.priority_decay,
            "max_decomposition_depth": self.max_decomposition_depth,
            "min_task_energy": self.min_task_energy,
            "max_task_energy": self.max_task_energy,
            "kb_similarity_threshold": self.kb_similarity_threshold,
            "kb_max_entries": self.kb_max_entries,
            "kb_cleanup_interval": self.kb_cleanup_interval,
            "enable_metrics": self.enable_metrics,
            "metrics_retention_days": self.metrics_retention_days,
            "log_level": self.log_level,
            "log_execution_details": self.log_execution_details,
        }

    def update_from_dict(self, config_dict: dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def validate(self) -> list[str]:
        """Validate configuration parameters."""
        errors = []

        if self.mode not in ["shadow", "active"]:
            errors.append("Mode must be 'shadow' or 'active'")

        if not 0.0 <= self.exploration_rate <= 1.0:
            errors.append("Exploration rate must be between 0.0 and 1.0")

        if not 0.0 <= self.bandit_decay_factor <= 1.0:
            errors.append("Bandit decay factor must be between 0.0 and 1.0")

        if self.max_tasks <= 0:
            errors.append("Max tasks must be positive")

        if self.energy_threshold <= 0:
            errors.append("Energy threshold must be positive")

        if not 0.0 <= self.priority_decay <= 1.0:
            errors.append("Priority decay must be between 0.0 and 1.0")

        if self.max_decomposition_depth <= 0:
            errors.append("Max decomposition depth must be positive")

        if self.min_task_energy <= 0 or self.min_task_energy >= self.max_task_energy:
            errors.append(
                "Min task energy must be positive and less than max task energy"
            )

        if not 0.0 <= self.kb_similarity_threshold <= 1.0:
            errors.append("KB similarity threshold must be between 0.0 and 1.0")

        if self.kb_max_entries <= 0:
            errors.append("KB max entries must be positive")

        if self.kb_cleanup_interval <= 0:
            errors.append("KB cleanup interval must be positive")

        if self.metrics_retention_days <= 0:
            errors.append("Metrics retention days must be positive")

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append(
                "Log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )

        return errors


# Global configuration instance
PLANNER_CONFIG = PlannerConfig.from_env()


def get_planner_config() -> PlannerConfig:
    """Get the global planner configuration."""
    return PLANNER_CONFIG


def update_planner_config(config_dict: dict[str, Any]) -> list[str]:
    """Update the global planner configuration."""
    global PLANNER_CONFIG

    # Validate the configuration
    temp_config = PlannerConfig.from_env()
    temp_config.update_from_dict(config_dict)
    errors = temp_config.validate()

    if not errors:
        PLANNER_CONFIG.update_from_dict(config_dict)

    return errors


def reset_planner_config() -> None:
    """Reset the global planner configuration to environment defaults."""
    global PLANNER_CONFIG
    PLANNER_CONFIG = PlannerConfig.from_env()
