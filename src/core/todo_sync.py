"""
Idempotent Todo Synchronization for Super Alita
Manages metric-driven todos without duplicates, with clean creation/closing.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import time

from src.core.decision_engine import get_metrics_classifier
from src.core.risk_engine import Priority, get_risk_engine

logger = logging.getLogger(__name__)


class TodoAction(Enum):
    """Actions that can be taken on todos"""

    CREATE = "create"
    UPDATE = "update"
    ESCALATE = "escalate"
    CLOSE = "close"
    NO_ACTION = "no_action"


@dataclass
class MetricTodo:
    """Represents a metric-driven todo item"""

    key: str  # Unique key like "fsm/mailbox_pressure"
    title: str  # Human readable title
    description: str  # Detailed description with context
    priority: Priority  # Current priority level
    labels: list[str] = field(default_factory=lambda: ["auto", "metrics"])
    created_time: float = field(default_factory=time)
    updated_time: float = field(default_factory=time)
    metric_value: float | None = None
    threshold_info: dict | None = None
    suggested_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            "key": self.key,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "labels": self.labels,
            "created_time": self.created_time,
            "updated_time": self.updated_time,
            "metric_value": self.metric_value,
            "threshold_info": self.threshold_info,
            "suggested_actions": self.suggested_actions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetricTodo":
        """Create from dictionary"""
        return cls(
            key=data["key"],
            title=data["title"],
            description=data["description"],
            priority=Priority(data["priority"]),
            labels=data.get("labels", ["auto", "metrics"]),
            created_time=data.get("created_time", time()),
            updated_time=data.get("updated_time", time()),
            metric_value=data.get("metric_value"),
            threshold_info=data.get("threshold_info"),
            suggested_actions=data.get("suggested_actions", []),
        )


class TodoStore:
    """
    Simple file-based storage for metric todos.
    In production, this could be backed by a database.
    """

    def __init__(self, store_path: str = "metric_todos.json"):
        self.store_path = Path(store_path)
        self._todos: dict[str, MetricTodo] = {}
        self._load()

    def _load(self):
        """Load todos from storage"""
        if self.store_path.exists():
            try:
                with open(self.store_path) as f:
                    data = json.load(f)
                    self._todos = {
                        key: MetricTodo.from_dict(todo_data)
                        for key, todo_data in data.items()
                    }
                logger.info(
                    f"Loaded {len(self._todos)} metric todos from {self.store_path}"
                )
            except Exception as e:
                logger.error(f"Failed to load todos: {e}")
                self._todos = {}

    def _save(self):
        """Save todos to storage"""
        try:
            data = {key: todo.to_dict() for key, todo in self._todos.items()}
            with open(self.store_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save todos: {e}")

    def get(self, key: str) -> MetricTodo | None:
        """Get todo by key"""
        return self._todos.get(key)

    def set(self, key: str, todo: MetricTodo):
        """Store todo"""
        self._todos[key] = todo
        self._save()

    def delete(self, key: str) -> bool:
        """Delete todo by key"""
        if key in self._todos:
            del self._todos[key]
            self._save()
            return True
        return False

    def list_all(self) -> dict[str, MetricTodo]:
        """List all todos"""
        return self._todos.copy()

    def list_by_priority(self, priority: Priority) -> list[MetricTodo]:
        """List todos by priority"""
        return [todo for todo in self._todos.values() if todo.priority == priority]


class TodoSync:
    """
    Idempotent todo synchronization that prevents duplicates and manages lifecycle.
    """

    def __init__(self, store: TodoStore | None = None):
        self.store = store or TodoStore()
        self.risk_engine = get_risk_engine()
        self.metrics_classifier = get_metrics_classifier()

    def upsert_metric_todo(
        self,
        metric_key: str,
        metric_value: float,
        priority: Priority,
        title: str,
        description: str,
        suggested_actions: list[str] = None,
        threshold_info: dict = None,
    ) -> tuple[TodoAction, str]:
        """
        Create or update a metric-driven todo idempotently.

        Args:
            metric_key: Unique metric identifier (e.g., "fsm/mailbox_pressure")
            metric_value: Current metric value
            priority: Suggested priority
            title: Todo title
            description: Detailed description
            suggested_actions: List of suggested mitigation actions
            threshold_info: Threshold configuration details

        Returns:
            Tuple of (action_taken, todo_key)
        """
        current_time = time()
        existing = self.store.get(metric_key)

        if suggested_actions is None:
            suggested_actions = []

        if not existing:
            # Create new todo
            todo = MetricTodo(
                key=metric_key,
                title=title,
                description=description,
                priority=priority,
                labels=["auto", "metrics", priority.value.lower()],
                metric_value=metric_value,
                threshold_info=threshold_info,
                suggested_actions=suggested_actions,
            )

            self.store.set(metric_key, todo)
            logger.info(f"Created metric todo: {metric_key} (P{priority.value})")
            return TodoAction.CREATE, metric_key

        # Check if escalation is needed
        if priority < existing.priority:  # Higher priority (lower enum value)
            existing.priority = priority
            existing.updated_time = current_time
            existing.metric_value = metric_value
            existing.description = description
            existing.suggested_actions = suggested_actions

            # Update labels to include new priority
            existing.labels = [l for l in existing.labels if not l.startswith("p")]
            existing.labels.append(priority.value.lower())

            self.store.set(metric_key, existing)
            logger.info(f"Escalated metric todo: {metric_key} to {priority.value}")
            return TodoAction.ESCALATE, metric_key

        # Check if update is needed (same priority but different details)
        if existing.metric_value != metric_value or existing.description != description:
            existing.updated_time = current_time
            existing.metric_value = metric_value
            existing.description = description
            existing.suggested_actions = suggested_actions

            self.store.set(metric_key, existing)
            logger.debug(f"Updated metric todo: {metric_key}")
            return TodoAction.UPDATE, metric_key

        return TodoAction.NO_ACTION, metric_key

    def close_if_clear(self, metric_key: str, note: str = "") -> bool:
        """
        Close a metric todo if conditions are met.

        Args:
            metric_key: Metric identifier
            note: Optional closing note

        Returns:
            True if todo was closed
        """
        existing = self.store.get(metric_key)
        if not existing:
            return False

        # Add closing note to description if provided
        if note:
            existing.description += f"\n\nClosed: {note}"
            existing.updated_time = time()
            self.store.set(metric_key, existing)

        # Remove from active todos
        self.store.delete(metric_key)
        logger.info(f"Closed metric todo: {metric_key}")
        return True

    def sync_from_metrics(
        self, metrics_data: dict[str, float]
    ) -> dict[str, TodoAction]:
        """
        Synchronize todos based on current metrics data.

        Args:
            metrics_data: Current metric values

        Returns:
            Dict mapping metric keys to actions taken
        """
        actions = {}

        # Standard metric mappings
        metric_configs = {
            "mailbox_pressure": {
                "title": "High FSM Mailbox Pressure",
                "thresholds": {"warn": 0.50, "crit": 0.80, "clear": 0.35},
                "actions": [
                    "Increase concurrency limits or optimize FSM transitions",
                    "Review input processing pipeline for bottlenecks",
                    "Consider implementing backpressure mechanisms",
                ],
            },
            "stale_rate": {
                "title": "High Stale Completion Rate",
                "thresholds": {"warn": 0.08, "crit": 0.15, "clear": 0.05},
                "actions": [
                    "Review operation ID lifecycle and timing",
                    "Check for race conditions in FSM state transitions",
                    "Validate concurrency safety mechanisms",
                ],
            },
            "concurrency_load": {
                "title": "High Concurrency Load",
                "thresholds": {"warn": 0.70, "crit": 0.90, "clear": 0.50},
                "actions": [
                    "Implement operation throttling or queue management",
                    "Review resource allocation and limits",
                    "Consider load shedding for non-critical operations",
                ],
            },
        }

        for metric_name, value in metrics_data.items():
            if metric_name not in metric_configs:
                continue

            config = metric_configs[metric_name]
            thresholds = config["thresholds"]

            # Classify metric state
            classification, should_act = self.metrics_classifier.classify_metric(
                metric_name, value
            )

            if should_act and classification in ["WARN", "CRIT"]:
                # Determine priority based on classification
                priority = Priority.P1 if classification == "CRIT" else Priority.P2

                # Generate description with context
                description = f"""
Metric: {metric_name}
Current Value: {value:.3f}
Classification: {classification}
Thresholds: warn={thresholds['warn']}, crit={thresholds['crit']}, clear={thresholds['clear']}

This metric indicates potential performance or stability issues that require attention.
                """.strip()

                action, key = self.upsert_metric_todo(
                    metric_key=f"fsm/{metric_name}",
                    metric_value=value,
                    priority=priority,
                    title=config["title"],
                    description=description,
                    suggested_actions=config["actions"],
                    threshold_info=thresholds,
                )
                actions[metric_name] = action

            elif should_act and classification == "CLEAR":
                # Close todo if it exists
                if self.close_if_clear(
                    f"fsm/{metric_name}",
                    f"Metric cleared: {value:.3f} < {thresholds['clear']}",
                ):
                    actions[metric_name] = TodoAction.CLOSE

        return actions

    def get_active_todos_summary(self) -> dict[str, any]:
        """Get summary of all active metric todos"""
        todos = self.store.list_all()

        if not todos:
            return {"count": 0, "by_priority": {}, "total_metrics": 0}

        # Count by priority
        priority_counts = {p.value: 0 for p in Priority}
        for todo in todos.values():
            priority_counts[todo.priority.value] += 1

        # Find most urgent
        most_urgent = min(
            todos.values(), key=lambda t: (t.priority.value, -t.updated_time)
        )

        return {
            "count": len(todos),
            "by_priority": priority_counts,
            "most_urgent": {
                "key": most_urgent.key,
                "title": most_urgent.title,
                "priority": most_urgent.priority.value,
                "age_s": time() - most_urgent.created_time,
            },
            "keys": list(todos.keys()),
        }


# Global todo sync instance
_todo_sync: TodoSync | None = None


def get_todo_sync() -> TodoSync:
    """Get the global todo sync instance"""
    global _todo_sync
    if _todo_sync is None:
        _todo_sync = TodoSync()
    return _todo_sync
