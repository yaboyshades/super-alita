"""
Planning-Aware Metrics Integration for Super Alita

This module integrates key operational metrics directly into the planning system,
allowing the agent to make informed decisions about priorities and resource allocation
based on real-time system performance data.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from src.core.metrics_registry import get_metrics_registry

logger = logging.getLogger(__name__)


class PlanningMetricsProvider:
    """
    Provides metrics that directly influence planning and todo prioritization.

    Key metrics for planning:
    - FSM mailbox size (input queue pressure)
    - Stale completion rate (concurrency issues)
    - Error recovery frequency (system stability)
    - Active operations (system load)
    - Response time trends (performance degradation)
    """

    def __init__(self):
        self.registry = get_metrics_registry()
        self._last_metrics_snapshot = {}
        self._trend_history = []

    def get_planning_priority_metrics(self) -> dict[str, Any]:
        """
        Get metrics that should influence todo planning and prioritization.

        Returns a structured set of metrics with planning implications.
        """
        current_time = datetime.now(UTC)

        # Core FSM concurrency metrics
        mailbox_size = self.registry.get_gauge("sa_fsm_mailbox_size")
        mailbox_max = self.registry.get_gauge("sa_fsm_mailbox_size_max")
        active_ops = self.registry.get_gauge("sa_fsm_active_operations")
        ignored_triggers = self.registry.get_counter("sa_fsm_ignored_triggers_total")
        stale_completions = self.registry.get_counter("sa_fsm_stale_completions_total")
        total_operations = self.registry.get_counter("sa_fsm_operations_total")

        # Calculate derived planning metrics
        mailbox_pressure = mailbox_size / max(mailbox_max, 1) if mailbox_max > 0 else 0
        stale_rate = (
            stale_completions / max(total_operations, 1) if total_operations > 0 else 0
        )
        concurrency_load = active_ops / 5.0  # Assume 5 is reasonable max concurrent ops

        planning_metrics = {
            "timestamp": current_time.isoformat(),
            "system_health": {
                "mailbox_pressure": mailbox_pressure,  # 0.0-1.0, higher = more input backlog
                "stale_completion_rate": stale_rate,  # 0.0-1.0, higher = more concurrency issues
                "concurrency_load": concurrency_load,  # 0.0-1.0, higher = more system load
                "ignored_trigger_count": ignored_triggers,
            },
            "performance_indicators": {
                "current_mailbox_size": mailbox_size,
                "peak_mailbox_size": mailbox_max,
                "active_operations": active_ops,
                "total_operations_processed": total_operations,
            },
            "planning_implications": self._derive_planning_implications(
                mailbox_pressure, stale_rate, concurrency_load, ignored_triggers
            ),
        }

        # Store for trend analysis
        self._trend_history.append(planning_metrics)
        if len(self._trend_history) > 100:  # Keep last 100 snapshots
            self._trend_history.pop(0)

        return planning_metrics

    def _derive_planning_implications(
        self,
        mailbox_pressure: float,
        stale_rate: float,
        concurrency_load: float,
        ignored_triggers: int,
    ) -> dict[str, Any]:
        """
        Derive actionable planning implications from metrics.
        """
        implications = {
            "priority_adjustments": [],
            "suggested_actions": [],
            "risk_factors": [],
            "performance_trends": "stable",  # stable, improving, degrading
        }

        # High mailbox pressure - prioritize concurrency improvements
        if mailbox_pressure > 0.7:
            implications["priority_adjustments"].append(
                {
                    "area": "concurrency_optimization",
                    "reason": f"High input queue pressure ({mailbox_pressure:.2f})",
                    "suggested_priority": "high",
                }
            )
            implications["suggested_actions"].append(
                "Consider increasing concurrency limits or optimizing FSM state transitions"
            )

        # High stale completion rate - prioritize FSM stability
        if stale_rate > 0.1:  # More than 10% stale completions
            implications["priority_adjustments"].append(
                {
                    "area": "fsm_stability",
                    "reason": f"High stale completion rate ({stale_rate:.2f})",
                    "suggested_priority": "high",
                }
            )
            implications["suggested_actions"].append(
                "Review operation ID lifecycle and state transition timing"
            )

        # High concurrency load - prioritize resource management
        if concurrency_load > 0.8:
            implications["priority_adjustments"].append(
                {
                    "area": "resource_management",
                    "reason": f"High concurrent operation load ({concurrency_load:.2f})",
                    "suggested_priority": "medium",
                }
            )
            implications["suggested_actions"].append(
                "Consider implementing operation throttling or queue management"
            )

        # Many ignored triggers - may indicate design issues
        if ignored_triggers > 50:
            implications["risk_factors"].append(
                {
                    "factor": "frequent_ignored_triggers",
                    "description": f"{ignored_triggers} triggers ignored - possible FSM design issue",
                    "impact": "medium",
                }
            )

        # Determine overall trend
        if len(self._trend_history) >= 2:
            recent_pressure = [
                m["system_health"]["mailbox_pressure"] for m in self._trend_history[-5:]
            ]
            if len(recent_pressure) >= 2:
                if recent_pressure[-1] > recent_pressure[0] * 1.2:
                    implications["performance_trends"] = "degrading"
                elif recent_pressure[-1] < recent_pressure[0] * 0.8:
                    implications["performance_trends"] = "improving"

        return implications

    def get_todo_integration_summary(self) -> dict[str, Any]:
        """
        Get a summary specifically designed for todo list integration.

        Returns priorities and suggested actions that can be added to todo items.
        """
        metrics = self.get_planning_priority_metrics()
        implications = metrics["planning_implications"]

        return {
            "metrics_timestamp": metrics["timestamp"],
            "system_status": self._get_system_status_summary(metrics["system_health"]),
            "priority_suggestions": implications["priority_adjustments"],
            "actionable_items": implications["suggested_actions"],
            "risk_alerts": implications["risk_factors"],
            "trend": implications["performance_trends"],
        }

    def _get_system_status_summary(self, health_metrics: dict[str, Any]) -> str:
        """Generate a human-readable system status summary."""
        pressure = health_metrics["mailbox_pressure"]
        stale_rate = health_metrics["stale_completion_rate"]
        load = health_metrics["concurrency_load"]

        if pressure > 0.8 or stale_rate > 0.15 or load > 0.9:
            return "critical"
        elif pressure > 0.5 or stale_rate > 0.08 or load > 0.7:
            return "warning"
        elif pressure < 0.3 and stale_rate < 0.05 and load < 0.5:
            return "optimal"
        else:
            return "normal"

    def suggest_todo_priorities(
        self, current_todos: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Analyze current todos and suggest priority adjustments based on metrics.

        Args:
            current_todos: List of todo items with id, title, description, status

        Returns:
            List of priority adjustment suggestions
        """
        integration_summary = self.get_todo_integration_summary()
        suggestions = []

        # Map metric implications to todo areas
        area_mapping = {
            "concurrency_optimization": ["concurrency", "fsm", "mailbox", "queue"],
            "fsm_stability": ["fsm", "state", "transition", "operation"],
            "resource_management": ["performance", "optimization", "resource", "load"],
            "error_recovery": ["error", "recovery", "resilience", "fallback"],
            "test_coverage": ["test", "coverage", "validation", "verify"],
        }

        for priority_adj in integration_summary["priority_suggestions"]:
            area = priority_adj["area"]
            reason = priority_adj["reason"]
            suggested_priority = priority_adj["suggested_priority"]

            # Find matching todos
            relevant_keywords = area_mapping.get(area, [area])
            matching_todos = []

            for todo in current_todos:
                todo_text = (
                    f"{todo.get('title', '')} {todo.get('description', '')}".lower()
                )
                if any(keyword in todo_text for keyword in relevant_keywords):
                    matching_todos.append(todo)

            if matching_todos:
                suggestions.append(
                    {
                        "metric_area": area,
                        "reason": reason,
                        "suggested_priority": suggested_priority,
                        "affected_todos": [t["id"] for t in matching_todos],
                        "todo_titles": [t["title"] for t in matching_todos],
                    }
                )

        return suggestions


def create_metrics_aware_todos(
    base_todos: list[dict[str, Any]], metrics_provider: PlanningMetricsProvider
) -> list[dict[str, Any]]:
    """
    Create an enhanced todo list that incorporates metrics-driven priorities.

    Args:
        base_todos: Original todo list
        metrics_provider: Provider for planning metrics

    Returns:
        Enhanced todo list with metrics-informed priorities and new metric-driven items
    """
    enhanced_todos = base_todos.copy()
    integration_summary = metrics_provider.get_todo_integration_summary()

    # Add system status to description of in-progress items
    system_status = integration_summary["system_status"]
    status_note = f" [System: {system_status}]"

    for todo in enhanced_todos:
        if todo["status"] == "in-progress":
            todo["description"] += status_note

    # Add new metric-driven todos if there are critical issues
    if system_status in ["critical", "warning"]:
        for i, action in enumerate(integration_summary["actionable_items"]):
            metric_todo = {
                "id": len(enhanced_todos) + i + 1,
                "title": f"Metrics Alert: {action[:50]}...",
                "description": f"Action suggested by metrics analysis: {action}",
                "status": "not-started",
            }
            enhanced_todos.append(metric_todo)

    return enhanced_todos


# Global instance for easy access
_planning_metrics_provider = None


def get_planning_metrics_provider() -> PlanningMetricsProvider:
    """Get the global planning metrics provider instance."""
    global _planning_metrics_provider
    if _planning_metrics_provider is None:
        _planning_metrics_provider = PlanningMetricsProvider()
    return _planning_metrics_provider
