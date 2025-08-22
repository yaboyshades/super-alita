"""
Risk Scoring Engine for Super Alita
Converts multiple metrics into a single risk score for stable prioritization.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from time import time

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Priority levels with clear ordering"""

    P1 = "P1"  # Blockers - immediate action required
    P2 = "P2"  # High - action required within hours
    P3 = "P3"  # Medium - action required within days
    P4 = "P4"  # Low - action when convenient

    def __lt__(self, other):
        priority_order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
        return priority_order[self.value] < priority_order[other.value]


@dataclass
class RiskWeights:
    """Configurable weights for risk score calculation"""

    mailbox_pressure: float = 0.4  # High weight - directly impacts responsiveness
    stale_rate: float = 0.3  # Medium-high - indicates concurrency issues
    concurrency_load: float = 0.2  # Medium - system utilization
    ignored_triggers: float = 0.1  # Low - may indicate design issues

    def __post_init__(self):
        total = (
            self.mailbox_pressure
            + self.stale_rate
            + self.concurrency_load
            + self.ignored_triggers
        )
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total}"


def normalize_metric(value: float, max_value: float = 1.0) -> float:
    """Normalize metric to 0-1 range, clipping if necessary"""
    return max(0.0, min(1.0, value / max_value))


def risk_score(
    mailbox_pressure: float,
    stale_rate: float,
    concurrency_load: float,
    ignored_triggers_rate: float,
    weights: RiskWeights | None = None,
) -> float:
    """
    Calculate overall risk score from normalized metrics.

    Args:
        mailbox_pressure: 0-1, mailbox utilization
        stale_rate: 0-1, rate of stale completions
        concurrency_load: 0-1, concurrent operation load
        ignored_triggers_rate: 0-1, rate of ignored triggers
        weights: Custom weights (uses default if None)

    Returns:
        Risk score 0-1 (higher = more risk)
    """
    if weights is None:
        weights = RiskWeights()

    # Ensure all inputs are normalized
    mailbox_pressure = normalize_metric(mailbox_pressure)
    stale_rate = normalize_metric(stale_rate)
    concurrency_load = normalize_metric(concurrency_load)
    ignored_triggers_rate = normalize_metric(ignored_triggers_rate)

    score = (
        weights.mailbox_pressure * mailbox_pressure
        + weights.stale_rate * stale_rate
        + weights.concurrency_load * concurrency_load
        + weights.ignored_triggers * ignored_triggers_rate
    )

    return max(0.0, min(1.0, score))


def score_to_priority(score: float) -> Priority:
    """Convert risk score to priority level"""
    if score >= 0.75:
        return Priority.P1  # Critical - immediate action
    elif score >= 0.50:
        return Priority.P2  # High - urgent action
    elif score >= 0.25:
        return Priority.P3  # Medium - timely action
    else:
        return Priority.P4  # Low - routine action


@dataclass
class PriorityState:
    """Tracks priority state with cooldown protection"""

    current_priority: Priority
    last_change_time: float
    cooldown_until: float = 0.0

    def can_downgrade(self, new_priority: Priority, min_cooldown_s: int = 1800) -> bool:
        """Check if priority can be downgraded (with cooldown protection)"""
        current_time = time()

        # Never downgrade during cooldown unless score is very low
        if current_time < self.cooldown_until:
            return False

        # Allow downgrade if enough time has passed
        return new_priority > self.current_priority

    def can_upgrade(self, new_priority: Priority) -> bool:
        """Check if priority can be upgraded (immediate for urgency)"""
        return new_priority < self.current_priority

    def update_priority(self, new_priority: Priority, cooldown_s: int = 1800) -> bool:
        """
        Update priority with cooldown protection.

        Returns:
            True if priority was actually changed
        """
        current_time = time()

        # Allow immediate upgrades (escalation)
        if self.can_upgrade(new_priority):
            self.current_priority = new_priority
            self.last_change_time = current_time
            self.cooldown_until = current_time + cooldown_s
            logger.info(f"Priority escalated to {new_priority.value}")
            return True

        # Allow downgrades only after cooldown
        if self.can_downgrade(new_priority):
            self.current_priority = new_priority
            self.last_change_time = current_time
            # Shorter cooldown for downgrades
            self.cooldown_until = current_time + (cooldown_s // 2)
            logger.info(f"Priority downgraded to {new_priority.value}")
            return True

        return False


class RiskEngine:
    """
    Main risk assessment engine with state tracking and cooldown protection.
    """

    def __init__(self, weights: RiskWeights | None = None, cooldown_s: int = 1800):
        self.weights = weights or RiskWeights()
        self.cooldown_s = cooldown_s
        self.priority_states: dict[str, PriorityState] = {}
        self.risk_history: dict[
            str, list[tuple[float, float]]
        ] = {}  # (timestamp, score)

    def assess_risk(
        self,
        component: str,
        mailbox_pressure: float,
        stale_rate: float,
        concurrency_load: float,
        ignored_triggers_rate: float,
    ) -> dict[str, any]:
        """
        Assess risk for a component and update priority state.

        Args:
            component: Component identifier (e.g., "fsm", "api")
            metrics: Current metric values

        Returns:
            Assessment with score, priority, and change info
        """
        current_time = time()

        # Calculate risk score
        score = risk_score(
            mailbox_pressure,
            stale_rate,
            concurrency_load,
            ignored_triggers_rate,
            self.weights,
        )

        # Store in history
        if component not in self.risk_history:
            self.risk_history[component] = []
        self.risk_history[component].append((current_time, score))

        # Keep only recent history
        cutoff_time = current_time - 3600  # 1 hour
        self.risk_history[component] = [
            (t, s) for t, s in self.risk_history[component] if t > cutoff_time
        ]

        # Determine new priority
        new_priority = score_to_priority(score)

        # Update priority state with cooldown protection
        if component not in self.priority_states:
            self.priority_states[component] = PriorityState(
                current_priority=Priority.P4, last_change_time=current_time
            )

        priority_state = self.priority_states[component]
        priority_changed = priority_state.update_priority(new_priority, self.cooldown_s)

        # Calculate trend
        trend = self._calculate_trend(component)

        return {
            "component": component,
            "timestamp": current_time,
            "risk_score": score,
            "current_priority": priority_state.current_priority.value,
            "new_priority": new_priority.value,
            "priority_changed": priority_changed,
            "cooldown_active": current_time < priority_state.cooldown_until,
            "cooldown_remaining_s": max(
                0, priority_state.cooldown_until - current_time
            ),
            "trend": trend,
            "metrics": {
                "mailbox_pressure": mailbox_pressure,
                "stale_rate": stale_rate,
                "concurrency_load": concurrency_load,
                "ignored_triggers_rate": ignored_triggers_rate,
            },
            "weights": {
                "mailbox_pressure": self.weights.mailbox_pressure,
                "stale_rate": self.weights.stale_rate,
                "concurrency_load": self.weights.concurrency_load,
                "ignored_triggers": self.weights.ignored_triggers,
            },
        }

    def _calculate_trend(self, component: str) -> str:
        """Calculate risk trend for component"""
        if component not in self.risk_history:
            return "unknown"

        history = self.risk_history[component]
        if len(history) < 3:
            return "insufficient_data"

        # Compare recent vs older scores
        recent_scores = [s for _, s in history[-3:]]
        older_scores = (
            [s for _, s in history[:-3]] if len(history) > 3 else recent_scores
        )

        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores)

        diff = recent_avg - older_avg

        if diff > 0.05:  # 5% increase
            return "increasing"
        elif diff < -0.05:  # 5% decrease
            return "decreasing"
        else:
            return "stable"

    def get_component_summary(self, component: str) -> dict[str, any] | None:
        """Get current summary for a component"""
        if component not in self.priority_states:
            return None

        priority_state = self.priority_states[component]
        current_time = time()

        return {
            "component": component,
            "current_priority": priority_state.current_priority.value,
            "cooldown_active": current_time < priority_state.cooldown_until,
            "cooldown_remaining_s": max(
                0, priority_state.cooldown_until - current_time
            ),
            "last_change_time": priority_state.last_change_time,
            "trend": self._calculate_trend(component),
            "risk_samples": len(self.risk_history.get(component, [])),
        }

    def get_overall_summary(self) -> dict[str, any]:
        """Get overall risk summary across all components"""
        if not self.priority_states:
            return {"status": "no_data", "components": 0}

        current_time = time()

        # Count by priority
        priority_counts = {p.value: 0 for p in Priority}
        for state in self.priority_states.values():
            priority_counts[state.current_priority.value] += 1

        # Determine overall status
        if priority_counts["P1"] > 0:
            overall_status = "critical"
        elif priority_counts["P2"] > 0:
            overall_status = "warning"
        elif priority_counts["P3"] > 0:
            overall_status = "attention"
        else:
            overall_status = "normal"

        return {
            "status": overall_status,
            "components": len(self.priority_states),
            "priority_counts": priority_counts,
            "active_cooldowns": sum(
                1
                for state in self.priority_states.values()
                if current_time < state.cooldown_until
            ),
            "timestamp": current_time,
        }


# Global risk engine instance
_risk_engine: RiskEngine | None = None


def get_risk_engine() -> RiskEngine:
    """Get the global risk engine instance"""
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskEngine()
    return _risk_engine
