"""
Decision Engine with Anti-Thrash Protection for Super Alita
Implements hysteresis, debouncing, and deduplication to prevent alert storms.
"""

import logging
from dataclasses import dataclass
from time import time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Threshold:
    """Threshold with hysteresis to prevent flapping"""

    warn: float
    crit: float
    clear: float  # must be < warn to add hysteresis

    def __post_init__(self):
        assert (
            self.clear < self.warn < self.crit
        ), "Thresholds must be: clear < warn < crit"


class AlertGate:
    """
    Prevents alert spam with debouncing and minimum intervals.
    Only allows alerts to fire if enough time has passed since last alert.
    """

    def __init__(self, min_interval_s: int = 600):  # 10 minutes default
        self.last_fired: dict[str, float] = {}
        self.open_alerts: set[str] = set()
        self.min_interval_s = min_interval_s

    def should_fire(self, key: str) -> bool:
        """Check if alert should fire based on debouncing rules"""
        current_time = time()
        last_fire_time = self.last_fired.get(key, 0)

        if current_time - last_fire_time < self.min_interval_s:
            logger.debug(
                f"Alert {key} debounced: {current_time - last_fire_time:.1f}s < {self.min_interval_s}s"
            )
            return False

        self.last_fired[key] = current_time
        self.open_alerts.add(key)
        logger.info(f"Alert {key} fired after {current_time - last_fire_time:.1f}s")
        return True

    def should_escalate(
        self, key: str, current_severity: str, new_severity: str
    ) -> bool:
        """Check if alert should be escalated to higher severity"""
        severity_order = {"CLEAR": 0, "WARN": 1, "CRIT": 2}
        current_level = severity_order.get(current_severity, 0)
        new_level = severity_order.get(new_severity, 0)

        # Allow escalation immediately, but only if significantly worse
        return new_level > current_level

    def can_clear(self, key: str) -> bool:
        """Check if alert can be cleared (was previously open)"""
        return key in self.open_alerts

    def clear_alert(self, key: str):
        """Mark alert as cleared"""
        self.open_alerts.discard(key)
        logger.info(f"Alert {key} cleared")


def classify_with_hysteresis(
    metric: float, threshold: Threshold, previous_state: str = "CLEAR"
) -> str:
    """
    Classify metric value with hysteresis to prevent flapping.

    Args:
        metric: Current metric value
        threshold: Threshold configuration with hysteresis
        previous_state: Previous classification state

    Returns:
        New classification: "CLEAR", "WARN", or "CRIT"
    """
    # Always escalate immediately if crossing critical
    if metric >= threshold.crit:
        return "CRIT"

    # Escalate to warning if crossing warning threshold
    if metric >= threshold.warn:
        return "WARN"

    # Only clear if below clear threshold (hysteresis)
    if metric < threshold.clear:
        return "CLEAR"

    # In hysteresis zone - maintain previous state
    if previous_state in ["WARN", "CRIT"] and metric >= threshold.clear:
        logger.debug(
            f"Metric {metric:.3f} in hysteresis zone, maintaining {previous_state}"
        )
        return previous_state

    return "CLEAR"


# Predefined thresholds for Super Alita metrics
THRESHOLDS = {
    "mailbox_pressure": Threshold(warn=0.50, crit=0.80, clear=0.35),
    "stale_rate": Threshold(warn=0.08, crit=0.15, clear=0.05),
    "concurrency_load": Threshold(warn=0.70, crit=0.90, clear=0.50),
    "ignored_triggers_rate": Threshold(
        warn=0.10, crit=0.20, clear=0.05
    ),  # per operation
}


class MetricsClassifier:
    """
    Classifies metrics with anti-thrash protection and state tracking.
    """

    def __init__(self, alert_gate: AlertGate | None = None):
        self.alert_gate = alert_gate or AlertGate()
        self.previous_states: dict[str, str] = {}
        self.consecutive_clears: dict[str, int] = {}

    def classify_metric(self, metric_name: str, value: float) -> tuple[str, bool]:
        """
        Classify a metric value and determine if action should be taken.

        Args:
            metric_name: Name of the metric (must be in THRESHOLDS)
            value: Current metric value

        Returns:
            Tuple of (classification, should_act)
        """
        if metric_name not in THRESHOLDS:
            logger.warning(f"Unknown metric: {metric_name}")
            return "CLEAR", False

        threshold = THRESHOLDS[metric_name]
        previous_state = self.previous_states.get(metric_name, "CLEAR")

        # Classify with hysteresis
        new_state = classify_with_hysteresis(value, threshold, previous_state)

        # Determine if we should act
        should_act = False

        if new_state in ["WARN", "CRIT"]:
            # Check if we should fire or escalate
            alert_key = f"{metric_name}/{new_state}"

            if previous_state == "CLEAR":
                # New alert
                should_act = self.alert_gate.should_fire(alert_key)
            elif self.alert_gate.should_escalate(
                metric_name, previous_state, new_state
            ):
                # Escalation
                should_act = True
                logger.info(
                    f"Escalating {metric_name}: {previous_state} -> {new_state}"
                )

            # Reset clear counter on any alert
            self.consecutive_clears[metric_name] = 0

        elif new_state == "CLEAR":
            # Only clear after consecutive CLEAR readings (double-check)
            self.consecutive_clears[metric_name] = (
                self.consecutive_clears.get(metric_name, 0) + 1
            )

            if self.consecutive_clears[metric_name] >= 2 and self.alert_gate.can_clear(
                metric_name
            ):
                should_act = True
                self.alert_gate.clear_alert(metric_name)
                logger.info(
                    f"Clearing {metric_name} after {self.consecutive_clears[metric_name]} consecutive CLEAR readings"
                )

        # Update state tracking
        self.previous_states[metric_name] = new_state

        return new_state, should_act

    def get_alert_summary(self) -> dict[str, any]:
        """Get summary of current alert state"""
        return {
            "open_alerts": list(self.alert_gate.open_alerts),
            "alert_count": len(self.alert_gate.open_alerts),
            "last_fired": dict(self.alert_gate.last_fired),
            "current_states": dict(self.previous_states),
        }


# Global classifier instance
_metrics_classifier: MetricsClassifier | None = None


def get_metrics_classifier() -> MetricsClassifier:
    """Get the global metrics classifier instance"""
    global _metrics_classifier
    if _metrics_classifier is None:
        _metrics_classifier = MetricsClassifier()
    return _metrics_classifier
