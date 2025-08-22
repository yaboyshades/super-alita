"""
Smoothing and Trend Analysis for Super Alita Metrics
Provides EWMA smoothing and trend detection to avoid acting on noise.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from time import time
from typing import Any

logger = logging.getLogger(__name__)


class EWMA:
    """
    Exponentially Weighted Moving Average for smooth metric tracking.
    Reduces noise and provides stable trending signal.
    """

    def __init__(self, alpha: float = 0.3):
        """
        Initialize EWMA smoother.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
                  Higher alpha = more responsive to recent changes
                  Lower alpha = more stable, less responsive
        """
        assert 0 < alpha <= 1, "Alpha must be between 0 and 1"
        self.alpha = alpha
        self.value: float | None = None
        self.initialized = False

    def update(self, x: float) -> float:
        """Update EWMA with new value and return smoothed result"""
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value

        return self.value

    def get(self) -> float | None:
        """Get current EWMA value"""
        return self.value

    def reset(self):
        """Reset EWMA state"""
        self.value = None
        self.initialized = False


@dataclass
class TrendWindow:
    """
    Sliding window for trend analysis with timestamps.
    """

    max_size: int = 10
    values: deque = field(default_factory=deque)
    timestamps: deque = field(default_factory=deque)

    def add(self, value: float, timestamp: float | None = None):
        """Add a new value with timestamp"""
        if timestamp is None:
            timestamp = time()

        self.values.append(value)
        self.timestamps.append(timestamp)

        # Maintain max size
        while len(self.values) > self.max_size:
            self.values.popleft()
            self.timestamps.popleft()

    def slope(self) -> float:
        """Calculate simple slope (last - first)"""
        if len(self.values) < 2:
            return 0.0
        return self.values[-1] - self.values[0]

    def rate_of_change(self) -> float:
        """Calculate rate of change per second"""
        if len(self.values) < 2 or len(self.timestamps) < 2:
            return 0.0

        time_delta = self.timestamps[-1] - self.timestamps[0]
        if time_delta <= 0:
            return 0.0

        value_delta = self.values[-1] - self.values[0]
        return value_delta / time_delta

    def is_trending_up(self, threshold: float = 0.01) -> bool:
        """Check if trend is significantly upward"""
        return self.slope() > threshold

    def is_trending_down(self, threshold: float = -0.01) -> bool:
        """Check if trend is significantly downward"""
        return self.slope() < threshold

    def is_stable(self, threshold: float = 0.01) -> bool:
        """Check if trend is stable (low variance)"""
        return abs(self.slope()) <= threshold


class SmoothedMetric:
    """
    Combines EWMA smoothing with trend analysis for a single metric.
    """

    def __init__(self, name: str, alpha: float = 0.3, trend_window: int = 10):
        self.name = name
        self.ewma = EWMA(alpha)
        self.trend = TrendWindow(max_size=trend_window)
        self.raw_history = deque(maxlen=50)  # Keep recent raw values

    def update(self, raw_value: float) -> dict[str, Any]:
        """
        Update metric with new raw value.

        Returns:
            Dict with smoothed value, trend info, and analysis
        """
        # Store raw value
        self.raw_history.append(raw_value)

        # Update EWMA
        smoothed = self.ewma.update(raw_value)

        # Update trend window with smoothed values
        self.trend.add(smoothed)

        # Analysis
        return {
            "raw": raw_value,
            "smoothed": smoothed,
            "slope": self.trend.slope(),
            "rate_of_change": self.trend.rate_of_change(),
            "trending_up": self.trend.is_trending_up(),
            "trending_down": self.trend.is_trending_down(),
            "stable": self.trend.is_stable(),
            "samples": len(self.trend.values),
        }

    def should_alert(self, threshold: float, require_trend: bool = True) -> bool:
        """
        Determine if metric should trigger alert based on smoothed value and trend.

        Args:
            threshold: Alert threshold
            require_trend: If True, require upward trend for alerts

        Returns:
            True if should alert
        """
        if self.ewma.value is None:
            return False

        # Check if smoothed value exceeds threshold
        exceeds_threshold = self.ewma.value >= threshold

        if not require_trend:
            return exceeds_threshold

        # Also require upward trend to avoid alerting on stable high values
        return exceeds_threshold and (
            self.trend.is_trending_up() or not self.trend.is_stable()
        )

    def should_clear(self, clear_threshold: float) -> bool:
        """
        Determine if alert should be cleared based on smoothed value.
        More conservative than raw alerting.
        """
        if self.ewma.value is None:
            return False

        # Clear only if smoothed value is well below threshold AND stable/trending down
        below_threshold = self.ewma.value < clear_threshold
        safe_trend = self.trend.is_stable() or self.trend.is_trending_down()

        return below_threshold and safe_trend


class MetricsSmoother:
    """
    Manages smoothing for multiple metrics with coordinated analysis.
    """

    def __init__(self, alpha: float = 0.3, trend_window: int = 10):
        self.metrics: dict[str, SmoothedMetric] = {}
        self.alpha = alpha
        self.trend_window = trend_window

    def update_metric(self, name: str, value: float) -> dict[str, Any]:
        """Update a metric and return analysis"""
        if name not in self.metrics:
            self.metrics[name] = SmoothedMetric(name, self.alpha, self.trend_window)

        return self.metrics[name].update(value)

    def get_metric_analysis(self, name: str) -> dict[str, Any] | None:
        """Get current analysis for a metric"""
        if name not in self.metrics:
            return None

        metric = self.metrics[name]
        if metric.ewma.value is None:
            return None

        return {
            "name": name,
            "current_smoothed": metric.ewma.value,
            "slope": metric.trend.slope(),
            "rate_of_change": metric.trend.rate_of_change(),
            "trending_up": metric.trend.is_trending_up(),
            "trending_down": metric.trend.is_trending_down(),
            "stable": metric.trend.is_stable(),
            "samples": len(metric.trend.values),
            "raw_history_count": len(metric.raw_history),
        }

    def get_overall_health(self) -> dict[str, Any]:
        """Get overall system health based on all smoothed metrics"""
        if not self.metrics:
            return {"status": "unknown", "metrics_count": 0}

        # Analyze trends across all metrics
        trending_up_count = sum(
            1 for m in self.metrics.values() if m.trend.is_trending_up()
        )
        trending_down_count = sum(
            1 for m in self.metrics.values() if m.trend.is_trending_down()
        )
        stable_count = sum(1 for m in self.metrics.values() if m.trend.is_stable())

        total_metrics = len(self.metrics)

        # Determine overall trend
        if trending_up_count > total_metrics * 0.6:
            overall_trend = "degrading"
        elif trending_down_count > total_metrics * 0.6:
            overall_trend = "improving"
        else:
            overall_trend = "stable"

        return {
            "status": overall_trend,
            "metrics_count": total_metrics,
            "trending_up": trending_up_count,
            "trending_down": trending_down_count,
            "stable": stable_count,
            "trend_percentages": {
                "up": trending_up_count / total_metrics if total_metrics > 0 else 0,
                "down": trending_down_count / total_metrics if total_metrics > 0 else 0,
                "stable": stable_count / total_metrics if total_metrics > 0 else 0,
            },
        }


# Global smoother instance
_metrics_smoother: MetricsSmoother | None = None


def get_metrics_smoother() -> MetricsSmoother:
    """Get the global metrics smoother instance"""
    global _metrics_smoother
    if _metrics_smoother is None:
        _metrics_smoother = MetricsSmoother()
    return _metrics_smoother


def slope(last_n: list[float]) -> float:
    """
    Simple trend calculation: last - first value.
    Assumes equal spacing between samples.
    """
    if len(last_n) < 2:
        return 0.0
    return last_n[-1] - last_n[0]
