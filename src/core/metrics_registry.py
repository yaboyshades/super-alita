"""
Centralized metrics registry for Super Alita agent system.
Collects live metrics from FSM, sessions, and other components for Prometheus export.
"""

import threading
from collections import defaultdict
from typing import Any, Optional


class MetricsRegistry:
    """Thread-safe metrics registry for collecting system metrics"""

    _instance: Optional["MetricsRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return

        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, dict[str, Any]] = defaultdict(dict)
        self._data_lock = threading.RLock()
        self._initialized = True

    def increment_counter(
        self, name: str, value: int = 1, labels: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric"""
        with self._data_lock:
            key = self._make_key(name, labels)
            self._counters[key] += value

    def set_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Set a gauge metric value"""
        with self._data_lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Add observation to histogram metric"""
        with self._data_lock:
            key = self._make_key(name, labels)
            if key not in self._histograms:
                self._histograms[key] = {
                    "buckets": defaultdict(int),
                    "sum": 0.0,
                    "count": 0,
                }

            hist = self._histograms[key]
            hist["sum"] += value
            hist["count"] += 1

            # Standard Prometheus buckets
            buckets = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")]
            for bucket in buckets:
                if value <= bucket:
                    hist["buckets"][bucket] += 1

    def get_prometheus_metrics(self) -> str:
        """Export all metrics in Prometheus format"""
        lines = []

        with self._data_lock:
            # Export counters
            for key, value in self._counters.items():
                name, labels_str = self._parse_key(key)
                if not lines or not lines[-1].startswith(f"# TYPE {name}"):
                    lines.append(f"# HELP {name} Generated counter metric")
                    lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{labels_str} {value}")

            # Export gauges
            for key, value in self._gauges.items():
                name, labels_str = self._parse_key(key)
                if not lines or not lines[-1].startswith(f"# TYPE {name}"):
                    lines.append(f"# HELP {name} Generated gauge metric")
                    lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{labels_str} {value}")

            # Export histograms
            for key, hist_data in self._histograms.items():
                name, labels_str = self._parse_key(key)
                if not lines or not lines[-1].startswith(f"# TYPE {name}"):
                    lines.append(f"# HELP {name} Generated histogram metric")
                    lines.append(f"# TYPE {name} histogram")

                # Export buckets
                for bucket, count in hist_data["buckets"].items():
                    bucket_str = "+Inf" if bucket == float("inf") else str(bucket)
                    bucket_labels = self._add_label(labels_str, "le", bucket_str)
                    lines.append(f"{name}_bucket{bucket_labels} {count}")

                # Export sum and count
                lines.append(f"{name}_sum{labels_str} {hist_data['sum']}")
                lines.append(f"{name}_count{labels_str} {hist_data['count']}")

        return "\n".join(lines)

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> int:
        """Get current counter value"""
        with self._data_lock:
            key = self._make_key(name, labels)
            return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get current gauge value"""
        with self._data_lock:
            key = self._make_key(name, labels)
            return self._gauges.get(key, 0.0)

    def _make_key(self, name: str, labels: dict[str, str] | None = None) -> str:
        """Create a unique key for the metric"""
        if not labels:
            return name

        label_pairs = sorted(labels.items())
        label_str = ",".join(f"{k}={v}" for k, v in label_pairs)
        return f"{name}[{label_str}]"

    def _parse_key(self, key: str) -> tuple[str, str]:
        """Parse a metric key back to name and labels string"""
        if "[" not in key:
            return key, ""

        name, labels_part = key.split("[", 1)
        labels_part = labels_part.rstrip("]")

        if not labels_part:
            return name, ""

        # Convert to Prometheus label format
        label_pairs = []
        for pair in labels_part.split(","):
            k, v = pair.split("=", 1)
            label_pairs.append(f'{k}="{v}"')

        return name, "{" + ",".join(label_pairs) + "}"

    def _add_label(self, labels_str: str, key: str, value: str) -> str:
        """Add a label to existing labels string"""
        if not labels_str:
            return f'{{"{key}"="{value}"}}'

        # Insert into existing labels
        labels_content = labels_str[1:-1]  # Remove { }
        if labels_content:
            return f'{{{labels_content},"{key}"="{value}"}}'
        else:
            return f'{{"{key}"="{value}"}}'


# Global instance
_registry: MetricsRegistry | None = None


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry instance"""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


def increment_counter(
    name: str, value: int = 1, labels: dict[str, str] | None = None
) -> None:
    """Convenience function to increment a counter"""
    get_metrics_registry().increment_counter(name, value, labels)


def set_gauge(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    """Convenience function to set a gauge"""
    get_metrics_registry().set_gauge(name, value, labels)


def observe_histogram(
    name: str, value: float, labels: dict[str, str] | None = None
) -> None:
    """Convenience function to observe a histogram"""
    get_metrics_registry().observe_histogram(name, value, labels)
