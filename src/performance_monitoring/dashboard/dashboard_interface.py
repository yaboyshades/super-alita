"""
Dashboard Interface

Real-time performance monitoring and constitutional compliance dashboard
with visual indicators and alerting capabilities.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetric:
    """Dashboard metric display data."""
    name: str
    value: Any
    unit: str
    status: str  # success, warning, error
    trend: str | None = None  # up, down, stable
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)


@dataclass
class AlertIndicator:
    """Alert indicator for dashboard display."""
    level: str  # info, warning, error, critical
    message: str
    source: str
    timestamp: datetime
    acknowledged: bool = False


class DashboardInterface:
    """
    Real-time dashboard interface for performance monitoring and
    constitutional compliance tracking.
    
    Implements Article V: Clarity through clear visual indicators.
    Implements Article III: Simplicity through focused dashboard design.
    """
    
    def __init__(
        self,
        update_interval_seconds: int = 5,
        max_alerts: int = 100
    ):
        self.update_interval_seconds = update_interval_seconds
        self.max_alerts = max_alerts
        
        # Dashboard state
        self.metrics: dict[str, DashboardMetric] = {}
        self.alerts: list[AlertIndicator] = []
        self.constitutional_status = {}
        self.performance_summary = {}
        
        # Update tracking
        self.last_update = datetime.now(UTC)
        self._update_active = False
        self._update_task: asyncio.Task | None = None
        
        # Dashboard subscribers (for real-time updates)
        self.subscribers = []
        
        logger.info("Dashboard Interface initialized")

    async def start_dashboard(self) -> None:
        """Start real-time dashboard updates."""
        if self._update_active:
            logger.warning("Dashboard already active")
            return
            
        self._update_active = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Dashboard started with real-time updates")

    async def stop_dashboard(self) -> None:
        """Stop dashboard updates."""
        if not self._update_active:
            return
            
        self._update_active = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Dashboard stopped")

    def update_performance_data(self, performance_monitor) -> None:
        """Update dashboard with performance monitoring data."""
        try:
            summary = performance_monitor.get_performance_summary()
            self.performance_summary = summary
            
            # Extract key metrics for dashboard display
            self._update_metric(
                "response_time",
                summary.get("average_response_time_ms", 0),
                "ms",
                self._get_performance_status(summary.get("average_response_time_ms", 0), 1000)
            )
            
            self._update_metric(
                "success_rate",
                summary.get("success_rate", 1.0) * 100,
                "%",
                self._get_performance_status(summary.get("success_rate", 1.0), 0.95, reverse=True)
            )
            
            self._update_metric(
                "interactions_count",
                summary.get("interactions_count", 0),
                "count",
                "success"
            )
            
            # Update constitutional compliance from performance data
            constitutional_data = summary.get("constitutional_compliance", {})
            if constitutional_data.get("status") != "no_data":
                self._update_constitutional_status(constitutional_data)
                
        except Exception as e:
            logger.error(f"Error updating performance data: {e}")
            self._add_alert("error", f"Performance data update failed: {e}", "dashboard")

    def update_constitutional_data(self, constitutional_engine) -> None:
        """Update dashboard with constitutional compliance data."""
        try:
            trend_data = constitutional_engine.get_compliance_trend()
            self.constitutional_status = trend_data
            
            if trend_data.get("status") != "no_data":
                avg_score = trend_data.get("average_score", 0)
                compliance_rate = trend_data.get("compliance_rate", 0) * 100
                
                self._update_metric(
                    "constitutional_score",
                    avg_score,
                    "score",
                    self._get_constitutional_status(avg_score)
                )
                
                self._update_metric(
                    "compliance_rate",
                    compliance_rate,
                    "%",
                    self._get_constitutional_status(compliance_rate / 100)
                )
                
                # Check for compliance alerts
                if avg_score < 0.75:
                    self._add_alert(
                        "warning",
                        f"Constitutional compliance below threshold: {avg_score:.3f}",
                        "constitutional"
                    )
                    
        except Exception as e:
            logger.error(f"Error updating constitutional data: {e}")
            self._add_alert("error", f"Constitutional data update failed: {e}", "dashboard")

    def update_telemetry_data(self, telemetry_bridge) -> None:
        """Update dashboard with telemetry data."""
        try:
            summary = telemetry_bridge.get_telemetry_summary()
            
            self._update_metric(
                "event_rate",
                summary.get("event_rate_per_minute", 0),
                "events/min",
                "success"
            )
            
            # Buffer utilization metrics
            buffer_util = summary.get("buffer_utilization", {})
            for buffer_name, utilization in buffer_util.items():
                self._update_metric(
                    f"buffer_{buffer_name}",
                    utilization * 100,
                    "%",
                    self._get_performance_status(utilization, 0.8)
                )
                
        except Exception as e:
            logger.error(f"Error updating telemetry data: {e}")
            self._add_alert("error", f"Telemetry data update failed: {e}", "dashboard")

    def get_dashboard_state(self) -> dict[str, Any]:
        """Get current dashboard state for display."""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "last_update": self.last_update.isoformat(),
            "metrics": {name: asdict(metric) for name, metric in self.metrics.items()},
            "alerts": [asdict(alert) for alert in self.alerts[-20:]],  # Last 20 alerts
            "constitutional_status": self.constitutional_status,
            "performance_summary": self.performance_summary,
            "status": self._get_overall_status()
        }

    def get_constitutional_dashboard(self) -> dict[str, Any]:
        """Get constitutional compliance focused dashboard data."""
        constitutional_metrics = {
            name: metric for name, metric in self.metrics.items()
            if "constitutional" in name or "compliance" in name
        }
        
        constitutional_alerts = [
            alert for alert in self.alerts
            if alert.source == "constitutional" and not alert.acknowledged
        ]
        
        return {
            "constitutional_metrics": {name: asdict(metric) for name, metric in constitutional_metrics.items()},
            "constitutional_alerts": [asdict(alert) for alert in constitutional_alerts],
            "constitutional_status": self.constitutional_status,
            "compliance_indicators": self._get_compliance_indicators()
        }

    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert by index."""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].acknowledged = True
            logger.info(f"Alert acknowledged: {self.alerts[alert_index].message}")
            return True
        return False

    def add_subscriber(self, callback) -> None:
        """Add a subscriber for real-time dashboard updates."""
        self.subscribers.append(callback)
        logger.info("Dashboard subscriber added")

    def remove_subscriber(self, callback) -> None:
        """Remove a dashboard subscriber."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info("Dashboard subscriber removed")

    async def _update_loop(self) -> None:
        """Background update loop for dashboard refresh."""
        while self._update_active:
            try:
                await asyncio.sleep(self.update_interval_seconds)
                await self._refresh_dashboard()
                await self._notify_subscribers()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard update loop error: {e}")

    async def _refresh_dashboard(self) -> None:
        """Refresh dashboard data."""
        self.last_update = datetime.now(UTC)
        
        # Update system metrics
        self._update_system_metrics()
        
        # Clean old alerts
        self._cleanup_old_alerts()

    async def _notify_subscribers(self) -> None:
        """Notify all subscribers of dashboard updates."""
        if not self.subscribers:
            return
            
        dashboard_state = self.get_dashboard_state()
        
        for subscriber in self.subscribers[:]:  # Copy list to avoid modification during iteration
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(dashboard_state)
                else:
                    subscriber(dashboard_state)
            except Exception as e:
                logger.error(f"Subscriber notification error: {e}")
                # Remove failed subscriber
                self.subscribers.remove(subscriber)

    def _update_metric(
        self,
        name: str,
        value: Any,
        unit: str,
        status: str,
        trend: str | None = None
    ) -> None:
        """Update a dashboard metric."""
        # Calculate trend if not provided
        if trend is None and name in self.metrics:
            old_value = self.metrics[name].value
            if isinstance(value, (int, float)) and isinstance(old_value, (int, float)):
                if value > old_value * 1.05:
                    trend = "up"
                elif value < old_value * 0.95:
                    trend = "down"
                else:
                    trend = "stable"
        
        self.metrics[name] = DashboardMetric(
            name=name,
            value=value,
            unit=unit,
            status=status,
            trend=trend
        )

    def _add_alert(self, level: str, message: str, source: str) -> None:
        """Add an alert to the dashboard."""
        alert = AlertIndicator(
            level=level,
            message=message,
            source=source,
            timestamp=datetime.now(UTC)
        )
        
        self.alerts.append(alert)
        
        # Maintain alert limit
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        logger.info(f"Dashboard alert added: {level} - {message}")

    def _update_constitutional_status(self, constitutional_data: dict[str, Any]) -> None:
        """Update constitutional compliance status."""
        score = constitutional_data.get("score", 0)
        status = constitutional_data.get("status", "unknown")
        
        self._update_metric(
            "constitutional_score",
            score,
            "score",
            self._get_constitutional_status(score)
        )
        
        # Add alerts for compliance issues
        if status == "non_compliant":
            self._add_alert(
                "warning",
                "Constitutional compliance threshold not met",
                "constitutional"
            )

    def _get_performance_status(self, value: float, threshold: float, reverse: bool = False) -> str:
        """Get performance status based on threshold."""
        if reverse:
            # For metrics where higher is better (like success rate)
            if value >= threshold:
                return "success"
            elif value >= threshold * 0.8:
                return "warning"
            else:
                return "error"
        else:
            # For metrics where lower is better (like response time)
            if value <= threshold:
                return "success"
            elif value <= threshold * 1.5:
                return "warning"
            else:
                return "error"

    def _get_constitutional_status(self, score: float) -> str:
        """Get constitutional compliance status."""
        if score >= 0.75:
            return "success"
        elif score >= 0.50:
            return "warning"
        else:
            return "error"

    def _get_overall_status(self) -> str:
        """Get overall dashboard status."""
        error_count = sum(1 for m in self.metrics.values() if m.status == "error")
        warning_count = sum(1 for m in self.metrics.values() if m.status == "warning")
        
        if error_count > 0:
            return "error"
        elif warning_count > 0:
            return "warning"
        else:
            return "success"

    def _get_compliance_indicators(self) -> dict[str, Any]:
        """Get constitutional compliance indicators."""
        return {
            "color_coding": {
                "green": "≥0.75 (Compliant)",
                "yellow": "0.50-0.74 (Warning)", 
                "red": "<0.50 (Non-Compliant)"
            },
            "threshold": 0.75,
            "enforcement": "Automatic rejection below threshold"
        }

    def _update_system_metrics(self) -> None:
        """Update system-level metrics."""
        # System health check
        current_time = datetime.now(UTC)
        uptime_seconds = (current_time - self.last_update).total_seconds()
        
        self._update_metric(
            "dashboard_uptime",
            uptime_seconds,
            "seconds",
            "success"
        )

    def _cleanup_old_alerts(self) -> None:
        """Clean up old acknowledged alerts."""
        cutoff_time = datetime.now(UTC).timestamp() - (24 * 3600)  # 24 hours
        
        self.alerts = [
            alert for alert in self.alerts
            if not alert.acknowledged or alert.timestamp.timestamp() > cutoff_time
        ]