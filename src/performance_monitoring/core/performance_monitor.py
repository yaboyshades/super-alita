"""
Core Performance Monitor

Tracks extension interactions, system metrics, and performance indicators
according to constitutional principles and design specifications.
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    
    name: str
    value: float
    timestamp: datetime
    category: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtensionInteraction:
    """Extension interaction tracking data."""
    
    extension_id: str
    function_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    host_calls: List[str] = field(default_factory=list)
    wasm_operations: List[str] = field(default_factory=list)
    lsp_messages: List[str] = field(default_factory=list)


class PerformanceMonitor:
    """
    Core performance monitoring system that tracks extension interactions,
    system metrics, and constitutional compliance indicators.
    
    Implements Article III: Simplicity through focused monitoring responsibilities.
    Implements Article I: Library-First through standard monitoring patterns.
    """
    
    def __init__(
        self,
        metrics_retention_hours: int = 24,
        collection_interval_seconds: int = 10,
        enable_real_time_alerts: bool = True
    ):
        self.metrics_retention_hours = metrics_retention_hours
        self.collection_interval_seconds = collection_interval_seconds
        self.enable_real_time_alerts = enable_real_time_alerts
        
        # Metric storage with time-based retention
        self.metrics: deque = deque(maxlen=10000)
        self.interactions: deque = deque(maxlen=5000)
        
        # Real-time metric aggregation
        self.metric_aggregates: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.interaction_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Alert thresholds and handlers
        self.alert_thresholds: Dict[str, float] = {
            "response_time_ms": 1000.0,
            "error_rate": 0.05,
            "memory_usage_mb": 512.0,
            "cpu_usage_percent": 80.0,
            "constitutional_compliance": 0.75
        }
        self.alert_handlers: List[Callable] = []
        
        # Background processing
        self._monitoring_active = False
        self._background_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        logger.info("Performance Monitor initialized with constitutional compliance tracking")

    async def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self._monitoring_active:
            logger.warning("Performance monitoring already active")
            return
            
        self._monitoring_active = True
        self._background_task = asyncio.create_task(self._background_monitoring_loop())
        logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        if not self._monitoring_active:
            return
            
        self._monitoring_active = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Performance monitoring stopped")

    def track_extension_interaction(
        self,
        extension_id: str,
        function_name: str
    ) -> ExtensionInteraction:
        """Start tracking an extension interaction."""
        interaction = ExtensionInteraction(
            extension_id=extension_id,
            function_name=function_name,
            start_time=datetime.now(timezone.utc)
        )
        
        with self._lock:
            self.interactions.append(interaction)
            
        logger.debug(f"Started tracking interaction: {extension_id}.{function_name}")
        return interaction

    def complete_interaction(
        self,
        interaction: ExtensionInteraction,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Complete tracking of an extension interaction."""
        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - interaction.start_time).total_seconds() * 1000
        
        interaction.end_time = end_time
        interaction.duration_ms = duration_ms
        interaction.success = success
        interaction.error_message = error_message
        
        # Update real-time statistics
        self._update_interaction_stats(interaction)
        
        # Check for performance alerts
        if self.enable_real_time_alerts:
            self._check_interaction_alerts(interaction)
            
        logger.debug(
            f"Completed interaction: {interaction.extension_id}.{interaction.function_name} "
            f"({duration_ms:.2f}ms, success={success})"
        )

    def add_metric(
        self,
        name: str,
        value: float,
        category: str = "system",
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            category=category,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self._lock:
            self.metrics.append(metric)
            
        # Update real-time aggregates
        self._update_metric_aggregates(metric)
        
        # Check for alerts
        if self.enable_real_time_alerts:
            self._check_metric_alerts(metric)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        with self._lock:
            current_time = datetime.now(timezone.utc)
            
            # Filter recent metrics (last hour)
            recent_metrics = [
                m for m in self.metrics
                if (current_time - m.timestamp).total_seconds() < 3600
            ]
            
            # Filter recent interactions (last hour)
            recent_interactions = [
                i for i in self.interactions
                if i.end_time and (current_time - i.end_time).total_seconds() < 3600
            ]
        
        return {
            "timestamp": current_time.isoformat(),
            "metrics_count": len(recent_metrics),
            "interactions_count": len(recent_interactions),
            "average_response_time_ms": self._calculate_average_response_time(recent_interactions),
            "success_rate": self._calculate_success_rate(recent_interactions),
            "metric_aggregates": dict(self.metric_aggregates),
            "interaction_stats": dict(self.interaction_stats),
            "constitutional_compliance": self._get_constitutional_compliance_summary()
        }

    def get_extension_statistics(self, extension_id: str) -> Dict[str, Any]:
        """Get statistics for a specific extension."""
        with self._lock:
            extension_interactions = [
                i for i in self.interactions
                if i.extension_id == extension_id and i.end_time
            ]
        
        if not extension_interactions:
            return {"extension_id": extension_id, "no_data": True}
        
        total_interactions = len(extension_interactions)
        successful_interactions = sum(1 for i in extension_interactions if i.success)
        total_duration = sum(i.duration_ms for i in extension_interactions if i.duration_ms)
        
        return {
            "extension_id": extension_id,
            "total_interactions": total_interactions,
            "success_rate": successful_interactions / total_interactions if total_interactions > 0 else 0,
            "average_duration_ms": total_duration / total_interactions if total_interactions > 0 else 0,
            "total_duration_ms": total_duration,
            "host_calls_count": sum(len(i.host_calls) for i in extension_interactions),
            "wasm_operations_count": sum(len(i.wasm_operations) for i in extension_interactions),
            "lsp_messages_count": sum(len(i.lsp_messages) for i in extension_interactions)
        }

    def add_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")

    def set_alert_threshold(self, metric_name: str, threshold: float) -> None:
        """Set alert threshold for a metric."""
        self.alert_thresholds[metric_name] = threshold
        logger.info(f"Set alert threshold for {metric_name}: {threshold}")

    async def _background_monitoring_loop(self) -> None:
        """Background monitoring loop for continuous data collection."""
        while self._monitoring_active:
            try:
                await self._collect_system_metrics()
                await self._cleanup_old_data()
                await asyncio.sleep(self.collection_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                await asyncio.sleep(self.collection_interval_seconds)

    async def _collect_system_metrics(self) -> None:
        """Collect system-level performance metrics."""
        import psutil
        
        # CPU and memory metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        self.add_metric("cpu_usage_percent", cpu_percent, "system")
        self.add_metric("memory_usage_mb", memory.used / 1024 / 1024, "system")
        self.add_metric("memory_usage_percent", memory.percent, "system")
        
        # Process-specific metrics if available
        try:
            process = psutil.Process()
            self.add_metric("process_memory_mb", process.memory_info().rss / 1024 / 1024, "process")
            self.add_metric("process_cpu_percent", process.cpu_percent(), "process")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    async def _cleanup_old_data(self) -> None:
        """Clean up old metrics and interactions based on retention policy."""
        current_time = datetime.now(timezone.utc)
        retention_seconds = self.metrics_retention_hours * 3600
        
        with self._lock:
            # Clean old metrics
            while (self.metrics and 
                   (current_time - self.metrics[0].timestamp).total_seconds() > retention_seconds):
                self.metrics.popleft()
                
            # Clean old interactions
            while (self.interactions and 
                   self.interactions[0].start_time and
                   (current_time - self.interactions[0].start_time).total_seconds() > retention_seconds):
                self.interactions.popleft()

    def _update_metric_aggregates(self, metric: PerformanceMetric) -> None:
        """Update real-time metric aggregates."""
        category_key = f"{metric.category}_{metric.name}"
        
        if category_key not in self.metric_aggregates:
            self.metric_aggregates[category_key] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0.0
            }
        
        agg = self.metric_aggregates[category_key]
        agg["count"] += 1
        agg["sum"] += metric.value
        agg["min"] = min(agg["min"], metric.value)
        agg["max"] = max(agg["max"], metric.value)
        agg["avg"] = agg["sum"] / agg["count"]

    def _update_interaction_stats(self, interaction: ExtensionInteraction) -> None:
        """Update real-time interaction statistics."""
        key = f"{interaction.extension_id}_{interaction.function_name}"
        
        if key not in self.interaction_stats:
            self.interaction_stats[key] = {
                "total_calls": 0,
                "successful_calls": 0,
                "total_duration_ms": 0.0,
                "avg_duration_ms": 0.0,
                "success_rate": 0.0
            }
        
        stats = self.interaction_stats[key]
        stats["total_calls"] += 1
        stats["successful_calls"] += 1 if interaction.success else 0
        
        if interaction.duration_ms:
            stats["total_duration_ms"] += interaction.duration_ms
            stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["total_calls"]
        
        stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]

    def _check_metric_alerts(self, metric: PerformanceMetric) -> None:
        """Check if metric triggers any alerts."""
        threshold_key = metric.name
        if threshold_key in self.alert_thresholds:
            threshold = self.alert_thresholds[threshold_key]
            if metric.value > threshold:
                self._trigger_alert(
                    f"Metric threshold exceeded: {metric.name}",
                    {
                        "metric_name": metric.name,
                        "value": metric.value,
                        "threshold": threshold,
                        "category": metric.category,
                        "timestamp": metric.timestamp.isoformat()
                    }
                )

    def _check_interaction_alerts(self, interaction: ExtensionInteraction) -> None:
        """Check if interaction triggers any alerts."""
        if interaction.duration_ms and interaction.duration_ms > self.alert_thresholds.get("response_time_ms", 1000):
            self._trigger_alert(
                f"Slow interaction detected: {interaction.extension_id}.{interaction.function_name}",
                {
                    "extension_id": interaction.extension_id,
                    "function_name": interaction.function_name,
                    "duration_ms": interaction.duration_ms,
                    "threshold": self.alert_thresholds["response_time_ms"]
                }
            )
        
        if not interaction.success:
            self._trigger_alert(
                f"Failed interaction: {interaction.extension_id}.{interaction.function_name}",
                {
                    "extension_id": interaction.extension_id,
                    "function_name": interaction.function_name,
                    "error_message": interaction.error_message
                }
            )

    def _trigger_alert(self, message: str, details: Dict[str, Any]) -> None:
        """Trigger alert handlers."""
        alert_data = {
            "message": message,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        for handler in self.alert_handlers:
            try:
                handler(message, alert_data)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

    def _calculate_average_response_time(self, interactions: List[ExtensionInteraction]) -> float:
        """Calculate average response time for interactions."""
        valid_interactions = [i for i in interactions if i.duration_ms is not None]
        if not valid_interactions:
            return 0.0
        return sum(i.duration_ms for i in valid_interactions) / len(valid_interactions)

    def _calculate_success_rate(self, interactions: List[ExtensionInteraction]) -> float:
        """Calculate success rate for interactions."""
        if not interactions:
            return 1.0
        successful = sum(1 for i in interactions if i.success)
        return successful / len(interactions)

    def _get_constitutional_compliance_summary(self) -> Dict[str, Any]:
        """Get constitutional compliance summary from recent metrics."""
        # Look for constitutional compliance metrics
        compliance_metrics = [
            m for m in self.metrics
            if "constitutional" in m.name.lower() or "compliance" in m.name.lower()
        ]
        
        if not compliance_metrics:
            return {"status": "no_data", "message": "No constitutional compliance data available"}
        
        # Get most recent compliance score
        recent_compliance = max(compliance_metrics, key=lambda m: m.timestamp)
        
        return {
            "status": "compliant" if recent_compliance.value >= 0.75 else "non_compliant",
            "score": recent_compliance.value,
            "threshold": 0.75,
            "last_updated": recent_compliance.timestamp.isoformat(),
            "total_measurements": len(compliance_metrics)
        }