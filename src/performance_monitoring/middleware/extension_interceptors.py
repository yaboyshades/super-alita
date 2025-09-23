"""
Extension Interaction Telemetry Middleware

Provides decorators and interceptors for automatic telemetry collection
around extension entrypoints with 100% error capture and 10% success sampling.
"""

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar

from ..telemetry.opentelemetry_config import (
    get_telemetry_collector,
    telemetry_span,
    telemetry_span_sync,
)

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class ExtensionMetrics:
    """Extension interaction metrics."""
    
    extension_id: str
    interaction_type: str  # query, validation, rule_check
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    payload_size_bytes: int = 0
    response_size_bytes: int = 0


class ExtensionTelemetryMiddleware:
    """Middleware for automatic extension telemetry collection."""
    
    def __init__(self):
        self.active_interactions: Dict[str, ExtensionMetrics] = {}
        self.telemetry_collector = get_telemetry_collector()
        
    def track_extension_call(self,
                            extension_id: str,
                            interaction_type: str = "query") -> Callable[[F], F]:
        """Decorator to track extension calls with telemetry."""
        
        def decorator(func: F) -> F:
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._track_async_call(
                        func, extension_id, interaction_type, args, kwargs
                    )
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self._track_sync_call(
                        func, extension_id, interaction_type, args, kwargs
                    )
                return sync_wrapper
        
        return decorator
    
    async def _track_async_call(self,
                                func: Callable,
                                extension_id: str,
                                interaction_type: str,
                                args: tuple,
                                kwargs: dict) -> Any:
        """Track async extension call with telemetry."""
        
        start_time = time.perf_counter()
        
        # Estimate payload size
        payload_size = self._estimate_payload_size(args, kwargs)
        
        async with telemetry_span(
            component="extension_middleware",
            operation=f"{extension_id}_{interaction_type}",
            tags={
                "extension_id": extension_id,
                "interaction_type": interaction_type,
                "payload_size_bytes": payload_size
            }
        ) as span:
            try:
                result = await func(*args, **kwargs)
                
                # Calculate response size and metrics
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                response_size = self._estimate_response_size(result)
                
                # Log success metrics
                self._log_interaction_metrics(
                    extension_id=extension_id,
                    interaction_type=interaction_type,
                    duration_ms=duration_ms,
                    payload_size_bytes=payload_size,
                    response_size_bytes=response_size,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # Calculate error metrics
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                # Log error metrics (100% capture)
                self._log_interaction_metrics(
                    extension_id=extension_id,
                    interaction_type=interaction_type,
                    duration_ms=duration_ms,
                    payload_size_bytes=payload_size,
                    response_size_bytes=0,
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                
                raise
    
    def _track_sync_call(self,
                        func: Callable,
                        extension_id: str,
                        interaction_type: str,
                        args: tuple,
                        kwargs: dict) -> Any:
        """Track sync extension call with telemetry."""
        
        start_time = time.perf_counter()
        
        # Estimate payload size
        payload_size = self._estimate_payload_size(args, kwargs)
        
        with telemetry_span_sync(
            component="extension_middleware",
            operation=f"{extension_id}_{interaction_type}",
            tags={
                "extension_id": extension_id,
                "interaction_type": interaction_type,
                "payload_size_bytes": payload_size
            }
        ) as span:
            try:
                result = func(*args, **kwargs)
                
                # Calculate response size and metrics
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                response_size = self._estimate_response_size(result)
                
                # Log success metrics
                self._log_interaction_metrics(
                    extension_id=extension_id,
                    interaction_type=interaction_type,
                    duration_ms=duration_ms,
                    payload_size_bytes=payload_size,
                    response_size_bytes=response_size,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # Calculate error metrics
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                # Log error metrics (100% capture)
                self._log_interaction_metrics(
                    extension_id=extension_id,
                    interaction_type=interaction_type,
                    duration_ms=duration_ms,
                    payload_size_bytes=payload_size,
                    response_size_bytes=0,
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                
                raise
    
    def _log_interaction_metrics(self,
                                extension_id: str,
                                interaction_type: str,
                                duration_ms: float,
                                payload_size_bytes: int,
                                response_size_bytes: int,
                                success: bool,
                                error_type: Optional[str] = None,
                                error_message: Optional[str] = None) -> None:
        """Log structured interaction metrics."""
        
        metrics = {
            "extension_id": extension_id,
            "interaction_type": interaction_type,
            "duration_ms": duration_ms,
            "payload_size_bytes": payload_size_bytes,
            "response_size_bytes": response_size_bytes,
            "success": success,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": time.time()
        }
        
        # Log to telemetry collector
        if self.telemetry_collector:
            # Create mock span for metrics tracking
            mock_span = type('MockSpan', (), {
                'duration_ms': duration_ms,
                'status_code': 'OK' if success else 'ERROR'
            })()
            self.telemetry_collector._update_metrics(mock_span)
        
        # Structured logging
        if success:
            logger.info(
                "Extension interaction completed: %s",
                metrics, 
                extra={"telemetry": metrics}
            )
        else:
            logger.error(
                "Extension interaction failed: %s",
                metrics,
                extra={"telemetry": metrics}
            )
    
    def _estimate_payload_size(self, args: tuple, kwargs: dict) -> int:
        """Estimate payload size in bytes."""
        try:
            import pickle
            return len(pickle.dumps((args, kwargs)))
        except Exception:
            # Fallback estimation
            return len(str(args)) + len(str(kwargs))
    
    def _estimate_response_size(self, response: Any) -> int:
        """Estimate response size in bytes."""
        try:
            import pickle
            return len(pickle.dumps(response))
        except Exception:
            # Fallback estimation
            return len(str(response))
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get summary of extension interactions."""
        # This would aggregate from telemetry collector
        if self.telemetry_collector:
            return self.telemetry_collector.get_metrics_summary()
        
        return {"status": "no_telemetry_collector"}


# Global middleware instance
_extension_middleware = ExtensionTelemetryMiddleware()


def get_extension_middleware() -> ExtensionTelemetryMiddleware:
    """Get global extension telemetry middleware."""
    return _extension_middleware


def track_extension_call(extension_id: str, 
                        interaction_type: str = "query") -> Callable[[F], F]:
    """Decorator to automatically track extension calls."""
    return _extension_middleware.track_extension_call(extension_id, interaction_type)


# Convenience decorators for common extension types
def track_validator_call(extension_id: str) -> Callable[[F], F]:
    """Track constitutional validator calls."""
    return track_extension_call(extension_id, "validation")


def track_rule_check(extension_id: str) -> Callable[[F], F]:
    """Track constitutional rule checks."""
    return track_extension_call(extension_id, "rule_check")


def track_query_call(extension_id: str) -> Callable[[F], F]:
    """Track general query calls."""
    return track_extension_call(extension_id, "query")


@asynccontextmanager
async def extension_interaction_context(extension_id: str,
                                       interaction_type: str = "query"):
    """Async context manager for manual extension interaction tracking."""
    middleware = get_extension_middleware()
    start_time = time.perf_counter()
    
    async with telemetry_span(
        component="extension_middleware",
        operation=f"{extension_id}_{interaction_type}",
        tags={
            "extension_id": extension_id,
            "interaction_type": interaction_type
        }
    ):
        try:
            yield
            
            # Log success
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            middleware._log_interaction_metrics(
                extension_id=extension_id,
                interaction_type=interaction_type,
                duration_ms=duration_ms,
                payload_size_bytes=0,
                response_size_bytes=0,
                success=True
            )
            
        except Exception as e:
            # Log error
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            middleware._log_interaction_metrics(
                extension_id=extension_id,
                interaction_type=interaction_type,
                duration_ms=duration_ms,
                payload_size_bytes=0,
                response_size_bytes=0,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            raise


# Performance threshold monitoring
class PerformanceThresholdMonitor:
    """Monitor and alert on performance threshold violations."""
    
    def __init__(self):
        self.slo_violations: List[Dict[str, Any]] = []
        self.telemetry_collector = get_telemetry_collector()
    
    def check_slo_violations(self) -> List[Dict[str, Any]]:
        """Check for SLO violations and return them."""
        if self.telemetry_collector:
            summary = self.telemetry_collector.get_metrics_summary()
            
            violations = []
            
            # Check latency SLO
            if (summary.get("latency_ms", {}).get("p95", 0) > 
                self.telemetry_collector.slos.latency_p95_ms):
                violations.append({
                    "type": "latency_slo_violation",
                    "metric": "p95_latency",
                    "threshold": self.telemetry_collector.slos.latency_p95_ms,
                    "actual": summary["latency_ms"]["p95"],
                    "timestamp": time.time()
                })
            
            # Check error rate SLO
            if (summary.get("error_rate", 0) > 
                self.telemetry_collector.slos.error_rate_threshold):
                violations.append({
                    "type": "error_rate_slo_violation",
                    "metric": "error_rate",
                    "threshold": self.telemetry_collector.slos.error_rate_threshold,
                    "actual": summary["error_rate"],
                    "timestamp": time.time()
                })
            
            self.slo_violations.extend(violations)
            return violations
        
        return []
    
    def get_violation_history(self) -> List[Dict[str, Any]]:
        """Get history of SLO violations."""
        return self.slo_violations[-100:]  # Last 100 violations


# Global performance monitor
_performance_monitor = PerformanceThresholdMonitor()


def get_performance_threshold_monitor() -> PerformanceThresholdMonitor:
    """Get global performance threshold monitor."""
    return _performance_monitor