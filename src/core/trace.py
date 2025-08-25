from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import functools
from .correlation import get_correlation_id

@dataclass
class TraceEntry:
    timestamp: datetime
    correlation_id: str
    turn_index: int
    event_type: str
    component: str
    details: Dict[str, Any]
    duration_ms: Optional[float] = None

class TurnTracer:
    """
    Per-turn trace logger with standardized format:
    timestamp, correlation_id, turn_index, event_type, component, details, duration_ms (optional)
    """
    def __init__(self):
        self.traces: List[TraceEntry] = []
        self.current_turn = 0

    def new_turn(self):
        self.current_turn += 1

    def log(self, event_type: str, component: str, details: Dict[str, Any] | None = None, 
            duration_ms: float | None = None):
        entry = TraceEntry(
            timestamp=datetime.now(timezone.utc),
            correlation_id=get_correlation_id(),
            turn_index=self.current_turn,
            event_type=event_type,
            component=component,
            details=details or {},
            duration_ms=duration_ms
        )
        self.traces.append(entry)

    def get_turn_traces(self, turn_index: int) -> List[TraceEntry]:
        return [t for t in self.traces if t.turn_index == turn_index]

    def get_latest_traces(self, limit: int = 10) -> List[TraceEntry]:
        return self.traces[-limit:] if limit > 0 else self.traces[:]

# Global tracer instance
_tracer = TurnTracer()

def get_tracer() -> TurnTracer:
    return _tracer

def trace_component_call(component: str):
    """Decorator to automatically trace component function calls with timing"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = datetime.now(timezone.utc)
            try:
                result = await func(*args, **kwargs)
                end = datetime.now(timezone.utc)
                duration_ms = (end - start).total_seconds() * 1000
                _tracer.log("component_call", component, 
                           {"function": func.__name__, "success": True}, duration_ms)
                return result
            except Exception as e:
                end = datetime.now(timezone.utc)
                duration_ms = (end - start).total_seconds() * 1000
                _tracer.log("component_call", component, 
                           {"function": func.__name__, "success": False, "error": str(e)}, duration_ms)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = datetime.now(timezone.utc)
            try:
                result = func(*args, **kwargs)
                end = datetime.now(timezone.utc)
                duration_ms = (end - start).total_seconds() * 1000
                _tracer.log("component_call", component, 
                           {"function": func.__name__, "success": True}, duration_ms)
                return result
            except Exception as e:
                end = datetime.now(timezone.utc)
                duration_ms = (end - start).total_seconds() * 1000
                _tracer.log("component_call", component, 
                           {"function": func.__name__, "success": False, "error": str(e)}, duration_ms)
                raise
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator