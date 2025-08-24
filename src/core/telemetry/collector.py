"""
Telemetry data collection and aggregation for Cortex runtime
"""

import asyncio
import json
import time
from datetime import datetime, UTC
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

from ..cortex.markers import PerformanceMarker, CortexPhase, MarkerType
from ..events import BaseEvent


@dataclass
class TelemetryEvent:
    """Individual telemetry event"""
    event_id: str
    timestamp: float
    event_type: str
    source: str
    phase: Optional[str] = None
    duration_ms: Optional[float] = None
    cycle_id: Optional[str] = None
    markers: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.markers is None:
            self.markers = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass 
class TelemetryMetrics:
    """Aggregated telemetry metrics"""
    total_cycles: int = 0
    total_events: int = 0
    avg_cycle_duration_ms: float = 0.0
    avg_perception_duration_ms: float = 0.0
    avg_reasoning_duration_ms: float = 0.0
    avg_action_duration_ms: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    active_cycles: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return asdict(self)


class TelemetryCollector:
    """
    Collects and aggregates telemetry data from Cortex runtime
    """
    
    def __init__(self, output_file: Optional[Path] = None):
        self.output_file = output_file or Path("telemetry.jsonl")
        self.events: List[TelemetryEvent] = []
        self.metrics = TelemetryMetrics()
        self.active_cycles: Dict[str, float] = {}
        self.phase_durations: Dict[str, List[float]] = {
            "perception": [],
            "reasoning": [], 
            "action": []
        }
        self.subscribers: List[Callable[[TelemetryEvent], None]] = []
        self._lock = asyncio.Lock()
        
    async def collect_event(self, event: BaseEvent) -> None:
        """Collect a telemetry event from the event bus"""
        async with self._lock:
            telemetry_event = TelemetryEvent(
                event_id=event.event_id,
                timestamp=time.time(),
                event_type=event.event_type,
                source=event.source_plugin,
                metadata=event.metadata
            )
            
            # Extract Cortex-specific data
            if hasattr(event, 'cycle_id'):
                telemetry_event.cycle_id = event.cycle_id
            
            if hasattr(event, 'phase'):
                telemetry_event.phase = event.phase.value if event.phase else None
                
            if hasattr(event, 'markers'):
                telemetry_event.markers = [
                    marker.to_dict() for marker in (event.markers or [])
                ]
                
            self.events.append(telemetry_event)
            await self._update_metrics(telemetry_event)
            await self._persist_event(telemetry_event)
            
            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber(telemetry_event)
                except Exception as e:
                    print(f"Error notifying telemetry subscriber: {e}")
    
    async def collect_marker(self, marker: PerformanceMarker) -> None:
        """Collect a performance marker directly"""
        async with self._lock:
            telemetry_event = TelemetryEvent(
                event_id=marker.id,
                timestamp=marker.timestamp,
                event_type="performance_marker",
                source="cortex_performance_tracker",
                phase=marker.phase.value if marker.phase else None,
                duration_ms=marker.duration_ms,
                markers=[marker.to_dict()],
                metadata=marker.metadata or {}
            )
            
            self.events.append(telemetry_event)
            await self._update_metrics(telemetry_event)
            await self._persist_event(telemetry_event)
            
            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber(telemetry_event)
                except Exception as e:
                    print(f"Error notifying telemetry subscriber: {e}")
    
    async def _update_metrics(self, event: TelemetryEvent) -> None:
        """Update aggregated metrics"""
        self.metrics.total_events += 1
        
        # Track cycle lifecycle
        if event.event_type == "cognitive_cycle":
            if event.cycle_id:
                if "start" in event.metadata.get("status", ""):
                    self.active_cycles[event.cycle_id] = event.timestamp
                    self.metrics.active_cycles = len(self.active_cycles)
                elif "end" in event.metadata.get("status", ""):
                    if event.cycle_id in self.active_cycles:
                        start_time = self.active_cycles.pop(event.cycle_id)
                        cycle_duration = (event.timestamp - start_time) * 1000
                        self._update_cycle_metrics(cycle_duration)
                        self.metrics.active_cycles = len(self.active_cycles)
        
        # Track phase durations from markers
        if event.phase and event.duration_ms:
            phase_key = event.phase.lower()
            if phase_key in self.phase_durations:
                self.phase_durations[phase_key].append(event.duration_ms)
                self._recalculate_phase_averages()
        
        # Track errors
        if "error" in event.event_type.lower() or event.metadata.get("success") is False:
            self.metrics.error_count += 1
        
        # Calculate success rate
        if self.metrics.total_cycles > 0:
            self.metrics.success_rate = max(0, 1.0 - (self.metrics.error_count / self.metrics.total_cycles))
    
    def _update_cycle_metrics(self, duration_ms: float) -> None:
        """Update cycle-related metrics"""
        self.metrics.total_cycles += 1
        
        # Calculate running average for cycle duration
        if self.metrics.total_cycles == 1:
            self.metrics.avg_cycle_duration_ms = duration_ms
        else:
            alpha = 2.0 / (self.metrics.total_cycles + 1)  # Exponential moving average
            self.metrics.avg_cycle_duration_ms = (
                alpha * duration_ms + 
                (1 - alpha) * self.metrics.avg_cycle_duration_ms
            )
    
    def _recalculate_phase_averages(self) -> None:
        """Recalculate phase duration averages"""
        if self.phase_durations["perception"]:
            self.metrics.avg_perception_duration_ms = sum(self.phase_durations["perception"]) / len(self.phase_durations["perception"])
        
        if self.phase_durations["reasoning"]:
            self.metrics.avg_reasoning_duration_ms = sum(self.phase_durations["reasoning"]) / len(self.phase_durations["reasoning"])
        
        if self.phase_durations["action"]:
            self.metrics.avg_action_duration_ms = sum(self.phase_durations["action"]) / len(self.phase_durations["action"])
    
    async def _persist_event(self, event: TelemetryEvent) -> None:
        """Persist event to JSONL file"""
        try:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            print(f"Error persisting telemetry event: {e}")
    
    def subscribe(self, callback: Callable[[TelemetryEvent], None]) -> None:
        """Subscribe to telemetry events"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[TelemetryEvent], None]) -> None:
        """Unsubscribe from telemetry events"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def get_metrics(self) -> TelemetryMetrics:
        """Get current aggregated metrics"""
        return self.metrics
    
    def get_recent_events(self, limit: int = 100) -> List[TelemetryEvent]:
        """Get recent telemetry events"""
        return self.events[-limit:] if len(self.events) > limit else self.events
    
    def get_events_by_cycle(self, cycle_id: str) -> List[TelemetryEvent]:
        """Get all events for a specific cycle"""
        return [event for event in self.events if event.cycle_id == cycle_id]
    
    def get_phase_statistics(self, phase: str) -> Dict[str, float]:
        """Get statistics for a specific phase"""
        if phase.lower() not in self.phase_durations:
            return {}
        
        durations = self.phase_durations[phase.lower()]
        if not durations:
            return {}
        
        return {
            "count": len(durations),
            "average_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "total_ms": sum(durations)
        }
    
    async def clear_old_events(self, keep_last: int = 10000) -> None:
        """Clear old events to prevent memory growth"""
        async with self._lock:
            if len(self.events) > keep_last:
                self.events = self.events[-keep_last:]
                print(f"Cleared old telemetry events, keeping {keep_last} most recent")