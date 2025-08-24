"""
Performance markers and events for Cortex runtime
Tracks execution metrics and provides telemetry data
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from enum import Enum
import time
import uuid

from ..events import BaseEvent as Event


class CortexPhase(Enum):
    """Phases of the Cortex processing cycle"""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    ACTION = "action"
    INTEGRATION = "integration"


class MarkerType(Enum):
    """Types of performance markers"""
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    MODULE_EXECUTION = "module_execution"
    ERROR = "error"
    METRIC = "metric"


@dataclass
class PerformanceMarker:
    """Performance marker for tracking Cortex execution metrics"""
    id: str
    marker_type: MarkerType
    phase: Optional[CortexPhase]
    timestamp: float
    duration_ms: Optional[float] = None
    module_name: Optional[str] = None
    metrics: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CortexEvent:
    """Cortex-specific event that extends the base Event system"""
    cycle_id: str
    phase: Optional[CortexPhase]
    markers: list[PerformanceMarker]
    context: Dict[str, Any]
    base_event: Event
    
    @property
    def total_duration_ms(self) -> float:
        """Calculate total duration from markers"""
        if not self.markers:
            return 0.0
        
        start_markers = [m for m in self.markers if m.marker_type == MarkerType.CYCLE_START]
        end_markers = [m for m in self.markers if m.marker_type == MarkerType.CYCLE_END]
        
        if start_markers and end_markers:
            start_time = min(m.timestamp for m in start_markers)
            end_time = max(m.timestamp for m in end_markers)
            return (end_time - start_time) * 1000
        
        return 0.0
    
    @property
    def phase_durations(self) -> Dict[str, float]:
        """Calculate duration for each phase"""
        durations = {}
        
        for phase in CortexPhase:
            phase_markers = [m for m in self.markers 
                           if m.phase == phase and m.duration_ms is not None]
            if phase_markers:
                durations[phase.value] = sum(m.duration_ms for m in phase_markers)
        
        return durations


class PerformanceTracker:
    """Tracks performance metrics during Cortex execution"""
    
    def __init__(self):
        self.markers: list[PerformanceMarker] = []
        self.current_cycle_id: Optional[str] = None
        self.phase_start_times: Dict[CortexPhase, float] = {}
    
    def start_cycle(self, cycle_id: str) -> str:
        """Start tracking a new Cortex cycle"""
        self.current_cycle_id = cycle_id
        self.markers.clear()
        self.phase_start_times.clear()
        
        marker = PerformanceMarker(
            id=str(uuid.uuid4()),
            marker_type=MarkerType.CYCLE_START,
            phase=None,
            timestamp=time.time()
        )
        self.markers.append(marker)
        return marker.id
    
    def end_cycle(self) -> Optional[str]:
        """End the current cycle tracking"""
        if not self.current_cycle_id:
            return None
        
        marker = PerformanceMarker(
            id=str(uuid.uuid4()),
            marker_type=MarkerType.CYCLE_END,
            phase=None,
            timestamp=time.time()
        )
        self.markers.append(marker)
        
        cycle_id = self.current_cycle_id
        self.current_cycle_id = None
        return marker.id
    
    def start_phase(self, phase: CortexPhase) -> str:
        """Start tracking a Cortex phase"""
        start_time = time.time()
        self.phase_start_times[phase] = start_time
        
        marker = PerformanceMarker(
            id=str(uuid.uuid4()),
            marker_type=MarkerType.PHASE_START,
            phase=phase,
            timestamp=start_time
        )
        self.markers.append(marker)
        return marker.id
    
    def end_phase(self, phase: CortexPhase) -> Optional[str]:
        """End tracking a Cortex phase"""
        end_time = time.time()
        start_time = self.phase_start_times.get(phase)
        
        if start_time is None:
            return None
        
        duration_ms = (end_time - start_time) * 1000
        
        marker = PerformanceMarker(
            id=str(uuid.uuid4()),
            marker_type=MarkerType.PHASE_END,
            phase=phase,
            timestamp=end_time,
            duration_ms=duration_ms
        )
        self.markers.append(marker)
        
        del self.phase_start_times[phase]
        return marker.id
    
    def track_module_execution(
        self, 
        module_name: str, 
        phase: CortexPhase,
        duration_ms: float,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track module execution metrics"""
        marker = PerformanceMarker(
            id=str(uuid.uuid4()),
            marker_type=MarkerType.MODULE_EXECUTION,
            phase=phase,
            timestamp=time.time(),
            duration_ms=duration_ms,
            module_name=module_name,
            metrics=metrics or {}
        )
        self.markers.append(marker)
        return marker.id
    
    def track_error(
        self, 
        error: str, 
        phase: Optional[CortexPhase] = None,
        module_name: Optional[str] = None
    ) -> str:
        """Track an error during execution"""
        marker = PerformanceMarker(
            id=str(uuid.uuid4()),
            marker_type=MarkerType.ERROR,
            phase=phase,
            timestamp=time.time(),
            module_name=module_name,
            error=error
        )
        self.markers.append(marker)
        return marker.id
    
    def add_metric(
        self, 
        metric_name: str, 
        value: Union[int, float, str],
        phase: Optional[CortexPhase] = None
    ) -> str:
        """Add a custom metric"""
        marker = PerformanceMarker(
            id=str(uuid.uuid4()),
            marker_type=MarkerType.METRIC,
            phase=phase,
            timestamp=time.time(),
            metrics={metric_name: value}
        )
        self.markers.append(marker)
        return marker.id
    
    def get_markers(self) -> list[PerformanceMarker]:
        """Get all tracked markers"""
        return self.markers.copy()
    
    def clear(self):
        """Clear all tracking data"""
        self.markers.clear()
        self.current_cycle_id = None
        self.phase_start_times.clear()


def create_cortex_event(
    event_type: str,
    cycle_id: str,
    phase: Optional[CortexPhase] = None,
    markers: Optional[list[PerformanceMarker]] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> CortexEvent:
    """Create a Cortex event with performance tracking"""
    from ..events import create_event
    
    # Create base event with required source_plugin
    # Put Cortex-specific fields in metadata
    base_event = create_event(
        event_type, 
        source_plugin="cortex_runtime",
        metadata={
            "cycle_id": cycle_id,
            "phase": phase.value if phase else None,
            **kwargs
        }
    )
    
    # Create Cortex event
    cortex_event = CortexEvent(
        cycle_id=cycle_id,
        phase=phase,
        markers=markers or [],
        context=context or {},
        base_event=base_event
    )
    
    return cortex_event