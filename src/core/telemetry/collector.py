"""
Telemetry data collection and aggregation for Cortex runtime
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

from src.cortex.markers import PerformanceMarker
from src.events import BaseEvent


@dataclass
class TelemetrySnapshot:
    """Snapshot of telemetry data at a point in time"""
    timestamp: float
    cycle_id: str
    phase: str
    markers: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    system_metrics: Dict[str, Any]


class TelemetryCollector:
    """Collects and aggregates telemetry data from Cortex runtime"""
    
    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path("telemetry")
        self.output_dir.mkdir(exist_ok=True)
        
        self.snapshots: List[TelemetrySnapshot] = []
        self.active_markers: Dict[str, PerformanceMarker] = {}
        self.event_handlers: List[Callable[[BaseEvent], None]] = []
        self.collection_enabled = True
        self.buffer_size = 1000
        self.auto_flush_interval = 60.0  # seconds
        
        # Start auto-flush task
        self._flush_task: Optional[asyncio.Task] = None
        self._start_auto_flush()
    
    def _start_auto_flush(self) -> None:
        """Start the auto-flush background task"""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._auto_flush_loop())
    
    async def _auto_flush_loop(self) -> None:
        """Background task to periodically flush telemetry data"""
        while self.collection_enabled:
            try:
                await asyncio.sleep(self.auto_flush_interval)
                if len(self.snapshots) > 0:
                    await self.flush_to_disk()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in telemetry auto-flush: {e}")
    
    def add_event_handler(self, handler: Callable[[BaseEvent], None]) -> None:
        """Add an event handler for telemetry processing"""
        self.event_handlers.append(handler)
    
    def record_marker(self, marker: PerformanceMarker) -> None:
        """Record a performance marker"""
        if not self.collection_enabled:
            return
        
        try:
            marker_id = f"{marker.cycle_id}_{marker.phase}_{marker.marker_type}"
            self.active_markers[marker_id] = marker
        except Exception as e:
            print(f"Error recording marker: {e}")
    
    def record_event(self, event: BaseEvent) -> None:
        """Record an event for telemetry"""
        if not self.collection_enabled:
            return
        
        try:
            # Process with registered handlers
            for handler in self.event_handlers:
                handler(event)
            
            # Create snapshot if we have enough data
            if hasattr(event, 'cycle_id') and hasattr(event, 'phase'):
                self._create_snapshot_for_event(event)
        except Exception as e:
            print(f"Error recording event: {e}")
    
    def _create_snapshot_for_event(self, event: BaseEvent) -> None:
        """Create a telemetry snapshot for an event"""
        try:
            # Extract markers for this cycle/phase
            cycle_markers = []
            if hasattr(event, 'markers') and event.markers:
                for marker in event.markers:
                    if hasattr(marker, 'to_dict'):
                        cycle_markers.append(marker.to_dict())
            
            # Create snapshot
            snapshot = TelemetrySnapshot(
                timestamp=time.time(),
                cycle_id=getattr(event, 'cycle_id', 'unknown'),
                phase=getattr(event, 'phase', 'unknown'),
                markers=cycle_markers,
                events=[self._event_to_dict(event)],
                system_metrics=self._collect_system_metrics()
            )
            
            self.snapshots.append(snapshot)
            
            # Auto-flush if buffer is full
            if len(self.snapshots) >= self.buffer_size:
                asyncio.create_task(self.flush_to_disk())
                
        except Exception as e:
            print(f"Error creating snapshot: {e}")
    
    def _event_to_dict(self, event: BaseEvent) -> Dict[str, Any]:
        """Convert event to dictionary representation"""
        event_dict = {
            'type': type(event).__name__,
            'timestamp': time.time(),
        }
        
        # Add event attributes safely
        for attr in ['cycle_id', 'phase', 'data', 'metadata']:
            if hasattr(event, attr):
                value = getattr(event, attr)
                if value is not None:
                    event_dict[attr] = value
        
        return event_dict
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect basic system metrics"""
        return {
            'timestamp': time.time(),
            'active_markers_count': len(self.active_markers),
            'snapshots_count': len(self.snapshots),
        }
    
    async def flush_to_disk(self) -> None:
        """Flush collected telemetry data to disk"""
        if not self.snapshots:
            return
        
        try:
            timestamp = int(time.time())
            filename = f"telemetry_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert snapshots to serializable format
            data = {
                'snapshots': [asdict(snapshot) for snapshot in self.snapshots],
                'metadata': {
                    'collection_time': time.time(),
                    'total_snapshots': len(self.snapshots),
                }
            }
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Clear snapshots after successful write
            self.snapshots.clear()
            
        except Exception as e:
            print(f"Error flushing telemetry to disk: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected telemetry data"""
        return {
            'snapshots_count': len(self.snapshots),
            'active_markers_count': len(self.active_markers),
            'collection_enabled': self.collection_enabled,
            'output_directory': str(self.output_dir),
        }
    
    async def shutdown(self) -> None:
        """Shutdown the telemetry collector"""
        self.collection_enabled = False
        
        # Cancel auto-flush task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush_to_disk()