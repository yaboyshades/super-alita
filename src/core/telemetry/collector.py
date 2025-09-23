"""
Telemetry data collection and aggregation for Cortex runtime
"""

import asyncio
import contextlib
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.cortex.markers import PerformanceMarker
from src.events import BaseEvent


@dataclass
class TelemetryEvent:
    """Telemetry event for compatibility with grpc server expectations"""

    timestamp: float
    event_type: str
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TelemetrySnapshot:
    """Snapshot of telemetry data at a point in time"""

    timestamp: float
    cycle_id: str
    phase: str
    markers: list[dict[str, Any]]
    events: list[dict[str, Any]]
    system_metrics: dict[str, Any]


class TelemetryCollector:
    """Collects and aggregates telemetry data from Cortex runtime"""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Path("telemetry")
        self.output_dir.mkdir(exist_ok=True)

        self.snapshots: list[TelemetrySnapshot] = []
        self.active_markers: dict[str, PerformanceMarker] = {}
        self.event_handlers: list[Callable[[BaseEvent], None]] = []
        self.collection_enabled = True
        self.buffer_size = 1000
        self.auto_flush_interval = 60.0  # seconds
        self._current_canonical_runs: dict[str, dict[str, Any]] = {}
        self.canonical_history: deque[dict[str, Any]] = deque(maxlen=50)

        # Start auto-flush task
        self._flush_task: asyncio.Task | None = None
        self._start_auto_flush()

    def _start_auto_flush(self) -> None:
        """Start the auto-flush background task"""
        if self._flush_task is not None and not self._flush_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._flush_task = None
            return
        self._flush_task = loop.create_task(self._auto_flush_loop())

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
            if hasattr(event, "cycle_id") and hasattr(event, "phase"):
                self._create_snapshot_for_event(event)
        except Exception as e:
            print(f"Error recording event: {e}")

    def record_canonical_event(self, event: dict[str, Any]) -> None:
        """Aggregate canonical orchestrator events into compact summaries."""
        if not self.collection_enabled or not isinstance(event, dict):
            return
        run_id = event.get("run_id")
        kind = event.get("kind")
        if not run_id or not kind:
            return

        stats = self._current_canonical_runs.setdefault(
            run_id,
            {
                "run_id": run_id,
                "counts": {"events": 0, "abilities": 0},
                "stages": {},
                "errors": [],
                "started_at": event.get("timestamp"),
            },
        )
        stats["counts"]["events"] += 1
        data = event.get("data") or {}

        if kind == "RunStarted":
            stats["started_at"] = event.get("timestamp")
        elif kind == "StageCompleted":
            stage_name = data.get("name", "unknown")
            stage_info = stats["stages"].setdefault(stage_name, {
                "duration_ms": 0,
                "status": "started",
            })
            stage_info["duration_ms"] = int(data.get("duration_ms", 0))
            stage_info["status"] = data.get("status", "ok")
        elif kind in {"AbilityInvocationStarted", "AbilityInvocationCompleted"}:
            stats["counts"]["abilities"] += 1
        elif kind == "RunError":
            error_type = data.get("error_type") or "UNKNOWN"
            stats["errors"].append(error_type)
        elif kind in {"RunTerminated", "RunFailed"}:
            summary = {
                "run_id": run_id,
                "success": bool(data.get("success", kind == "RunTerminated")),
                "total_duration_ms": data.get("total_duration_ms"),
                "stages": stats["stages"],
                "errors": stats["errors"],
                "abilities_invoked": data.get("abilities_invoked", stats["counts"]["abilities"]),
                "constitutional_score": event.get("constitutional_score"),
            }
            self.canonical_history.append(summary)
            self._current_canonical_runs.pop(run_id, None)

    def _create_snapshot_for_event(self, event: BaseEvent) -> None:
        """Create a telemetry snapshot for an event"""
        try:
            # Extract markers for this cycle/phase
            cycle_markers = []
            if hasattr(event, "markers") and event.markers:
                for marker in event.markers:
                    if hasattr(marker, "to_dict"):
                        cycle_markers.append(marker.to_dict())

            # Create snapshot
            snapshot = TelemetrySnapshot(
                timestamp=time.time(),
                cycle_id=getattr(event, "cycle_id", "unknown"),
                phase=getattr(event, "phase", "unknown"),
                markers=cycle_markers,
                events=[self._event_to_dict(event)],
                system_metrics=self._collect_system_metrics(),
            )

            self.snapshots.append(snapshot)

            # Auto-flush if buffer is full
            if len(self.snapshots) >= self.buffer_size:
                asyncio.create_task(self.flush_to_disk())

        except Exception as e:
            print(f"Error creating snapshot: {e}")

    def _event_to_dict(self, event: BaseEvent) -> dict[str, Any]:
        """Convert event to dictionary representation"""
        event_dict = {
            "type": type(event).__name__,
            "timestamp": time.time(),
        }

        # Add event attributes safely
        for attr in ["cycle_id", "phase", "data", "metadata"]:
            if hasattr(event, attr):
                value = getattr(event, attr)
                if value is not None:
                    event_dict[attr] = value

        return event_dict

    def _collect_system_metrics(self) -> dict[str, Any]:
        """Collect basic system metrics"""
        return {
            "timestamp": time.time(),
            "active_markers_count": len(self.active_markers),
            "snapshots_count": len(self.snapshots),
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
                "snapshots": [asdict(snapshot) for snapshot in self.snapshots],
                "metadata": {
                    "collection_time": time.time(),
                    "total_snapshots": len(self.snapshots),
                },
            }

            # Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

            # Clear snapshots after successful write
            self.snapshots.clear()

        except Exception as e:
            print(f"Error flushing telemetry to disk: {e}")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of collected telemetry data"""
        return {
            "snapshots_count": len(self.snapshots),
            "active_markers_count": len(self.active_markers),
            "collection_enabled": self.collection_enabled,
            "output_directory": str(self.output_dir),
        }

    async def shutdown(self) -> None:
        """Shutdown the telemetry collector"""
        self.collection_enabled = False

        # Cancel auto-flush task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task

        # Final flush
        await self.flush_to_disk()
