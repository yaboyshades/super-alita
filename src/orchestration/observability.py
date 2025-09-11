"""
Observability and Logging for Unified Orchestrator and SDD Workflow
Provides structured logging, metrics, and event schemas for monitoring.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class LogLevel(Enum):
    """Log levels for orchestrator events."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Event types for orchestrator monitoring."""

    RUN_STARTED = "UnifiedRunStarted"
    RUN_COMPLETED = "UnifiedRunCompleted"
    RUN_FAILED = "UnifiedRunFailed"
    STAGE_STARTED = "UnifiedStageStarted"
    STAGE_SUCCEEDED = "UnifiedStageSucceeded"
    STAGE_FAILED = "UnifiedStageFailed"
    SDD_VALIDATION_STARTED = "SDDValidationStarted"
    SDD_VALIDATION_COMPLETED = "SDDValidationCompleted"
    CONSTITUTIONAL_GATE_CHECK = "ConstitutionalGateCheck"
    CONSTITUTIONAL_VIOLATION = "ConstitutionalViolation"


@dataclass
class MetricData:
    """Structured metric data."""

    name: str
    value: float | int
    unit: str
    timestamp: float
    tags: dict[str, str]
    run_id: str
    session_id: str


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: float
    level: LogLevel
    message: str
    event_type: EventType | None
    run_id: str
    session_id: str
    stage: str | None
    data: dict[str, Any]
    duration_ms: int | None = None
    error: str | None = None


class OrchestatorObserver:
    """Observability layer for unified orchestrator."""

    def __init__(self, logger_name: str = "unified_orchestrator"):
        self.logger = logging.getLogger(logger_name)
        self.metrics = []
        self.log_entries = []
        self._setup_logger()

    def _setup_logger(self):
        """Configure structured logging."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_run_started(self, config: dict[str, Any]) -> None:
        """Log orchestrator run start."""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            message=f"Orchestrator run started: {config.get('run_id')}",
            event_type=EventType.RUN_STARTED,
            run_id=config.get("run_id", "unknown"),
            session_id=config.get("session_id", "default"),
            stage=None,
            data={
                "prompt": config.get("prompt", ""),
                "sdd_mode": config.get("sdd_mode", False),
                "enabled_stages": {
                    k: v for k, v in config.items() if k.startswith("enable_")
                },
            },
        )
        self._emit_log(entry)

    def log_run_completed(
        self, run_id: str, session_id: str, stages: dict[str, Any]
    ) -> None:
        """Log orchestrator run completion."""
        success_count = sum(
            1 for stage in stages.values() if stage.get("status") == "success"
        )
        total_duration = sum(stage.get("duration_ms", 0) for stage in stages.values())

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            message=f"Orchestrator run completed: {run_id}",
            event_type=EventType.RUN_COMPLETED,
            run_id=run_id,
            session_id=session_id,
            stage=None,
            duration_ms=total_duration,
            data={
                "stages_executed": len(stages),
                "stages_successful": success_count,
                "success_rate": success_count / len(stages) if stages else 0,
                "stage_summary": {
                    name: {
                        "status": stage.get("status"),
                        "duration_ms": stage.get("duration_ms"),
                    }
                    for name, stage in stages.items()
                },
            },
        )
        self._emit_log(entry)

        # Emit metrics
        self._emit_metric(
            "run_duration_ms", total_duration, "milliseconds", run_id, session_id
        )
        self._emit_metric("stages_executed", len(stages), "count", run_id, session_id)
        self._emit_metric(
            "success_rate",
            success_count / len(stages) if stages else 0,
            "ratio",
            run_id,
            session_id,
        )

    def log_stage_started(self, run_id: str, session_id: str, stage: str) -> None:
        """Log stage start."""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            message=f"Stage started: {stage}",
            event_type=EventType.STAGE_STARTED,
            run_id=run_id,
            session_id=session_id,
            stage=stage,
            data={"stage_type": self._classify_stage(stage)},
        )
        self._emit_log(entry)

    def log_stage_succeeded(
        self,
        run_id: str,
        session_id: str,
        stage: str,
        duration_ms: int,
        output_summary: dict[str, Any],
    ) -> None:
        """Log stage success."""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            message=f"Stage succeeded: {stage} ({duration_ms}ms)",
            event_type=EventType.STAGE_SUCCEEDED,
            run_id=run_id,
            session_id=session_id,
            stage=stage,
            duration_ms=duration_ms,
            data={
                "stage_type": self._classify_stage(stage),
                "output_summary": output_summary,
            },
        )
        self._emit_log(entry)

        # Emit metrics
        self._emit_metric(
            f"stage_{stage}_duration_ms",
            duration_ms,
            "milliseconds",
            run_id,
            session_id,
        )
        self._emit_metric(f"stage_{stage}_success", 1, "count", run_id, session_id)

    def log_stage_failed(
        self, run_id: str, session_id: str, stage: str, duration_ms: int, error: str
    ) -> None:
        """Log stage failure."""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.ERROR,
            message=f"Stage failed: {stage} ({duration_ms}ms) - {error}",
            event_type=EventType.STAGE_FAILED,
            run_id=run_id,
            session_id=session_id,
            stage=stage,
            duration_ms=duration_ms,
            error=error,
            data={
                "stage_type": self._classify_stage(stage),
                "error_type": (
                    type(error).__name__ if hasattr(error, "__class__") else "unknown"
                ),
            },
        )
        self._emit_log(entry)

        # Emit metrics
        self._emit_metric(f"stage_{stage}_failure", 1, "count", run_id, session_id)

    def log_sdd_validation_started(
        self, run_id: str, session_id: str, phase: str, content_summary: str
    ) -> None:
        """Log SDD validation start."""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            message=f"SDD validation started: {phase}",
            event_type=EventType.SDD_VALIDATION_STARTED,
            run_id=run_id,
            session_id=session_id,
            stage=f"{phase}_validation",
            data={
                "sdd_phase": phase,
                "content_length": len(content_summary),
                "validation_type": "constitutional",
            },
        )
        self._emit_log(entry)

    def log_sdd_validation_completed(
        self, run_id: str, session_id: str, phase: str, results: dict[str, Any]
    ) -> None:
        """Log SDD validation completion."""
        compliance_score = results.get("overall_score", 0)
        passed = results.get("passed", False)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO if passed else LogLevel.WARNING,
            message=f"SDD validation completed: {phase} (score: {compliance_score:.2f})",
            event_type=EventType.SDD_VALIDATION_COMPLETED,
            run_id=run_id,
            session_id=session_id,
            stage=f"{phase}_validation",
            data={
                "sdd_phase": phase,
                "compliance_score": compliance_score,
                "passed": passed,
                "validation_details": {
                    k: v for k, v in results.items() if k != "overall_score"
                },
            },
        )
        self._emit_log(entry)

        # Emit metrics
        self._emit_metric(
            f"sdd_{phase}_compliance_score",
            compliance_score,
            "score",
            run_id,
            session_id,
        )
        self._emit_metric(
            f"sdd_{phase}_validation_pass",
            1 if passed else 0,
            "boolean",
            run_id,
            session_id,
        )

    def log_constitutional_gate_check(
        self,
        run_id: str,
        session_id: str,
        gate_name: str,
        threshold: float,
        actual_score: float,
        passed: bool,
    ) -> None:
        """Log constitutional gate check."""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO if passed else LogLevel.WARNING,
            message=f"Constitutional gate {gate_name}: {actual_score:.2f} (threshold: {threshold:.2f})",
            event_type=EventType.CONSTITUTIONAL_GATE_CHECK,
            run_id=run_id,
            session_id=session_id,
            stage=gate_name,
            data={
                "gate_name": gate_name,
                "threshold": threshold,
                "actual_score": actual_score,
                "passed": passed,
                "margin": actual_score - threshold,
            },
        )
        self._emit_log(entry)

        # Emit metrics
        self._emit_metric(
            f"constitutional_gate_{gate_name}_score",
            actual_score,
            "score",
            run_id,
            session_id,
        )
        self._emit_metric(
            f"constitutional_gate_{gate_name}_pass",
            1 if passed else 0,
            "boolean",
            run_id,
            session_id,
        )

    def log_constitutional_violation(
        self,
        run_id: str,
        session_id: str,
        article: str,
        violation_details: dict[str, Any],
    ) -> None:
        """Log constitutional violation."""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.ERROR,
            message=f"Constitutional violation: {article}",
            event_type=EventType.CONSTITUTIONAL_VIOLATION,
            run_id=run_id,
            session_id=session_id,
            stage=None,
            data={
                "violated_article": article,
                "violation_severity": violation_details.get("severity", "unknown"),
                "violation_details": violation_details,
            },
        )
        self._emit_log(entry)

        # Emit metrics
        self._emit_metric(
            f"constitutional_violation_{article}", 1, "count", run_id, session_id
        )

    def _classify_stage(self, stage: str) -> str:
        """Classify stage type for analytics."""
        if "validation" in stage:
            return "validation"
        elif stage in ["specification", "planning", "tasks"]:
            return "sdd_workflow"
        elif stage in ["consensus", "code_generation", "scoring"]:
            return "generation"
        else:
            return "other"

    def _emit_log(self, entry: LogEntry) -> None:
        """Emit structured log entry."""
        self.log_entries.append(entry)

        # Format for standard logger
        log_data = {
            "event_type": entry.event_type.value if entry.event_type else None,
            "run_id": entry.run_id,
            "session_id": entry.session_id,
            "stage": entry.stage,
            "duration_ms": entry.duration_ms,
            "data": entry.data,
        }

        if entry.error:
            log_data["error"] = entry.error

        formatted_message = f"{entry.message} | {json.dumps(log_data)}"

        if entry.level == LogLevel.DEBUG:
            self.logger.debug(formatted_message)
        elif entry.level == LogLevel.INFO:
            self.logger.info(formatted_message)
        elif entry.level == LogLevel.WARNING:
            self.logger.warning(formatted_message)
        elif entry.level == LogLevel.ERROR:
            self.logger.error(formatted_message)
        elif entry.level == LogLevel.CRITICAL:
            self.logger.critical(formatted_message)

    def _emit_metric(
        self,
        name: str,
        value: float | int,
        unit: str,
        run_id: str,
        session_id: str,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Emit metric data."""
        metric = MetricData(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            tags=tags or {},
            run_id=run_id,
            session_id=session_id,
        )
        self.metrics.append(metric)

    def get_metrics_for_run(self, run_id: str) -> list[MetricData]:
        """Get metrics for specific run."""
        return [m for m in self.metrics if m.run_id == run_id]

    def get_logs_for_run(self, run_id: str) -> list[LogEntry]:
        """Get logs for specific run."""
        return [log for log in self.log_entries if log.run_id == run_id]

    def export_run_report(self, run_id: str) -> dict[str, Any]:
        """Export comprehensive run report."""
        logs = self.get_logs_for_run(run_id)
        metrics = self.get_metrics_for_run(run_id)

        return {
            "run_id": run_id,
            "summary": {
                "total_logs": len(logs),
                "total_metrics": len(metrics),
                "log_levels": {
                    level.value: len([log for log in logs if log.level == level])
                    for level in LogLevel
                },
                "event_types": {
                    event_type.value: len(
                        [log for log in logs if log.event_type == event_type]
                    )
                    for event_type in EventType
                },
            },
            "logs": [asdict(log) for log in logs],
            "metrics": [asdict(metric) for metric in metrics],
            "generated_at": time.time(),
        }


# Global observer instance
_global_observer: OrchestatorObserver | None = None


def get_observer() -> OrchestatorObserver:
    """Get or create global observer instance."""
    global _global_observer
    if _global_observer is None:
        _global_observer = OrchestatorObserver()
    return _global_observer


def setup_observability(
    logger_name: str = "unified_orchestrator",
) -> OrchestatorObserver:
    """Setup orchestrator observability."""
    global _global_observer
    _global_observer = OrchestatorObserver(logger_name)
    return _global_observer
