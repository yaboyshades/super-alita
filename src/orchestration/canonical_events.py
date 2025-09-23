from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Any

CANONICAL_VERSION = "v1"


class EventKind(str, Enum):
    """Canonical event discriminator enumerated in the P0 schema."""

    RUN_STARTED = "RunStarted"
    STAGE_STARTED = "StageStarted"
    STAGE_COMPLETED = "StageCompleted"
    ABILITY_INVOCATION_STARTED = "AbilityInvocationStarted"
    ABILITY_INVOCATION_CHUNK = "AbilityInvocationChunk"
    ABILITY_INVOCATION_COMPLETED = "AbilityInvocationCompleted"
    RUN_LOG = "RunLog"
    RUN_ERROR = "RunError"
    RUN_TERMINATED = "RunTerminated"
    RUN_FAILED = "RunFailed"


@dataclass(slots=True)
class CanonicalEvent:
    """Base envelope for canonical events."""

    kind: EventKind
    run_id: str
    sequence: int
    timestamp: str
    correlation_id: str
    parent_correlation_id: str | None = None
    stage: str | None = None
    trace_id: str | None = None
    constitutional_score: float | None = None
    meta: Mapping[str, Any] | None = None
    data: Mapping[str, Any] = field(default_factory=dict)
    version: str = CANONICAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kind": self.kind.value,
            "run_id": self.run_id,
            "sequence": int(self.sequence),
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "parent_correlation_id": self.parent_correlation_id,
            "stage": self.stage,
            "trace_id": self.trace_id,
            "constitutional_score": self.constitutional_score,
            "meta": dict(self.meta or {}),
            "data": _coerce_mapping(self.data),
        }

    def with_sequence(self, sequence: int) -> CanonicalEvent:
        return replace(self, sequence=sequence)


@dataclass(slots=True)
class RunStartedData:
    input_summary: str
    config: Mapping[str, Any]


@dataclass(slots=True)
class StageStartedData:
    name: str
    index: int


@dataclass(slots=True)
class StageCompletedData:
    name: str
    index: int
    duration_ms: int
    output_summary: str | None
    status: str


@dataclass(slots=True)
class AbilityInvocationStartedData:
    ability: str
    args_hash: str


@dataclass(slots=True)
class AbilityInvocationChunkData:
    ability: str
    chunk: str
    index: int
    is_final: bool


@dataclass(slots=True)
class AbilityInvocationCompletedData:
    ability: str
    duration_ms: int
    result_preview: str | None
    status: str
    error_type: str | None


@dataclass(slots=True)
class RunLogData:
    level: str
    message: str
    context: Mapping[str, Any]


@dataclass(slots=True)
class RunErrorData:
    scope: str
    stage: str | None
    ability: str | None
    error_type: str
    message: str
    retryable: bool


@dataclass(slots=True)
class RunTerminatedData:
    success: bool
    total_duration_ms: int
    stages_executed: int
    abilities_invoked: int
    final_output_preview: str | None


@dataclass(slots=True)
class RunFailedData:
    fatal_error_type: str
    message: str
    last_stage: str | None
    total_duration_ms: int


def _timestamp_string(raw: str | datetime | None) -> str:
    if raw is None:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(raw, datetime):
        return raw.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return raw


def _build_event(
    kind: EventKind,
    run_id: str,
    sequence: int,
    timestamp: str | datetime | None,
    correlation_id: str,
    parent_correlation_id: str | None,
    stage: str | None,
    trace_id: str | None,
    constitutional_score: float | None,
    meta: Mapping[str, Any] | None,
    data: Mapping[str, Any],
) -> CanonicalEvent:
    return CanonicalEvent(
        kind=kind,
        run_id=run_id,
        sequence=sequence,
        timestamp=_timestamp_string(timestamp),
        correlation_id=correlation_id,
        parent_correlation_id=parent_correlation_id,
        stage=stage,
        trace_id=trace_id,
        constitutional_score=constitutional_score,
        meta=meta,
        data=data,
    )


def make_run_started_event(
    *,
    run_id: str,
    sequence: int,
    timestamp: str | datetime,
    correlation_id: str,
    parent_correlation_id: str | None,
    stage: str | None,
    trace_id: str | None,
    constitutional_score: float | None,
    meta: Mapping[str, Any] | None,
    input_summary: str,
    config: Mapping[str, Any],
) -> CanonicalEvent:
    data = RunStartedData(input_summary=input_summary, config=config)
    return _build_event(
        EventKind.RUN_STARTED,
        run_id,
        sequence,
        timestamp,
        correlation_id,
        parent_correlation_id,
        stage,
        trace_id,
        constitutional_score,
        meta,
        asdict(data),
    )


def make_stage_started_event(
    *,
    run_id: str,
    sequence: int,
    timestamp: str | datetime,
    correlation_id: str,
    parent_correlation_id: str | None,
    stage: str | None,
    trace_id: str | None,
    constitutional_score: float | None,
    meta: Mapping[str, Any] | None,
    name: str,
    index: int,
) -> CanonicalEvent:
    data = StageStartedData(name=name, index=index)
    return _build_event(
        EventKind.STAGE_STARTED,
        run_id,
        sequence,
        timestamp,
        correlation_id,
        parent_correlation_id,
        stage,
        trace_id,
        constitutional_score,
        meta,
        asdict(data),
    )


def make_stage_completed_event(
    *,
    run_id: str,
    sequence: int,
    timestamp: str | datetime,
    correlation_id: str,
    parent_correlation_id: str | None,
    stage: str | None,
    trace_id: str | None,
    constitutional_score: float | None,
    meta: Mapping[str, Any] | None,
    name: str,
    index: int,
    duration_ms: int,
    output_summary: str | None,
    status: str,
) -> CanonicalEvent:
    data = StageCompletedData(
        name=name,
        index=index,
        duration_ms=duration_ms,
        output_summary=output_summary,
        status=status,
    )
    return _build_event(
        EventKind.STAGE_COMPLETED,
        run_id,
        sequence,
        timestamp,
        correlation_id,
        parent_correlation_id,
        stage,
        trace_id,
        constitutional_score,
        meta,
        asdict(data),
    )


def make_ability_invocation_started_event(
    *,
    run_id: str,
    sequence: int,
    timestamp: str | datetime,
    correlation_id: str,
    parent_correlation_id: str | None,
    stage: str | None,
    trace_id: str | None,
    constitutional_score: float | None,
    meta: Mapping[str, Any] | None,
    ability: str,
    args_hash: str,
) -> CanonicalEvent:
    data = AbilityInvocationStartedData(ability=ability, args_hash=args_hash)
    return _build_event(
        EventKind.ABILITY_INVOCATION_STARTED,
        run_id,
        sequence,
        timestamp,
        correlation_id,
        parent_correlation_id,
        stage,
        trace_id,
        constitutional_score,
        meta,
        asdict(data),
    )


def make_ability_invocation_chunk_event(
    *,
    run_id: str,
    sequence: int,
    timestamp: str | datetime,
    correlation_id: str,
    parent_correlation_id: str | None,
    stage: str | None,
    trace_id: str | None,
    constitutional_score: float | None,
    meta: Mapping[str, Any] | None,
    ability: str,
    chunk: str,
    index: int,
    is_final: bool,
) -> CanonicalEvent:
    data = AbilityInvocationChunkData(
        ability=ability,
        chunk=chunk,
        index=index,
        is_final=is_final,
    )
    return _build_event(
        EventKind.ABILITY_INVOCATION_CHUNK,
        run_id,
        sequence,
        timestamp,
        correlation_id,
        parent_correlation_id,
        stage,
        trace_id,
        constitutional_score,
        meta,
        asdict(data),
    )


def make_ability_invocation_completed_event(
    *,
    run_id: str,
    sequence: int,
    timestamp: str | datetime,
    correlation_id: str,
    parent_correlation_id: str | None,
    stage: str | None,
    trace_id: str | None,
    constitutional_score: float | None,
    meta: Mapping[str, Any] | None,
    ability: str,
    duration_ms: int,
    result_preview: str | None,
    status: str,
    error_type: str | None,
) -> CanonicalEvent:
    data = AbilityInvocationCompletedData(
        ability=ability,
        duration_ms=duration_ms,
        result_preview=result_preview,
        status=status,
        error_type=error_type,
    )
    return _build_event(
        EventKind.ABILITY_INVOCATION_COMPLETED,
        run_id,
        sequence,
        timestamp,
        correlation_id,
        parent_correlation_id,
        stage,
        trace_id,
        constitutional_score,
        meta,
        asdict(data),
    )


def make_run_log_event(
    *,
    run_id: str,
    sequence: int,
    timestamp: str | datetime,
    correlation_id: str,
    parent_correlation_id: str | None,
    stage: str | None,
    trace_id: str | None,
    constitutional_score: float | None,
    meta: Mapping[str, Any] | None,
    level: str,
    message: str,
    context: Mapping[str, Any],
) -> CanonicalEvent:
    data = RunLogData(level=level, message=message, context=context)
    return _build_event(
        EventKind.RUN_LOG,
        run_id,
        sequence,
        timestamp,
        correlation_id,
        parent_correlation_id,
        stage,
        trace_id,
        constitutional_score,
        meta,
        asdict(data),
    )


def make_run_error_event(
    *,
    run_id: str,
    sequence: int,
    timestamp: str | datetime,
    correlation_id: str,
    parent_correlation_id: str | None,
    stage: str | None,
    trace_id: str | None,
    constitutional_score: float | None,
    meta: Mapping[str, Any] | None,
    scope: str,
    stage_name: str | None,
    ability: str | None,
    error_type: str,
    message: str,
    retryable: bool,
) -> CanonicalEvent:
    data = RunErrorData(
        scope=scope,
        stage=stage_name,
        ability=ability,
        error_type=error_type,
        message=message,
        retryable=retryable,
    )
    return _build_event(
        EventKind.RUN_ERROR,
        run_id,
        sequence,
        timestamp,
        correlation_id,
        parent_correlation_id,
        stage,
        trace_id,
        constitutional_score,
        meta,
        asdict(data),
    )


def make_run_terminated_event(
    *,
    run_id: str,
    sequence: int,
    timestamp: str | datetime,
    correlation_id: str,
    parent_correlation_id: str | None,
    stage: str | None,
    trace_id: str | None,
    constitutional_score: float | None,
    meta: Mapping[str, Any] | None,
    success: bool,
    total_duration_ms: int,
    stages_executed: int,
    abilities_invoked: int,
    final_output_preview: str | None,
) -> CanonicalEvent:
    data = RunTerminatedData(
        success=success,
        total_duration_ms=total_duration_ms,
        stages_executed=stages_executed,
        abilities_invoked=abilities_invoked,
        final_output_preview=final_output_preview,
    )
    return _build_event(
        EventKind.RUN_TERMINATED,
        run_id,
        sequence,
        timestamp,
        correlation_id,
        parent_correlation_id,
        stage,
        trace_id,
        constitutional_score,
        meta,
        asdict(data),
    )


def make_run_failed_event(
    *,
    run_id: str,
    sequence: int,
    timestamp: str | datetime,
    correlation_id: str,
    parent_correlation_id: str | None,
    stage: str | None,
    trace_id: str | None,
    constitutional_score: float | None,
    meta: Mapping[str, Any] | None,
    fatal_error_type: str,
    message: str,
    last_stage: str | None,
    total_duration_ms: int,
) -> CanonicalEvent:
    data = RunFailedData(
        fatal_error_type=fatal_error_type,
        message=message,
        last_stage=last_stage,
        total_duration_ms=total_duration_ms,
    )
    return _build_event(
        EventKind.RUN_FAILED,
        run_id,
        sequence,
        timestamp,
        correlation_id,
        parent_correlation_id,
        stage,
        trace_id,
        constitutional_score,
        meta,
        asdict(data),
    )


def _coerce_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(mapping, dict):
        return {k: _coerce_value(v) for k, v in mapping.items()}
    return {k: _coerce_value(v) for k, v in dict(mapping).items()}


def _coerce_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return]
    if _is_dataclass_instance(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return _coerce_mapping(value)
    return value


def _is_dataclass_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__")


__all__ = [
    "EventKind",
    "CanonicalEvent",
    "make_run_started_event",
    "make_stage_started_event",
    "make_stage_completed_event",
    "make_ability_invocation_started_event",
    "make_ability_invocation_chunk_event",
    "make_ability_invocation_completed_event",
    "make_run_log_event",
    "make_run_error_event",
    "make_run_terminated_event",
    "make_run_failed_event",
    "CANONICAL_VERSION",
]
