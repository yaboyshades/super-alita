from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Any

from src.orchestration.canonical_events import (
    make_run_error_event,
    make_run_failed_event,
    make_run_started_event,
    make_run_terminated_event,
    make_stage_completed_event,
    make_stage_started_event,
)
from src.orchestration.event_sanitizer import truncate_preview


def canonicalize_legacy_stream(
    run_id: str, legacy_events: Iterable[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    """Translate legacy orchestrator events into canonical envelopes."""

    sequence = 0
    stage_index = 0
    stage_correlations: dict[str, str] = {}
    stages_executed = 0

    def next_sequence() -> int:
        nonlocal sequence
        value = sequence
        sequence += 1
        return value

    for legacy in legacy_events:
        kind = str(legacy.get("type", ""))
        if kind == "UnifiedRunStarted":
            input_summary = truncate_preview(
                str(legacy.get("prompt", "")), 200
            )
            config = _legacy_config_snapshot(legacy)
            canonical = make_run_started_event(
                run_id=run_id,
                sequence=next_sequence(),
                timestamp=_iso_now(),
                correlation_id=_correlation("run"),
                parent_correlation_id=None,
                stage=None,
                trace_id=None,
                constitutional_score=None,
                meta=None,
                input_summary=input_summary or "",
                config=config,
            )
            yield canonical
            continue

        if kind == "UnifiedStageStarted":
            stage = str(legacy.get("stage", "unknown"))
            stage_corr = stage_correlations.setdefault(
                stage, _correlation(stage)
            )
            canonical = make_stage_started_event(
                run_id=run_id,
                sequence=next_sequence(),
                timestamp=_iso_now(),
                correlation_id=stage_corr,
                parent_correlation_id=None,
                stage=stage,
                trace_id=None,
                constitutional_score=None,
                meta=None,
                name=stage,
                index=stage_index,
            )
            stage_index += 1
            yield canonical
            continue

        if kind in {"UnifiedStageSucceeded", "UnifiedStageFailed"}:
            stage = str(legacy.get("stage", "unknown"))
            stage_corr = stage_correlations.setdefault(
                stage, _correlation(stage)
            )
            status = "ok" if kind == "UnifiedStageSucceeded" else "partial"
            if status == "ok":
                stages_executed += 1
            duration_ms = int(legacy.get("duration_ms", 0) or 0)
            summary_source = legacy.get("output_summary")
            summary_text = _stringify(summary_source)
            canonical = make_stage_completed_event(
                run_id=run_id,
                sequence=next_sequence(),
                timestamp=_iso_now(),
                correlation_id=stage_corr,
                parent_correlation_id=None,
                stage=stage,
                trace_id=None,
                constitutional_score=None,
                meta=None,
                name=stage,
                index=max(stage_index - 1, 0),
                duration_ms=duration_ms,
                output_summary=summary_text,
                status=status,
            )
            yield canonical
            if status != "ok":
                error = make_run_error_event(
                    run_id=run_id,
                    sequence=next_sequence(),
                    timestamp=_iso_now(),
                    correlation_id=_correlation(f"error-{stage}"),
                    parent_correlation_id=stage_corr,
                    stage=stage,
                    trace_id=None,
                    constitutional_score=None,
                    meta=None,
                    scope="stage",
                    stage_name=stage,
                    ability=None,
                    error_type="LEGACY_STAGE_FAILURE",
                    message=truncate_preview(
                        _stringify(legacy.get("error")), 200
                    )
                    or "stage failed",
                    retryable=False,
                )
                yield error
            continue

        if kind == "UnifiedRunCompleted":
            aggregate = legacy.get("aggregate", {}) or {}
            final_preview = truncate_preview(
                _stringify(
                    aggregate.get("consensus_text")
                    or aggregate.get("final_output")
                    or ""
                ),
                200,
            )
            canonical = make_run_terminated_event(
                run_id=run_id,
                sequence=next_sequence(),
                timestamp=_iso_now(),
                correlation_id=_correlation("run"),
                parent_correlation_id=None,
                stage=None,
                trace_id=None,
                constitutional_score=0.82,
                meta=None,
                success=True,
                total_duration_ms=int(legacy.get("duration_ms", 0) or 0),
                stages_executed=stages_executed,
                abilities_invoked=legacy.get("abilities_invoked", 0) or 0,
                final_output_preview=final_preview,
            )
            yield canonical
            continue

        if kind == "UnifiedRunFailed":
            canonical = make_run_failed_event(
                run_id=run_id,
                sequence=next_sequence(),
                timestamp=_iso_now(),
                correlation_id=_correlation("run"),
                parent_correlation_id=None,
                stage=None,
                trace_id=None,
                constitutional_score=0.0,
                meta=None,
                fatal_error_type=str(legacy.get("error_type", "unknown")),
                message=_stringify(legacy.get("error", "fatal")) or "fatal",
                last_stage=(
                    str(legacy.get("stage")) if legacy.get("stage") else None
                ),
                total_duration_ms=int(legacy.get("duration_ms", 0) or 0),
            )
            yield canonical
            continue


def _legacy_config_snapshot(legacy: dict[str, Any]) -> dict[str, Any]:
    config = legacy.get("config") or {}
    stages = config.get("stages") or []
    abilities = config.get("abilities") or []
    return {
        "stages": list(stages),
        "abilities": list(abilities),
        "ledger_enabled": bool(config.get("ledger_enabled", False)),
    }


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _correlation(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return str(value)
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


__all__ = ["canonicalize_legacy_stream"]
