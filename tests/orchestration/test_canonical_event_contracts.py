from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from src.orchestration.canonical_events import (
    EventKind,
    make_ability_invocation_chunk_event,
    make_ability_invocation_completed_event,
    make_ability_invocation_started_event,
    make_run_error_event,
    make_run_failed_event,
    make_run_log_event,
    make_run_started_event,
    make_run_terminated_event,
    make_stage_completed_event,
    make_stage_started_event,
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _base_kwargs(
    run_id: str,
    sequence: int,
    *,
    correlation_id: str,
    parent: str | None = None,
    stage: str | None = None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "sequence": sequence,
        "timestamp": _utc_now_iso(),
        "correlation_id": correlation_id,
        "parent_correlation_id": parent,
        "stage": stage,
        "trace_id": "trace-123",
        "constitutional_score": None,
        "meta": {"source": "contract-test"},
    }


@pytest.mark.parametrize(
    "builder,kind,required_keys",
    [
        (
            lambda run_id: make_run_started_event(
                **_base_kwargs(run_id, 0, correlation_id="run"),
                input_summary="Demo run",
                config={
                    "stages": ["ingest", "plan"],
                    "abilities": ["planner"],
                    "ledger_enabled": False,
                },
            ),
            EventKind.RUN_STARTED,
            {"input_summary", "config"},
        ),
        (
            lambda run_id: make_stage_started_event(
                **_base_kwargs(run_id, 1, correlation_id="stage-plan", stage="plan"),
                name="plan",
                index=0,
            ),
            EventKind.STAGE_STARTED,
            {"name", "index"},
        ),
        (
            lambda run_id: make_stage_completed_event(
                **_base_kwargs(
                    run_id,
                    2,
                    correlation_id="stage-plan",
                    parent="stage-plan",
                    stage="plan",
                ),
                name="plan",
                index=0,
                duration_ms=123,
                output_summary="Plan ready",
                status="ok",
            ),
            EventKind.STAGE_COMPLETED,
            {"name", "index", "duration_ms", "output_summary", "status"},
        ),
        (
            lambda run_id: make_ability_invocation_started_event(
                **_base_kwargs(
                    run_id,
                    3,
                    correlation_id="ability-plan",
                    parent="stage-plan",
                    stage="plan",
                ),
                ability="planner",
                args_hash="abc123456789def0",
            ),
            EventKind.ABILITY_INVOCATION_STARTED,
            {"ability", "args_hash"},
        ),
        (
            lambda run_id: make_ability_invocation_chunk_event(
                **_base_kwargs(
                    run_id,
                    4,
                    correlation_id="ability-plan",
                    parent="stage-plan",
                    stage="plan",
                ),
                ability="planner",
                chunk="chunk-0",
                index=0,
                is_final=False,
            ),
            EventKind.ABILITY_INVOCATION_CHUNK,
            {"ability", "chunk", "index", "is_final"},
        ),
        (
            lambda run_id: make_ability_invocation_completed_event(
                **_base_kwargs(
                    run_id,
                    5,
                    correlation_id="ability-plan",
                    parent="stage-plan",
                    stage="plan",
                ),
                ability="planner",
                duration_ms=321,
                result_preview="Steps ready",
                status="ok",
                error_type=None,
            ),
            EventKind.ABILITY_INVOCATION_COMPLETED,
            {"ability", "duration_ms", "result_preview", "status", "error_type"},
        ),
        (
            lambda run_id: make_run_log_event(
                **_base_kwargs(run_id, 6, correlation_id="log"),
                level="INFO",
                message="Stage completed",
                context={"stage": "plan"},
            ),
            EventKind.RUN_LOG,
            {"level", "message", "context"},
        ),
        (
            lambda run_id: make_run_error_event(
                **_base_kwargs(run_id, 7, correlation_id="err", stage="plan"),
                scope="stage",
                stage_name="plan",
                ability=None,
                error_type="PLAN_TIMEOUT",
                message="Plan timed out",
                retryable=True,
            ),
            EventKind.RUN_ERROR,
            {"scope", "stage", "ability", "error_type", "message", "retryable"},
        ),
        (
            lambda run_id: make_run_terminated_event(
                **_base_kwargs(run_id, 8, correlation_id="run"),
                success=True,
                total_duration_ms=999,
                stages_executed=3,
                abilities_invoked=1,
                final_output_preview="Ok",
            ),
            EventKind.RUN_TERMINATED,
            {
                "success",
                "total_duration_ms",
                "stages_executed",
                "abilities_invoked",
                "final_output_preview",
            },
        ),
        (
            lambda run_id: make_run_failed_event(
                **_base_kwargs(run_id, 9, correlation_id="run"),
                fatal_error_type="SYSTEM_DOWN",
                message="Fatal error",
                last_stage="plan",
                total_duration_ms=1000,
            ),
            EventKind.RUN_FAILED,
            {"fatal_error_type", "message", "last_stage", "total_duration_ms"},
        ),
    ],
)
def test_canonical_event_payload_contracts(builder, kind, required_keys):
    run_id = str(uuid.uuid4())
    event = builder(run_id)
    payload = event.to_dict()
    assert payload["version"] == "v1"
    assert payload["kind"] == kind.value
    assert payload["run_id"] == run_id
    assert isinstance(payload["sequence"], int)
    assert payload["timestamp"].endswith("Z")
    assert payload["correlation_id"]
    assert "parent_correlation_id" in payload
    assert "meta" in payload and isinstance(payload["meta"], dict)
    assert set(required_keys).issubset(payload["data"].keys())


def test_event_kind_enum_matches_spec():
    expected = {
        "RunStarted",
        "StageStarted",
        "StageCompleted",
        "AbilityInvocationStarted",
        "AbilityInvocationChunk",
        "AbilityInvocationCompleted",
        "RunLog",
        "RunError",
        "RunTerminated",
        "RunFailed",
    }
    assert {member.value for member in EventKind} == expected
