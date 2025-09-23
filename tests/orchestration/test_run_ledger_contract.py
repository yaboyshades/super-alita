from __future__ import annotations

import json
import os

import pytest

from src.orchestration.canonical_events import (
    make_run_started_event,
    make_run_terminated_event,
)
from src.orchestration.run_ledger import RunLedgerWriter


@pytest.mark.parametrize("shadow_mode", [True, False])
def test_run_ledger_appends_in_order(tmp_path, shadow_mode):
    ledger_path = tmp_path / "run_ledger.ndjson"
    writer = RunLedgerWriter(ledger_path, enable_shadow=shadow_mode)

    start_event = make_run_started_event(
        run_id="run-123",
        sequence=0,
        timestamp="2025-09-18T12:00:00Z",
        correlation_id="run",
        parent_correlation_id=None,
        stage=None,
        trace_id=None,
        constitutional_score=None,
        meta=None,
        input_summary="short",
        config={"stages": ["plan"], "abilities": ["planner"], "ledger_enabled": True},
    )
    final_event = make_run_terminated_event(
        run_id="run-123",
        sequence=1,
        timestamp="2025-09-18T12:00:01Z",
        correlation_id="run",
        parent_correlation_id=None,
        stage=None,
        trace_id=None,
        constitutional_score=0.82,
        meta=None,
        success=True,
        total_duration_ms=1000,
        stages_executed=4,
        abilities_invoked=2,
        final_output_preview="x" * 500,
    )

    writer.append(start_event)
    writer.append(final_event)

    assert ledger_path.exists()
    lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]
    assert [rec["sequence"] for rec in records] == [0, 1]
    assert all(rec["kind"] in {"RunStarted", "RunTerminated"} for rec in records)

    terminated = records[1]
    preview = terminated["data"].get("final_output_preview", "")
    assert len(preview) <= 200

    if os.name != "nt":
        mode = os.stat(ledger_path).st_mode & 0o777
        assert mode == 0o600


def test_run_ledger_writes_atomic_lines(tmp_path):
    ledger_path = tmp_path / "ledger.ndjson"
    writer = RunLedgerWriter(ledger_path)
    event = make_run_started_event(
        run_id="run-456",
        sequence=0,
        timestamp="2025-09-18T12:00:00Z",
        correlation_id="run",
        parent_correlation_id=None,
        stage=None,
        trace_id=None,
        constitutional_score=None,
        meta=None,
        input_summary="short",
        config={"stages": [], "abilities": [], "ledger_enabled": False},
    )

    writer.append(event)
    writer.append(event.with_sequence(5))

    data = ledger_path.read_text(encoding="utf-8")
    assert data.endswith("\n")
    assert sum(1 for _ in data.splitlines() if _) == 2

    # Ensure file permissions remain restrictive across subsequent appends.
    if os.name != "nt":
        mode = os.stat(ledger_path).st_mode & 0o777
        assert mode == 0o600
