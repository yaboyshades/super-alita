from __future__ import annotations

from src.reug_runtime.canonical_event_adapter import canonicalize_legacy_stream


def test_legacy_events_mirror_canonical_sequence():
    legacy_events = [
        {
            "type": "UnifiedRunStarted",
            "run_id": "run-1",
            "prompt": "hello",
        },
        {
            "type": "UnifiedStageStarted",
            "run_id": "run-1",
            "stage": "planning",
        },
        {
            "type": "UnifiedStageSucceeded",
            "run_id": "run-1",
            "stage": "planning",
            "duration_ms": 12,
            "output_summary": {"keys": ["steps"]},
        },
        {
            "type": "UnifiedRunCompleted",
            "run_id": "run-1",
            "aggregate": {"consensus_text": "hello"},
        },
    ]

    canonical_events = [
        event.to_dict()
        for event in canonicalize_legacy_stream("run-1", legacy_events)
    ]

    assert [event["kind"] for event in canonical_events] == [
        "RunStarted",
        "StageStarted",
        "StageCompleted",
        "RunTerminated",
    ]
    sequences = [event["sequence"] for event in canonical_events]
    assert sequences == sorted(sequences)
    assert canonical_events[-1]["data"]["success"] is True
