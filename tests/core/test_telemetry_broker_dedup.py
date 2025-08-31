from src.core.telemetry_broker import TelemetryBroker


def test_envelope_merges_identical_events_losslessly() -> None:
    b = TelemetryBroker(max_events_per_category=10, max_tokens=9999)
    for _ in range(5):
        b.ingest("planner", "Same message", importance=1.0, meta={"k": "v"})
    # Slightly different meta shouldn't merge
    b.ingest("planner", "Same message", importance=1.0, meta={"k": "v2"})

    env = b.build_envelope()
    cat = env["categories"]["planner"]["events"]
    # Expect two entries: one merged with count=5, one with count=1
    counts = sorted([e.get("count", 1) for e in cat])
    assert counts == [1, 5]
