import pytest

from cortex.planner import parallel_wrapper


async def _noop():
    return None


async def _fail():
    raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_emits_telemetry_parallel_success(monkeypatch):
    events = []

    def record(mode: str, steps: int, duration: float, success: bool) -> None:
        events.append(
            {
                "mode": mode,
                "steps": steps,
                "duration": duration,
                "success": success,
            }
        )

    monkeypatch.setattr(parallel_wrapper, "_emit_telemetry", record)

    await parallel_wrapper.decide_and_run([_noop, _noop], parallel=True)

    assert events
    event = events[0]
    assert event["mode"] == "parallel"
    assert event["steps"] == 2
    assert event["success"] is True
    assert event["duration"] >= 0


@pytest.mark.asyncio
async def test_emits_telemetry_sequential_failure(monkeypatch):
    events = []

    def record(mode: str, steps: int, duration: float, success: bool) -> None:
        events.append(
            {
                "mode": mode,
                "steps": steps,
                "duration": duration,
                "success": success,
            }
        )

    monkeypatch.setattr(parallel_wrapper, "_emit_telemetry", record)

    with pytest.raises(RuntimeError):
        await parallel_wrapper.decide_and_run([_noop, _fail], parallel=False)

    assert events
    event = events[0]
    assert event["mode"] == "sequential"
    assert event["steps"] == 2
    assert event["success"] is False
    assert event["duration"] >= 0
