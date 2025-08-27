
import asyncio
from cortex.planner.parallel_wrapper import ParallelLadderWrapper


class _SuccessRunnable:
    async def ainvoke(self, input: dict[str, str]) -> str:  # pragma: no cover - trivial
        await asyncio.sleep(0.01)
        return "ok"


class _FailRunnable:
    async def ainvoke(self, input: dict[str, str]) -> str:  # pragma: no cover - trivial
        raise RuntimeError("boom")


def test_parallel_wrapper_returns_error_payload() -> None:
    wrapper = ParallelLadderWrapper(
        [
            ("good", _SuccessRunnable()),
            ("bad", _FailRunnable()),
        ]
    )

    result = asyncio.run(wrapper.ainvoke({}))

    assert result["good"] == {"result": "ok", "success": True}
    assert result["bad"]["success"] is False
    assert result["bad"]["error"] == "boom"

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

