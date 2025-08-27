
import pytest
from cortex.planner.parallel_wrapper import ParallelWrapper


class DummyPlanner:
    async def ainvoke(self, *_args, **_kwargs):
        return {"step1": {"value": 1}, "step2": {"value": 2}}

    async def decide_and_run(self, *_args, **_kwargs):
        return {"a": {"result": "ok"}, "b": {"result": "done"}}


@pytest.mark.asyncio
async def test_parallel_wrapper_standardizes_results():
    wrapper = ParallelWrapper(DummyPlanner())

    parallel = await wrapper.ainvoke()
    assert parallel["parallel"] is True
    assert parallel["steps"] == [{"value": 1}, {"value": 2}]

    parallel2 = await wrapper.decide_and_run()
    assert parallel2["parallel"] is True
    assert parallel2["steps"] == [{"result": "ok"}, {"result": "done"}]


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


