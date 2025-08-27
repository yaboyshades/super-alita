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
