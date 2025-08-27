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
