from __future__ import annotations
import asyncio, os
from reug.events import EventEmitter
from reug.services import create_services
from reug.fsm import ExecutionFlow, FSMContext  # type: ignore

def test_services_end_to_end_fallback(tmp_path, monkeypatch):
    # Force fallback mode (no repo-local SoT/executor needed)
    monkeypatch.setenv("REUG_EVENT_LOG_DIR", str(tmp_path))
    emitter = EventEmitter(str(tmp_path / "events.jsonl"))

    services = create_services(emitter)

    async def run():
        flow = ExecutionFlow(services, lambda event: None)
        ctx = FSMContext(raw_input={"sot": ["compute: 1+1", "analyze: 2*3", "generate: ok"]}, results=[])
        # Kick manually: decompose then iterate SELECT→EXECUTE→PROCESS until done
        ctx.plan = await services["decompose"](ctx.raw_input)
        ctx.step_index = 0
        while ctx.step_index < len(ctx.plan.steps):
            step = ctx.plan.steps[ctx.step_index]
            sel = await services["select_tool"](step, ctx.__dict__)
            assert sel["status"] == "FOUND"
            ex = await services["execute"](sel["tool"], step.args, {**ctx.__dict__, "step_index": ctx.step_index})
            assert ex["status"] == "SUCCESS"
            ctx.results.append(ex["result"])
            ctx.step_index += 1
            pr = await services["process_result"](ctx.__dict__)
            if pr["task_complete"]:
                break
        return ctx

    ctx = asyncio.run(run())
    assert len(ctx.results) == 3
    # Sanity check values from fallback executor
    assert ctx.results[0]["value"] == 2
    assert ctx.results[1]["value"] == 6
