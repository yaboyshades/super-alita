"""Unified orchestration pipeline for Super Alita.

This module provides a thin stage-based orchestrator that reuses existing
tools/abilities. It is intentionally conservative (no deep refactors of
current runtime) and focuses on: determinism, graceful degradation, and
stable event emission.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass
from typing import Any

from .reliability_manager import ReliabilityConfig, ReliabilityManager


@dataclass(slots=True)
class UnifiedRunConfig:
    prompt: str
    run_id: str
    session_id: str = "default"
    enable_specification: bool = False
    enable_planning: bool = True
    enable_tasks: bool = False
    enable_consensus: bool = True
    enable_code_generation: bool = False
    enable_validation: bool = False
    enable_scoring: bool = False
    test_first: bool = True
    file_path: str | None = None
    language: str = "python"
    timeout_s: int = 120
    # SDD-specific configuration
    sdd_mode: bool = False
    sdd_feature_id: str | None = None
    sdd_phase: str | None = None  # specify, plan, tasks, implement
    constitutional_threshold: float = 0.75
    sdd_template_dir: str = "templates/sdd"

    @classmethod
    def from_args(cls, prompt: str, args: dict[str, Any]) -> UnifiedRunConfig:
        return cls(
            prompt=prompt,
            run_id=args.get("run_id") or str(uuid.uuid4()),
            session_id=(args.get("session_id") or args.get("session") or "default"),
            enable_specification=bool(args.get("enable_specification", False)),
            enable_planning=bool(args.get("enable_planning", True)),
            enable_tasks=bool(args.get("enable_tasks", False)),
            enable_consensus=bool(args.get("enable_consensus", True)),
            enable_code_generation=bool(args.get("enable_code_generation", False)),
            enable_validation=bool(args.get("enable_validation", False)),
            enable_scoring=bool(args.get("enable_scoring", False)),
            test_first=bool(args.get("test_first", True)),
            file_path=(args.get("file_path") or None),
            language=str(args.get("language") or "python"),
            timeout_s=int(args.get("timeout_s", 120) or 120),
        )


class UnifiedOrchestrator:
    """Core orchestrator implementing the stage pipeline.

    It does not depend on FastAPI; only on the provided *ability_registry*
    and *event_bus*. All tool invocations are awaited. Each stage is
    wrapped with timing + error capture; failures do not abort unless
    explicitly marked fatal (none are fatal in v0).
    """

    def __init__(self, ability_registry: Any, event_bus: Any):
        self.registry = ability_registry
        self.event_bus = event_bus
        # Reliability manager (default config). Future: env-based overrides.
        self.reliability = ReliabilityManager(ReliabilityConfig())
        # Cached last run result for non-stream callers
        self._last_result: dict[str, Any] | None = None

    async def run(self, config: UnifiedRunConfig) -> dict[str, Any]:
        """Run configured stages (non-stream) and return aggregate result."""
        aggregate: dict[str, Any] = {
            "run_id": config.run_id,
            "prompt": config.prompt,
            "stages": {},
        }
        async for _ in self.run_stream(config):  # drain generator
            pass
        # Reconstruct from cached stage results (events are ephemeral).
        return getattr(self, "_last_result", aggregate)

    async def run_stream(
        self, config: UnifiedRunConfig
    ) -> AsyncGenerator[dict[str, Any], None]:
        stages_output: dict[str, Any] = {}
        run_meta = {"config": asdict(config)}
        started_ev = {
            "type": "UnifiedRunStarted",
            "run_id": config.run_id,
            "session_id": config.session_id,
            "prompt": config.prompt,
            **run_meta,
        }
        await self._emit(started_ev)
        yield started_ev

        async def stage(name: str, enabled: bool, coro_fn):
            if not enabled:
                stages_output[name] = {"status": "skipped"}
                return
            t0 = time.time()
            start_ev = {
                "type": "UnifiedStageStarted",
                "run_id": config.run_id,
                "stage": name,
            }
            await self._emit(start_ev)
            yield start_ev
            try:
                reliability_result = await self.reliability.execute_with_retries(
                    name, coro_fn, config.timeout_s, self._emit
                )
                dt = int((time.time() - t0) * 1000)
                if reliability_result["status"] == "skipped":
                    skip_ev = {
                        "type": "UnifiedStageSkipped",
                        "run_id": config.run_id,
                        "stage": name,
                        "duration_ms": dt,
                        "reason": reliability_result.get("reason", "unknown"),
                        "reliability": {
                            k: reliability_result.get(k)
                            for k in [
                                "attempts",
                                "retries",
                                "classified_error",
                                "circuit_state",
                            ]
                        },
                    }
                    stages_output[name] = {"status": "skipped", **skip_ev}
                    await self._emit(skip_ev)
                    yield skip_ev
                    return
                if reliability_result["status"] == "success":
                    result = reliability_result.get("output")
                    succ_ev = {
                        "type": "UnifiedStageSucceeded",
                        "run_id": config.run_id,
                        "stage": name,
                        "duration_ms": dt,
                        "output_summary": _summarize(result),
                        "reliability": {
                            k: reliability_result.get(k)
                            for k in [
                                "attempts",
                                "retries",
                                "classified_error",
                                "circuit_state",
                            ]
                        },
                    }
                    stages_output[name] = {
                        "status": "success",
                        "output": result,
                        "duration_ms": dt,
                        "reliability": succ_ev["reliability"],
                    }
                    await self._emit(succ_ev)
                    yield succ_ev
                else:  # failed
                    err_ev = {
                        "type": "UnifiedStageFailed",
                        "run_id": config.run_id,
                        "stage": name,
                        "duration_ms": dt,
                        "error": reliability_result.get("error", "unknown"),
                        "reliability": {
                            k: reliability_result.get(k)
                            for k in [
                                "attempts",
                                "retries",
                                "classified_error",
                                "circuit_state",
                            ]
                        },
                    }
                    stages_output[name] = {
                        "status": "failed",
                        "error": err_ev["error"],
                        "duration_ms": dt,
                        "reliability": err_ev["reliability"],
                    }
                    await self._emit(err_ev)
                    yield err_ev
            except Exception as e:  # noqa: BLE001
                dt = int((time.time() - t0) * 1000)
                tb = traceback.format_exc(limit=4)
                fail_ev = {
                    "type": "UnifiedStageFailed",
                    "run_id": config.run_id,
                    "stage": name,
                    "duration_ms": dt,
                    "error": str(e),
                    "reliability": {"unexpected_wrapper_error": True},
                }
                stages_output[name] = {
                    "status": "failed",
                    "error": str(e),
                    "trace": tb.splitlines()[-1],
                    "duration_ms": dt,
                    "reliability": fail_ev["reliability"],
                }
                await self._emit(fail_ev)
                yield fail_ev

        # --- Stage implementations --- #
        async def do_spec():
            # SDD Integration: Use SDD specification if enabled
            if (
                config.sdd_mode
                and hasattr(self.registry, "execute")
                and self.registry.knows("sdd_specify")
            ):
                try:
                    return await self.registry.execute(
                        "sdd_specify", {"content": config.prompt}
                    )
                except Exception:
                    pass
            # Fallback: minimal placeholder
            return {"spec": config.prompt}

        async def do_plan():
            # SDD Integration: Use SDD planning if enabled
            if (
                config.sdd_mode
                and hasattr(self.registry, "execute")
                and self.registry.knows("sdd_plan")
            ):
                spec_content = (
                    stages_output.get("specification", {})
                    .get("output", {})
                    .get("spec", config.prompt)
                )
                try:
                    return await self.registry.execute(
                        "sdd_plan", {"spec_content": spec_content}
                    )
                except Exception:
                    pass
            # Original planning logic
            if hasattr(self.registry, "execute") and self.registry.knows(
                "task_planner"
            ):
                res = await self.registry.execute(
                    "task_planner", {"prompt": config.prompt}
                )
            else:
                res = {
                    "steps": [
                        {
                            "id": 1,
                            "action": "Echo goal",
                            "rationale": "fallback",
                        }
                    ],
                    "source": "fallback",
                }
            return res

        async def do_tasks(plan_out: dict[str, Any]):
            # SDD Integration: Use SDD task breakdown if enabled
            if (
                config.sdd_mode
                and hasattr(self.registry, "execute")
                and self.registry.knows("sdd_tasks")
            ):
                plan_content = plan_out.get("plan", plan_out.get("steps", ""))
                try:
                    return await self.registry.execute(
                        "sdd_tasks", {"plan_content": str(plan_content)}
                    )
                except Exception:
                    pass
            # Original task breakdown logic
            steps = plan_out.get("steps") or []
            tasks = [f"Task {s.get('id')}: {s.get('action')}" for s in steps]
            return {"tasks": tasks}

        async def do_sdd_validate(phase: str, content: dict[str, Any]):
            """Constitutional validation for SDD phases."""
            if hasattr(self.registry, "execute") and self.registry.knows(
                "sdd_validate"
            ):
                try:
                    return await self.registry.execute(
                        "sdd_validate",
                        {
                            "content": str(content),
                            "phase": phase,
                            "constitutional_threshold": (
                                config.constitutional_threshold
                            ),
                        },
                    )
                except Exception:
                    pass
            return {
                "status": "skipped",
                "reason": "sdd_validate not available",
            }

        async def do_consensus():
            base_text = config.prompt
            if hasattr(self.registry, "execute") and self.registry.knows(
                "deepconf_consensus"
            ):
                try:
                    c = await self.registry.execute(
                        "deepconf_consensus",
                        {
                            "prompt": base_text,
                            "method": "weighted_vote",
                            "num_samples": 3,
                        },
                    )
                    text = (
                        c.get("consensus_text")
                        or c.get("best_response")
                        or c.get("consensus_result")
                        or base_text
                    )
                    return {"consensus_text": text, "raw": c}
                except Exception:
                    pass
            return {"consensus_text": base_text, "raw": {"fallback": True}}

        async def do_codegen(consensus_out: dict[str, Any]):
            # Optional CMA enforcement: require blueprint + phases validated
            if os.getenv("CMA_ENFORCEMENT", "0").lower() in {
                "1",
                "true",
                "yes",
            }:
                from src.cma.enforcement import (
                    CMAConfig,
                    CMAEnforcer,
                    load_blueprint_from_env,
                )

                blueprint_text, phases = load_blueprint_from_env()
                if not blueprint_text:
                    # fallback: attempt to use consensus text as blueprint
                    candidate = consensus_out.get("consensus_text") or config.prompt
                    blueprint_text = str(candidate)
                enforcer = CMAEnforcer(CMAConfig())
                report = enforcer.validate_pre_generation(
                    blueprint_text=blueprint_text, phases_completed=phases
                )
                if not report.get("ok"):
                    raise RuntimeError(
                        "CMA enforcement blocked generation: "
                        + "; ".join(report.get("reasons", []))
                    )

            if not config.file_path:
                return {"skipped": True, "reason": "no file_path provided"}
            args = {
                "language": config.language,
                "spec": consensus_out.get("consensus_text") or config.prompt,
                "file_path": config.file_path,
                "test_first": config.test_first,
                "consolidate_tests": True,
                "force_write": True,
            }
            if hasattr(self.registry, "execute") and self.registry.knows(
                "code_synthesize_and_write"
            ):
                return await self.registry.execute("code_synthesize_and_write", args)
            return {"error": "code_synthesize_and_write tool missing"}

        async def do_validation():
            results: dict[str, Any] = {}
            if self.registry.knows("python_import_smoke"):
                try:
                    results["import_smoke"] = await self.registry.execute(
                        "python_import_smoke", {"path": "src"}
                    )
                except Exception as e:  # noqa: BLE001
                    results["import_smoke_error"] = str(e)
            if self.registry.knows("pytest_run"):
                try:
                    results["pytest"] = await self.registry.execute(
                        "pytest_run", {"target": "tests", "quiet": True}
                    )
                except Exception as e:  # noqa: BLE001
                    results["pytest_error"] = str(e)
            return results

        async def do_scoring(code_out: dict[str, Any]):
            if not code_out or code_out.get("error"):
                return {"skipped": True}
            code_text: str | None = None
            if isinstance(code_out.get("synth"), dict):
                # No raw code returned; skip scoring
                pass
            if self.registry.knows("shadow_reward_score") and code_text:
                try:
                    return await self.registry.execute(
                        "shadow_reward_score", {"code": code_text}
                    )
                except Exception:
                    return {"skipped": True, "error": "shadow scoring failed"}
            return {"skipped": True}

        # Execute pipeline (dependencies + optional constitutional checks)
        plan_out: dict[str, Any] | None = None
        consensus_out: dict[str, Any] | None = None
        code_out: dict[str, Any] | None = None

        # Specification stage with constitutional validation
        async for ev in stage("specification", config.enable_specification, do_spec):
            yield ev

        # Constitutional validation for specification (if SDD mode enabled)
        if config.sdd_mode and config.enable_specification:
            spec_payload = stages_output.get("specification", {}).get("output") or {
                "spec": config.prompt
            }
            async for ev in stage(
                "specification_validation",
                True,
                lambda: do_sdd_validate("specification", spec_payload),  # noqa: E731
            ):
                yield ev

        # Planning stage with constitutional validation
        if config.enable_planning:
            async for ev in stage("planning", True, do_plan):
                if ev["type"] == "UnifiedStageSucceeded":
                    plan_out = stages_output["planning"]["output"]
                yield ev

        # Constitutional validation for planning (if SDD mode enabled)
        if config.sdd_mode and config.enable_planning:
            plan_payload = stages_output.get("planning", {}).get("output") or {}
            async for ev in stage(
                "planning_validation",
                True,
                lambda: do_sdd_validate("plan", plan_payload),  # noqa: E731
            ):
                yield ev

        # Tasks stage with constitutional validation
        if config.enable_tasks:
            async for ev in stage(
                "tasks",
                True,
                lambda: do_tasks(plan_out or {"steps": []}),  # noqa: E731
            ):
                yield ev

        # Constitutional validation for tasks (if SDD mode enabled)
        if config.sdd_mode and config.enable_tasks:
            tasks_payload = stages_output.get("tasks", {}).get("output") or {}
            async for ev in stage(
                "tasks_validation",
                True,
                lambda: do_sdd_validate("tasks", tasks_payload),  # noqa: E731
            ):
                yield ev
        if config.enable_consensus:
            async for ev in stage("consensus", True, do_consensus):
                if ev["type"] == "UnifiedStageSucceeded":
                    consensus_out = stages_output["consensus"]["output"]
                yield ev
        if config.enable_code_generation:
            async for ev in stage(
                "code_generation",
                True,
                lambda: do_codegen(
                    consensus_out or {"consensus_text": config.prompt}
                ),  # noqa: E731
            ):
                if ev["type"] == "UnifiedStageSucceeded":
                    code_out = stages_output["code_generation"]["output"]
                yield ev
        if config.enable_validation:
            async for ev in stage("validation", True, do_validation):
                yield ev
        if config.enable_scoring:
            async for ev in stage(
                "scoring",
                True,
                lambda: do_scoring(code_out or {}),  # noqa: E731
            ):
                yield ev

        completed = {
            "type": "UnifiedRunCompleted",
            "run_id": config.run_id,
            "success": True,
            "stages": {
                k: {**v, "output": _truncate(v.get("output"))}
                for k, v in stages_output.items()
            },
            "aggregate": _aggregate(stages_output),
        }
        self._last_result = {  # cache for non-streaming caller
            "run_id": config.run_id,
            "prompt": config.prompt,
            "stages": stages_output,
            "aggregate": completed["aggregate"],
        }
        await self._emit(completed)
        yield completed

    async def _emit(self, event: dict[str, Any]) -> None:
        try:
            if self.event_bus is not None:
                await self.event_bus.emit(event)
        except Exception:  # pragma: no cover - emission failures non-fatal
            pass


def _summarize(output: Any) -> dict[str, Any]:
    if output is None:
        return {"none": True}
    if isinstance(output, dict):
        keys = list(output.keys())[:6]
        return {"keys": keys}
    if isinstance(output, list):
        return {"list_len": len(output)}
    return {"type": type(output).__name__}


def _truncate(obj: Any, limit: int = 5000) -> Any:
    try:
        s = json.dumps(obj)
        if len(s) <= limit:
            return obj
        return {"truncated": True, "approx": s[:limit]}
    except Exception:
        return obj


def _aggregate(stages: dict[str, Any]) -> dict[str, Any]:
    agg: dict[str, Any] = {}
    if "consensus" in stages and stages["consensus"]["status"] == "success":
        agg["consensus_text"] = stages["consensus"]["output"].get("consensus_text")
    if "code_generation" in stages and stages["code_generation"]["status"] == "success":
        fp = stages["code_generation"]["output"].get("file_path")
        if fp:
            agg["written_files"] = [fp]
    return agg


__all__ = [
    "UnifiedRunConfig",
    "UnifiedOrchestrator",
]
