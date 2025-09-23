"""Unified orchestration pipeline for Super Alita.

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_correlation(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _run_config_snapshot(config: UnifiedRunConfig) -> dict[str, Any]:
    return {
        "stages": _enabled_stage_names(config),
        "abilities": _default_abilities(config),
        "ledger_enabled": False,
    }


def _enabled_stage_names(config: UnifiedRunConfig) -> list[str]:
    stage_flags = [
        ("specification", config.enable_specification),
        ("planning", config.enable_planning),
        ("tasks", config.enable_tasks),
        ("planning_validation", config.sdd_mode and config.enable_planning),
        ("tasks_validation", config.sdd_mode and config.enable_tasks),
        ("consensus", config.enable_consensus),
        ("code_generation", config.enable_code_generation),
        ("validation", config.enable_validation),
        ("scoring", config.enable_scoring),
    ]
    return [name for name, enabled in stage_flags if enabled]


def _default_abilities(config: UnifiedRunConfig) -> list[str]:
    abilities: list[str] = []
    if config.enable_planning:
        abilities.append("task_planner")
    if config.enable_consensus:
        abilities.append("deepconf_consensus")
    if config.enable_code_generation:
        abilities.append("code_generator")
    if config.enable_validation:
        abilities.append("validation_suite")
    if config.enable_scoring:
        abilities.append("score_alignment")
    return abilities


def _stage_summary(output: Any) -> str | None:
    if output is None:
        return None
    try:
        serialized = json.dumps(_summarize(output))
    except TypeError:
        serialized = str(output)
    return truncate_preview(serialized, 200)


def _final_output_preview(stages_output: dict[str, Any]) -> str | None:
    consensus = stages_output.get("consensus")
    if consensus and consensus.get("status") == "success":
        text = (consensus.get("output") or {}).get("consensus_text")
        if text:
            return truncate_preview(str(text), 200)
    code_generation = stages_output.get("code_generation")
    if code_generation and code_generation.get("status") == "success":
        generated = (code_generation.get("output") or {}).get("generated")
        if generated:
            return truncate_preview(str(generated), 200)
    return None


def _count_stage_successes(stages: dict[str, Any]) -> int:
    return sum(1 for data in stages.values() if data.get("status") == "success")


This module provides a thin stage-based orchestrator that reuses existing
tools/abilities. It is intentionally conservative (no deep refactors of
current runtime) and focuses on: determinism, graceful degradation, and
stable event emission.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.core.settings import (
    CANONICAL_EVENTS_ENABLED,
    RUN_LEDGER_ENABLED,
    RUN_LEDGER_PATH,
)

from .canonical_events import (
    CanonicalEvent,
    make_run_error_event,
    make_run_started_event,
    make_run_terminated_event,
    make_stage_completed_event,
    make_stage_started_event,
)
from .constitutional_gate_stub import compute_gate_score
from .event_sanitizer import truncate_preview
from .observability import get_observer
from .reliability_manager import ReliabilityConfig, ReliabilityManager
from .run_ledger import RunLedgerWriter


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
        run_start = time.time()
        observer = get_observer()
        stages_output: dict[str, Any] = {}
        sequence = 0
        ledger_writer = None
        if RUN_LEDGER_ENABLED:
            ledger_writer = RunLedgerWriter(RUN_LEDGER_PATH, enable_shadow=not CANONICAL_EVENTS_ENABLED)
        stage_counter = 0
        abilities_invoked = 0
        run_corr = _new_correlation("run")

        def next_sequence() -> int:
            nonlocal sequence
            current = sequence
            sequence += 1
            return current

        async def emit(event: CanonicalEvent) -> dict[str, Any]:
            payload = event.to_dict()
            await self._emit(payload)
            if ledger_writer is not None:
                try:
                    ledger_writer.append(event)
                except Exception:
                    pass
            if observer is not None:
                try:
                    observer.record_canonical_event(payload)
                except Exception:
                    pass
            return payload

        run_started_payload = await emit(
            make_run_started_event(
                run_id=config.run_id,
                sequence=next_sequence(),
                timestamp=_iso_now(),
                correlation_id=run_corr,
                parent_correlation_id=None,
                stage=None,
                trace_id=None,
                constitutional_score=None,
                meta={
                    "session_id": config.session_id,
                    "sdd_mode": config.sdd_mode,
                },
                input_summary=truncate_preview(config.prompt, 200),
                config=_run_config_snapshot(config),
            )
        )
        yield run_started_payload

        async def stage(name: str, enabled: bool, coro_fn):
            nonlocal stage_counter
            if not enabled:
                stages_output[name] = {"status": "skipped"}
                return
            index = stage_counter
            stage_counter += 1
            stage_corr = _new_correlation(f"stage-{name}")
            started_payload = await emit(
                make_stage_started_event(
                    run_id=config.run_id,
                    sequence=next_sequence(),
                    timestamp=_iso_now(),
                    correlation_id=stage_corr,
                    parent_correlation_id=None,
                    stage=name,
                    trace_id=None,
                    constitutional_score=None,
                    meta=None,
                    name=name,
                    index=index,
                )
            )
            yield started_payload

            t0 = time.time()
            reliability_result = await self.reliability.execute_with_retries(
                name,
                coro_fn,
                config.timeout_s,
                None,
                correlation_id=stage_corr,
            )
            duration_ms = int((time.time() - t0) * 1000)
            reliability_info = {
                key: reliability_result.get(key)
                for key in (
                    "attempts",
                    "retries",
                    "classified_error",
                    "circuit_state",
                )
            }
            reliability_info["correlation_id"] = reliability_result.get("correlation_id") or stage_corr

            if reliability_result["status"] == "skipped":
                stages_output[name] = {
                    "status": "skipped",
                    "duration_ms": duration_ms,
                    "reliability": reliability_info,
                    "reason": reliability_result.get("reason"),
                }
                completed_payload = await emit(
                    make_stage_completed_event(
                        run_id=config.run_id,
                        sequence=next_sequence(),
                        timestamp=_iso_now(),
                        correlation_id=stage_corr,
                        parent_correlation_id=None,
                        stage=name,
                        trace_id=None,
                        constitutional_score=None,
                        meta={"reliability": reliability_info},
                        name=name,
                        index=index,
                        duration_ms=duration_ms,
                        output_summary=None,
                        status="skipped",
                    )
                )
                yield completed_payload
                return

            if reliability_result["status"] == "success":
                result = reliability_result.get("output")
                stages_output[name] = {
                    "status": "success",
                    "output": result,
                    "duration_ms": duration_ms,
                    "reliability": reliability_info,
                }
                completed_payload = await emit(
                    make_stage_completed_event(
                        run_id=config.run_id,
                        sequence=next_sequence(),
                        timestamp=_iso_now(),
                        correlation_id=stage_corr,
                        parent_correlation_id=None,
                        stage=name,
                        trace_id=None,
                        constitutional_score=None,
                        meta={"reliability": reliability_info},
                        name=name,
                        index=index,
                        duration_ms=duration_ms,
                        output_summary=_stage_summary(result),
                        status="ok",
                    )
                )
                yield completed_payload
                return

            error_message = reliability_result.get("error", "unknown")
            stages_output[name] = {
                "status": "failed",
                "error": error_message,
                "duration_ms": duration_ms,
                "reliability": reliability_info,
            }
            error_payload = await emit(
                make_run_error_event(
                    run_id=config.run_id,
                    sequence=next_sequence(),
                    timestamp=_iso_now(),
                    correlation_id=_new_correlation(f"error-{name}"),
                    parent_correlation_id=stage_corr,
                    stage=name,
                    trace_id=None,
                    constitutional_score=None,
                    meta={"reliability": reliability_info},
                    scope="stage",
                    stage_name=name,
                    ability=None,
                    error_type=reliability_result.get("classified_error")
                    or "STAGE_ERROR",
                    message=truncate_preview(str(error_message), 200)
                    or "stage failed",
                    retryable=False,
                )
            )
            yield error_payload
            completed_payload = await emit(
                make_stage_completed_event(
                    run_id=config.run_id,
                    sequence=next_sequence(),
                    timestamp=_iso_now(),
                    correlation_id=stage_corr,
                    parent_correlation_id=None,
                    stage=name,
                    trace_id=None,
                    constitutional_score=None,
                    meta=None,
                    name=name,
                    index=index,
                    duration_ms=duration_ms,
                    output_summary=None,
                    status="partial",
                )
            )
            yield completed_payload

        async def do_spec():
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
            return {"spec": config.prompt}

        async def do_plan():
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
            if hasattr(self.registry, "execute") and self.registry.knows(
                "task_planner"
            ):
                return await self.registry.execute(
                    "task_planner", {"prompt": config.prompt}
                )
            return {
                "steps": [
                    {
                        "id": 1,
                        "action": "Echo goal",
                        "rationale": "fallback",
                    }
                ],
                "source": "fallback",
            }

        async def do_tasks(plan_out: dict[str, Any]):
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
            steps = plan_out.get("steps") or []
            tasks = [f"Task {s.get('id')}: {s.get('action')}" for s in steps]
            return {"tasks": tasks}

        async def do_sdd_validate(phase: str, content: dict[str, Any]):
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
                    consensus = await self.registry.execute(
                        "deepconf_consensus",
                        {
                            "prompt": base_text,
                            "method": "weighted_vote",
                            "num_samples": 3,
                        },
                    )
                    text = (
                        consensus.get("consensus_text")
                        or consensus.get("best_response")
                        or consensus.get("consensus_result")
                        or base_text
                    )
                    return {"consensus_text": text, "raw": consensus}
                except Exception:
                    pass
            return {"consensus_text": base_text, "raw": {"fallback": True}}

        async def do_codegen(consensus_out: dict[str, Any]):
            code_prompt = consensus_out.get("consensus_text", config.prompt)
            if hasattr(self.registry, "execute") and self.registry.knows(
                "code_generator"
            ):
                try:
                    return await self.registry.execute(
                        "code_generator", {"prompt": code_prompt}
                    )
                except Exception:
                    pass
            return {"generated": code_prompt}

        async def do_validation():
            if hasattr(self.registry, "execute") and self.registry.knows(
                "validation_suite"
            ):
                try:
                    return await self.registry.execute(
                        "validation_suite", {"prompt": config.prompt}
                    )
                except Exception:
                    pass
            return {"status": "skipped"}

        async def do_scoring(code_out: dict[str, Any]):
            if hasattr(self.registry, "execute") and self.registry.knows(
                "score_alignment"
            ):
                try:
                    return await self.registry.execute(
                        "score_alignment", {"code": str(code_out)}
                    )
                except Exception:
                    pass
            return {"score": 0.0, "reason": "scoring not available"}

        plan_out: dict[str, Any] | None = None
        tasks_out: dict[str, Any] | None = None
        consensus_out: dict[str, Any] | None = None
        code_out: dict[str, Any] | None = None

        async for event in stage(
            "specification", config.enable_specification, do_spec
        ):
            yield event

        async for event in stage("planning", config.enable_planning, do_plan):
            yield event
        if stages_output.get("planning", {}).get("status") == "success":
            plan_out = stages_output["planning"].get("output")

        async for event in stage(
            "tasks",
            config.enable_tasks,
            lambda: do_tasks(plan_out or {"steps": []}),
        ):
            yield event
        if stages_output.get("tasks", {}).get("status") == "success":
            tasks_out = stages_output["tasks"].get("output")

        if config.sdd_mode and config.enable_planning:
            async for event in stage(
                "planning_validation",
                True,
                lambda: do_sdd_validate("plan", plan_out or {}),
            ):
                yield event

        if config.sdd_mode and config.enable_tasks:
            async for event in stage(
                "tasks_validation",
                True,
                lambda: do_sdd_validate("tasks", tasks_out or {}),
            ):
                yield event

        async for event in stage("consensus", config.enable_consensus, do_consensus):
            yield event
        if stages_output.get("consensus", {}).get("status") == "success":
            consensus_out = stages_output["consensus"].get("output")

        async for event in stage(
            "code_generation",
            config.enable_code_generation,
            lambda: do_codegen(consensus_out or {}),
        ):
            yield event
        if stages_output.get("code_generation", {}).get("status") == "success":
            code_out = stages_output["code_generation"].get("output")

        async for event in stage(
            "validation", config.enable_validation, do_validation
        ):
            yield event

        async for event in stage(
            "scoring",
            config.enable_scoring,
            lambda: do_scoring(code_out or {}),
        ):
            yield event

        gate_score = compute_gate_score(stages_output)
        total_duration_ms = int((time.time() - run_start) * 1000)
        terminated_payload = await emit(
            make_run_terminated_event(
                run_id=config.run_id,
                sequence=next_sequence(),
                timestamp=_iso_now(),
                correlation_id=run_corr,
                parent_correlation_id=None,
                stage=None,
                trace_id=None,
                constitutional_score=gate_score,
                meta={"stages": list(stages_output.keys())},
                success=True,
                total_duration_ms=total_duration_ms,
                stages_executed=_count_stage_successes(stages_output),
                abilities_invoked=abilities_invoked,
                final_output_preview=_final_output_preview(stages_output),
            )
        )
        aggregate = _aggregate(stages_output)
        self._last_result = {
            "run_id": config.run_id,
            "prompt": config.prompt,
            "stages": stages_output,
            "aggregate": aggregate,
        }
        yield terminated_payload
    async def _emit(self, event: dict[str, Any]) -> None:
        try:
            if self.event_bus is not None:
                await self.event_bus.emit(event)
        except Exception:  # pragma: no cover - emission failures non-fatal
            pass




def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _new_correlation(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _run_config_snapshot(config: UnifiedRunConfig) -> dict[str, Any]:
    return {
        "stages": _enabled_stage_names(config),
        "abilities": _default_abilities(config),
        "ledger_enabled": False,
    }


def _enabled_stage_names(config: UnifiedRunConfig) -> list[str]:
    stage_flags = [
        ("specification", config.enable_specification),
        ("planning", config.enable_planning),
        ("tasks", config.enable_tasks),
        ("planning_validation", config.sdd_mode and config.enable_planning),
        ("tasks_validation", config.sdd_mode and config.enable_tasks),
        ("consensus", config.enable_consensus),
        ("code_generation", config.enable_code_generation),
        ("validation", config.enable_validation),
        ("scoring", config.enable_scoring),
    ]
    return [name for name, enabled in stage_flags if enabled]


def _default_abilities(config: UnifiedRunConfig) -> list[str]:
    abilities: list[str] = []
    if config.enable_planning:
        abilities.append("task_planner")
    if config.enable_consensus:
        abilities.append("deepconf_consensus")
    if config.enable_code_generation:
        abilities.append("code_generator")
    if config.enable_validation:
        abilities.append("validation_suite")
    if config.enable_scoring:
        abilities.append("score_alignment")
    return abilities


def _stage_summary(output: Any) -> str | None:
    if output is None:
        return None
    try:
        serialized = json.dumps(_summarize(output))
    except TypeError:
        serialized = str(output)
    return truncate_preview(serialized, 200)


def _final_output_preview(stages_output: dict[str, Any]) -> str | None:
    consensus = stages_output.get("consensus")
    if consensus and consensus.get("status") == "success":
        text = (consensus.get("output") or {}).get("consensus_text")
        if text:
            return truncate_preview(str(text), 200)
    code_generation = stages_output.get("code_generation")
    if code_generation and code_generation.get("status") == "success":
        generated = (code_generation.get("output") or {}).get("generated")
        if generated:
            return truncate_preview(str(generated), 200)
    return None


def _count_stage_successes(stages: dict[str, Any]) -> int:
    return sum(1 for data in stages.values() if data.get("status") == "success")

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
