#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List, Optional

from src.core.plugin_interface import PluginInterface
try:
    from src.core.observability import ObservabilityManager  # type: ignore
except Exception:  # pragma: no cover
    ObservabilityManager = None  # type: ignore

from src.core.diff_utils import normalize_diffs

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class DeepCodeClientInterface:
    async def plan(self, request: Dict[str, Any]) -> Dict[str, Any]: ...
    async def collect_references(self, plan: Dict[str, Any]) -> Dict[str, Any]: ...
    async def generate_code(self, plan: Dict[str, Any], references: Dict[str, Any]) -> Dict[str, Any]: ...
    async def validate(self, implementation: Dict[str, Any]) -> Dict[str, Any]: ...


class _StubDeepCodeClient(DeepCodeClientInterface):
    async def plan(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return {"steps": ["draft plan", "produce impl"], "confidence": 0.73, "request": request}
    async def collect_references(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {"snippets": [], "confidence": 0.78}
    async def generate_code(self, plan: Dict[str, Any], references: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "proposal_id": None,
            "diffs": [{
                "path": "docs/deepcode_result.md",
                "change_type": "add",
                "unified_diff": "--- /dev/null\n+++ b/docs/deepcode_result.md\n@@\n+# Result\n+Hello DeepCode\n",
                "new_content": "# Result\n\nHello DeepCode\n",
                "confidence": 0.84
            }],
            "tests": [{"path": "tests/test_deepcode_result.py", "content": "def test_ok(): assert True"}],
            "docs": [{"path": "docs/plan.json", "content": json.dumps(plan)}],
            "confidence": 0.84
        }
    async def validate(self, implementation: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "pass", "lint_errors": 0, "tests_passed": True, "confidence": 0.91}


class DeepCodeOrchestratorPlugin(PluginInterface):
    """
    Orchestrates DeepCode multi-phase generation inside Super-Alita.
    Events (snake_case):
      - deepcode_request_received
      - deepcode_plan_ready
      - deepcode_references_compiled
      - deepcode_implementation_proposed
      - deepcode_validation_report
      - deepcode_ready_for_apply
      - deepcode_pipeline_failed
    """
    def __init__(self, client: DeepCodeClientInterface | None = None):
        super().__init__()
        self._client = client or _StubDeepCodeClient()
        self._active: Dict[str, Dict[str, Any]] = {}
        self._tasks: set[asyncio.Task] = set()
        self._obs: Optional[ObservabilityManager] = None  # type: ignore[name-defined]

    @property
    def name(self) -> str:
        return "deepcode_orchestrator"

    async def setup(self, event_bus: Any, store: Any, config: Dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        self._obs = config.get("observability_manager") if isinstance(config, dict) else None
        logger.info("DeepCode Orchestrator setup complete")

    async def start(self) -> None:
        await super().start()
        await self.subscribe("deepcode_request", self._on_request)
        logger.info("DeepCode Orchestrator started (listening for deepcode_request)")

    async def shutdown(self) -> None:
        logger.info("DeepCode Orchestrator shutting down")
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await super().shutdown()

    async def stop(self) -> None:
        await self.shutdown()

    async def _on_request(self, event: Dict[str, Any]) -> None:
        if not self.is_running:
            return
        request_id = event.get("request_id") or f"dc_req_{int(datetime.now(timezone.utc).timestamp()*1000)}"
        conversation_id = event.get("conversation_id")
        task_kind = event.get("task_kind", "generic")
        requirements = event.get("requirements", "")
        correlation_id = event.get("correlation_id")

        self._active[request_id] = {
            "task_kind": task_kind,
            "requirements": requirements,
            "conversation_id": conversation_id,
            "started_at": _utcnow(),
            "correlation_id": correlation_id,
        }
        await self.emit_event(
            "deepcode_request_received",
            source_plugin=self.name,
            request_id=request_id,
            task_kind=task_kind,
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            timestamp=_utcnow(),
        )
        task = asyncio.create_task(self._run_pipeline(request_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_pipeline(self, request_id: str) -> None:
        ctx = self._active.get(request_id)
        if not ctx:
            return
        conversation_id = ctx["conversation_id"]
        correlation_id = ctx.get("correlation_id")
        base_req = {"task_kind": ctx["task_kind"], "requirements": ctx["requirements"], "request_id": request_id}
        try:
            plan = await self._phase("planning", self._client.plan, base_req)
            await self.emit_event(
                "deepcode_plan_ready",
                source_plugin=self.name,
                request_id=request_id,
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                timestamp=_utcnow(),
                plan=plan,
                confidence=float(plan.get("confidence", 0.72)),
            )
            refs = await self._phase("reference_collection", self._client.collect_references, plan)
            await self.emit_event(
                "deepcode_references_compiled",
                source_plugin=self.name,
                request_id=request_id,
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                timestamp=_utcnow(),
                references=refs,
                reference_count=len(refs.get("snippets", [])),
                confidence=float(refs.get("confidence", 0.78)),
            )
            impl = await self._phase("implementation_generation", self._client.generate_code, plan, refs)
            pid = impl.get("proposal_id")
            if not pid:
                pid = "dc_prop_" + sha256(
                    (json.dumps(plan, sort_keys=True) + "|" + json.dumps(refs, sort_keys=True) + "|" + request_id).encode("utf-8")
                ).hexdigest()[:16]
            diffs = normalize_diffs(impl.get("diffs", []), proposed_by=self.name)
            await self.emit_event(
                "deepcode_implementation_proposed",
                source_plugin=self.name,
                request_id=request_id,
                proposal_id=pid,
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                timestamp=_utcnow(),
                diffs=diffs,
                tests=impl.get("tests", []),
                docs=impl.get("docs", []),
                confidence=float(impl.get("confidence", 0.84)),
                dry_run=True,
            )
            validation = await self._phase("validation", self._client.validate, {"proposal_id": pid, "diffs": diffs})
            success = validation.get("status") == "pass"
            await self.emit_event(
                "deepcode_validation_report",
                source_plugin=self.name,
                request_id=request_id,
                proposal_id=pid,
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                timestamp=_utcnow(),
                validation=validation,
                success=bool(success),
                confidence=float(validation.get("confidence", 0.9)),
            )
            if success:
                await self.emit_event(
                    "deepcode_ready_for_apply",
                    source_plugin=self.name,
                    request_id=request_id,
                    proposal_id=pid,
                    conversation_id=conversation_id,
                    correlation_id=correlation_id,
                    timestamp=_utcnow(),
                    diffs=diffs,
                    tests=impl.get("tests", []),
                    docs=impl.get("docs", []),
                    validation_summary={
                        "lint_errors": validation.get("lint_errors", 0),
                        "tests_passed": validation.get("tests_passed", True),
                    },
                )
        except asyncio.CancelledError:
            await self.emit_event(
                "deepcode_pipeline_failed",
                source_plugin=self.name,
                request_id=request_id,
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                timestamp=_utcnow(),
                error="cancelled",
            )
            raise
        except Exception as e:
            logger.exception("DeepCode pipeline error")
            await self.emit_event(
                "deepcode_pipeline_failed",
                source_plugin=self.name,
                request_id=request_id,
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                timestamp=_utcnow(),
                error=str(e),
            )

    async def _phase(self, name: str, func, *args, **kwargs) -> Dict[str, Any]:
        if self._obs:
            try:
                async with self._obs.trace_operation(f"deepcode_{name}"):
                    return await func(*args, **kwargs)
            except Exception:
                return await func(*args, **kwargs)
        return await func(*args, **kwargs)

    def get_tools(self):
        return [{
            "name": "deepcode_request",
            "description": "Trigger a DeepCode generation flow (paper2code/text2web/text2backend). Emits diff-first proposal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_kind": {"type": "string", "description": "paper2code | text2web | text2backend | generic"},
                    "requirements": {"type": "string", "description": "Raw textual requirement or summary"},
                    "conversation_id": {"type": "string", "description": "Conversation/session identifier"},
                },
                "required": ["task_kind", "requirements"],
                "additionalProperties": False
            },
            "cost_hint": "medium",
            "latency_hint": "high",
            "safety_level": "medium",
            "test_reference": "tests/plugins/test_deepcode_orchestrator.py::test_emits_full_sequence",
            "category": "implementation",
            "complexity": "advanced",
            "version": "0.1.1",
            "dependencies": ["deepcode-hku (optional)"]
            ,
            "integration_requirements": "Diff-first only; apply gated by guardian/compliance route"
        }]

def create_plugin():
    return DeepCodeOrchestratorPlugin()
