#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.core.plugin_interface import PluginInterface

try:
    from src.core.observability import ObservabilityManager  # type: ignore
except Exception:  # pragma: no cover
    ObservabilityManager = None  # type: ignore

from src.core.diff_utils import normalize_diffs

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


@runtime_checkable
class DeepCodeClientInterface(Protocol):
    async def plan(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    async def collect_references(
        self, plan: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...
    async def generate_code(
        self, plan: Mapping[str, Any], references: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...
    async def validate(
        self, implementation: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class _StubDeepCodeClient(DeepCodeClientInterface):
    async def plan(self, request: Mapping[str, Any]) -> Mapping[str, Any]:  # noqa: D401
        return {
            "steps": ["draft plan", "produce impl"],
            "confidence": 0.73,
            "request": request,
        }

    async def collect_references(
        self, plan: Mapping[str, Any]
    ) -> Mapping[str, Any]:  # noqa: D401, ARG002
        return {"snippets": [], "confidence": 0.78}

    async def generate_code(  # noqa: D401, ARG002
        self, plan: Mapping[str, Any], references: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return {
            "proposal_id": None,
            "diffs": [
                {
                    "path": "docs/deepcode_result.md",
                    "change_type": "add",
                    "unified_diff": (
                        "--- /dev/null\n+++ b/docs/deepcode_result.md\n@@\n+# Result\n+Hello DeepCode\n"
                    ),
                    "new_content": "# Result\n\nHello DeepCode\n",
                    "confidence": 0.84,
                },
                {
                    "path": "src/new_dir/deepcode_module.py",
                    "change_type": "add",
                    "unified_diff": (
                        "--- /dev/null\n+++ b/src/new_dir/deepcode_module.py\n@@\n+def greet():\n+    return 'deepcode'\n"
                    ),
                    "new_content": 'def greet():\n    return "deepcode"\n',
                    "confidence": 0.83,
                },
            ],
            "tests": [
                {
                    "path": "tests/test_deepcode_result.py",
                    "content": "def test_ok(): assert True",
                },
                {
                    "path": "tests/new_dir/test_deepcode_module.py",
                    "content": (
                        "from src.new_dir.deepcode_module import greet\n\n"
                        'def test_greet():\n    assert greet() == "deepcode"'
                    ),
                },
            ],
            "docs": [
                {"path": "docs/plan.json", "content": json.dumps(plan)},
                {
                    "path": "docs/new_dir/summary.md",
                    "content": "# Summary\nGenerated module\n",
                },
            ],
            "confidence": 0.84,
        }

    async def validate(  # noqa: D401, ARG002
        self, implementation: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return {
            "status": "pass",
            "lint_errors": 0,
            "tests_passed": True,
            "confidence": 0.91,
        }


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
        # Allow environment-configured real client
        self._client: DeepCodeClientInterface = client or self._maybe_real_client()
        self._active: dict[str, dict[str, Any]] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._obs: ObservabilityManager | None = None  # type: ignore[name-defined]
        self._latest: dict[str, Any] | None = None
        self._latest_path = Path(
            os.getenv("DEEPCODE_LATEST_PATH", "logs/deepcode_latest.json")
        )
        self._latest_path.parent.mkdir(parents=True, exist_ok=True)

    def _maybe_real_client(self) -> DeepCodeClientInterface:
        try:  # local import to avoid hard dep if unused
            from src.deepcode.client import DeepCodeHTTPClient  # type: ignore
        except Exception:  # pragma: no cover
            return _StubDeepCodeClient()
        client = DeepCodeHTTPClient()
        if getattr(client, "enabled", False):  # type: ignore[attr-defined]
            logger.info("DeepCode HTTP client enabled (DEEPCODE_API_URL configured)")
            return client  # type: ignore[return-value]
        return _StubDeepCodeClient()

    @property
    def name(self) -> str:
        return "deepcode_orchestrator"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        self._obs = (
            config.get("observability_manager") if isinstance(config, dict) else None
        )
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

    async def _on_request(self, event: dict[str, Any]) -> None:
        if not self.is_running:
            return
        request_id = event.get("request_id") or (
            f"dc_req_{int(datetime.now(UTC).timestamp() * 1000)}"
        )
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
        task: asyncio.Task[None] = asyncio.create_task(self._run_pipeline(request_id))
        self._tasks.append(task)
        task.add_done_callback(
            lambda t: self._tasks.remove(t) if t in self._tasks else None
        )

    async def _run_pipeline(self, request_id: str) -> None:
        ctx = self._active.get(request_id)
        if not ctx:
            return
        conversation_id = ctx["conversation_id"]
        correlation_id = ctx.get("correlation_id")
        base_req = {
            "task_kind": ctx["task_kind"],
            "requirements": ctx["requirements"],
            "request_id": request_id,
        }
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
            refs = await self._phase(
                "reference_collection", self._client.collect_references, plan
            )
            await self.emit_event(
                "deepcode_references_collected",
                source_plugin=self.name,
                request_id=request_id,
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                timestamp=_utcnow(),
                references=refs,
                reference_count=len(refs.get("snippets", [])),
                confidence=float(refs.get("confidence", 0.78)),
            )
            impl = await self._phase(
                "implementation_generation",
                self._client.generate_code,
                plan,
                refs,
            )
            pid = impl.get("proposal_id")
            if not pid:
                pid = (
                    "dc_prop_"
                    + sha256(
                        (
                            json.dumps(plan, sort_keys=True)
                            + "|"
                            + json.dumps(refs, sort_keys=True)
                            + "|"
                            + request_id
                        ).encode("utf-8")
                    ).hexdigest()[:16]
                )
            diffs = normalize_diffs(impl.get("diffs", []), proposed_by=self.name)

            # --- Advanced analysis: risk scoring (moved here after diffs) ---
            def _score_diff(d: dict[str, Any]) -> dict[str, Any]:
                unified = d.get("unified_diff") or d.get("diff") or ""
                path = d.get("path") or d.get("file") or "unknown"
                lines_added = sum(
                    1
                    for ln in unified.splitlines()
                    if ln.startswith("+") and not ln.startswith("+++")
                )
                lines_removed = sum(
                    1
                    for ln in unified.splitlines()
                    if ln.startswith("-") and not ln.startswith("---")
                )
                size_factor = min((lines_added + lines_removed) / 80.0, 1.0)
                ext = os.path.splitext(str(path))[1].lower()
                ext_factor = (
                    0.4
                    if ext in {".py", ".rs", ".ts"}
                    else 0.2 if ext in {".md", ".json"} else 0.3
                )
                critical_tokens = [
                    "eval(",
                    "exec(",
                    "subprocess",
                    "os.system",
                    "pickle.loads",
                    "open(",
                ]
                token_hits = [tok for tok in critical_tokens if tok in unified]
                token_factor = min(len(token_hits) * 0.2, 0.6)
                test_related = path.startswith("tests/") or "test_" in os.path.basename(
                    str(path)
                )
                test_factor = -0.15 if test_related else 0.0
                raw_score = size_factor + ext_factor + token_factor + test_factor
                score = max(0.0, min(raw_score, 1.0))
                factors: dict[str, Any] = {
                    "size_factor": round(size_factor, 3),
                    "ext_factor": ext_factor,
                    "token_factor": round(token_factor, 3),
                    "test_factor": test_factor,
                }
                if token_hits:
                    factors["tokens"] = token_hits
                return {**d, "risk_score": round(score, 3), "risk_factors": factors}

            enriched_diffs = [_score_diff(d) for d in diffs]
            risk_summary: dict[str, Any]
            if enriched_diffs:
                avg_risk = sum(d.get("risk_score", 0.0) for d in enriched_diffs) / len(
                    enriched_diffs
                )
                high_risk = [d for d in enriched_diffs if d.get("risk_score", 0) >= 0.7]
                suggested_tests: list[str] = []
                for d in enriched_diffs:
                    pth = str(d.get("path", ""))
                    if pth.endswith(".py") and "tests/" not in pth:
                        candidate = "tests/test_" + os.path.basename(pth).replace(
                            ".py", ".py"
                        )
                        suggested_tests.append(candidate)
                risk_summary = {
                    "average_risk": round(avg_risk, 3),
                    "high_risk_count": len(high_risk),
                    "suggested_tests": sorted(set(suggested_tests))[:25],
                }
            else:
                risk_summary = {
                    "average_risk": 0.0,
                    "high_risk_count": 0,
                    "suggested_tests": [],
                }
            await self.emit_event(
                "deepcode_implementation_proposed",
                source_plugin=self.name,
                request_id=request_id,
                proposal_id=pid,
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                timestamp=_utcnow(),
                diffs=enriched_diffs,
                tests=impl.get("tests", []),
                docs=impl.get("docs", []),
                confidence=float(impl.get("confidence", 0.84)),
                dry_run=True,
            )
            validation = await self._phase(
                "validation",
                self._client.validate,
                {"proposal_id": pid, "diffs": enriched_diffs},
            )
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
                    diffs=enriched_diffs,
                    tests=impl.get("tests", []),
                    docs=impl.get("docs", []),
                    validation_summary={
                        "lint_errors": validation.get("lint_errors", 0),
                        "tests_passed": validation.get("tests_passed", True),
                    },
                )
                # Persist latest proposal for HTTP retrieval
                self._latest = {
                    "request_id": request_id,
                    "proposal_id": pid,
                    "plan": plan,
                    "references": refs,
                    "diffs": enriched_diffs,
                    "tests": impl.get("tests", []),
                    "docs": impl.get("docs", []),
                    "validation": validation,
                    "risk_summary": risk_summary,
                    "success": success,
                    "stored_at": _utcnow(),
                }
                try:
                    self._latest_path.write_text(
                        json.dumps(self._latest, indent=2), encoding="utf-8"
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning("Failed writing latest deepcode proposal: %s", e)
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

    async def _phase(
        self,
        name: str,
        func: Callable[..., Awaitable[Mapping[str, Any]]],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self._obs:
            try:
                async with self._obs.trace_operation(f"deepcode_{name}"):
                    out = await func(*args, **kwargs)
                    return dict(out)
            except Exception:
                out = await func(*args, **kwargs)
                return dict(out)
        out = await func(*args, **kwargs)
        return dict(out)

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "deepcode_request",
                "description": (
                    "Trigger a DeepCode generation flow (paper2code/text2web/"
                    "text2backend). Emits diff-first proposal."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_kind": {
                            "type": "string",
                            "description": (
                                "paper2code | text2web | text2backend | generic"
                            ),
                        },
                        "requirements": {
                            "type": "string",
                            "description": "Raw textual requirement or summary",
                        },
                        "conversation_id": {
                            "type": "string",
                            "description": "Conversation/session identifier",
                        },
                    },
                    "required": ["task_kind", "requirements"],
                    "additionalProperties": False,
                },
                "cost_hint": "medium",
                "latency_hint": "high",
                "safety_level": "medium",
                "test_reference": (
                    "tests/plugins/test_deepcode_orchestrator.py::"
                    "test_emits_full_sequence"
                ),
                "category": "implementation",
                "complexity": "advanced",
                "version": "0.1.1",
                "dependencies": ["deepcode-hku (optional)"],
                "integration_requirements": (
                    "Diff-first only; apply gated by guardian/compliance route"
                ),
            }
        ]

    # Public accessors for HTTP layer
    def get_latest(self) -> dict[str, Any] | None:
        if self._latest is not None:
            return self._latest
        if self._latest_path.exists():  # lazy load if available
            try:
                self._latest = json.loads(self._latest_path.read_text(encoding="utf-8"))
                return self._latest
            except Exception:  # pragma: no cover
                return None
        return None

    async def apply_latest(
        self, filter_paths: list[str] | None = None
    ) -> dict[str, Any]:
        latest = self.get_latest()
        if not latest:
            return {"status": "no-proposal"}
        diffs = latest.get("diffs", [])
        if filter_paths:
            diffs = [d for d in diffs if d.get("path") in set(filter_paths)]
        # Emit event for guardian/other plugins to actually apply
        await self.emit_event(
            "deepcode_apply",
            source_plugin=self.name,
            proposal_id=latest.get("proposal_id"),
            request_id=latest.get("request_id"),
            file_count=len(diffs),
            timestamp=_utcnow(),
        )
        # For now just return diffs; real apply logic handled elsewhere
        return {
            "status": "ok",
            "applied": False,
            "file_count": len(diffs),
            "diffs": diffs,
        }


def create_plugin():
    return DeepCodeOrchestratorPlugin()
