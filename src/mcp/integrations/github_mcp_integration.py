"""Constitutionally-aware GitHub MCP integration helpers."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.constitutional.scorer import (
    ConstitutionalResult,
    ConstitutionalScorer,
)
from src.mcp.server.github_tools import GitHubMCPTools
from src.mcp.server.main import SuperAlitaMCPServer
from src.orchestration.reliability_manager import (
    ReliabilityConfig,
    ReliabilityManager,
)


@dataclass(slots=True)
class GitHubIntegrationStatus:
    """Runtime details captured when bootstrapping the GitHub MCP integration."""

    valid: bool
    constitutional_score: float
    token_present: bool
    details: dict[str, Any]
    violations: list[dict[str, Any]]


@dataclass(slots=True)
class GitHubOperationReport:
    """Structured response returned after executing a GitHub MCP operation."""

    success: bool
    operation: str
    data: Any
    constitutional_score: float
    entry_score: float
    exit_score: float
    reliability_snapshot: dict[str, Any]
    violations: list[dict[str, Any]]


class GitHubMCPIntegration:
    """High-level facade that wires GitHub MCP tools into Super-Alita workflows."""

    _DEFAULT_TIMEOUT_S = 25

    def __init__(
        self,
        *,
        constitutional_threshold: float = 0.75,
        github_token: str | None = None,
        reliability: ReliabilityManager | None = None,
    ) -> None:
        self.constitutional_threshold = constitutional_threshold
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.scorer = ConstitutionalScorer(
            compliance_threshold=constitutional_threshold
        )
        self.tools = GitHubMCPTools(github_token=self.github_token)
        self.reliability = reliability or ReliabilityManager(
            ReliabilityConfig()
        )
        self.server: SuperAlitaMCPServer | None = None
        self._init_status: GitHubIntegrationStatus | None = None

    @property
    def initialized(self) -> bool:
        return self._init_status is not None and self._init_status.valid

    async def initialize(self) -> GitHubIntegrationStatus:
        """Validate the environment and warm the embedded MCP server."""

        status = await self._validate_environment()
        if not status.valid:
            self._init_status = status
            raise RuntimeError(
                "GitHub MCP integration failed constitutional validation"
            )

        # Lazily instantiate the MCP server once validation passes.
        if self.server is None:
            self.server = SuperAlitaMCPServer()
            self.server.warmup()
        self._init_status = status
        return status

    async def ensure_initialized(self) -> GitHubIntegrationStatus:
        """Bootstrap the integration on-demand."""

        if self.initialized:
            assert self._init_status is not None
            return self._init_status
        return await self.initialize()

    async def execute_operation(
        self,
        operation: str,
        parameters: dict[str, Any] | None = None,
        *,
        timeout_s: int | None = None,
    ) -> GitHubOperationReport:
        """Execute a GitHub MCP operation with constitutional entry/exit gates."""

        params = dict(parameters or {})
        timeout = timeout_s or self._DEFAULT_TIMEOUT_S

        # Entry gate: synthesize a specification-style summary for scoring.
        entry_summary = self._build_operation_summary(
            "entry", operation, params
        )
        entry_result = self.scorer.score_specification(entry_summary)
        if not entry_result.is_compliant:
            return self._as_report(
                success=False,
                operation=operation,
                payload=None,
                reliability_snapshot={
                    "status": "skipped",
                    "reason": "entry_gate_failure",
                },
                entry_result=entry_result,
                exit_result=None,
            )

        async def _runner() -> Any:
            return await self._invoke_operation(operation, params)

        reliability_payload = await self.reliability.execute_with_retries(
            stage=f"github.{operation}",
            coro_fn=_runner,
            timeout_s=timeout,
            emit_cb=None,
        )

        if reliability_payload.get("status") != "success":
            return self._as_report(
                success=False,
                operation=operation,
                payload=reliability_payload.get("error"),
                reliability_snapshot=reliability_payload,
                entry_result=entry_result,
                exit_result=None,
            )

        payload = reliability_payload.get("output")
        exit_summary = self._build_operation_summary(
            "exit", operation, payload
        )
        exit_result = self.scorer.score_specification(exit_summary)

        success = exit_result.is_compliant and entry_result.is_compliant
        return self._as_report(
            success=success,
            operation=operation,
            payload=payload,
            reliability_snapshot=reliability_payload,
            entry_result=entry_result,
            exit_result=exit_result,
        )

    async def _invoke_operation(
        self, operation: str, params: dict[str, Any]
    ) -> Any:
        """Dispatch to the underlying GitHubMCPTools implementation."""

        operation_map: dict[str, Callable[..., Any]] = {
            "list_resources": self.tools.list_resources,
            "read_resource": self.tools.read_resource,
            "create_issue": self.tools.create_issue,
            "create_pull_request": self.tools.create_pull_request,
            "search_code": self.tools.search_code,
            "get_workflow_runs": self.tools.get_workflow_runs,
            "analyze_context": self.tools.analyze_super_alita_context,
        }
        if operation not in operation_map:
            raise ValueError(f"Unknown GitHub MCP operation: {operation}")

        call = operation_map[operation]
        clean_params = dict(params)
        result = call(**clean_params) if clean_params else call()
        if inspect.isawaitable(result):
            return await result
        return result

    async def _validate_environment(self) -> GitHubIntegrationStatus:
        """Gather validation telemetry and compute constitutional score."""

        token_present = bool(self.github_token)
        resources: list[dict[str, Any]] = []
        violation_snapshots: list[dict[str, Any]] = []
        try:
            resources = await self._soft_list_resources()
        except Exception as exc:  # noqa: BLE001
            violation_snapshots.append(
                {
                    "article": "Article IV",
                    "principle": "Integration-First Testing",
                    "message": f"Failed to connect to GitHub: {exc}",
                    "severity": "high",
                }
            )

        intent_summary = self._build_validation_summary(
            token_present, resources
        )
        result = self.scorer.score_specification(intent_summary)
        violations = [
            {
                "article": v.article,
                "principle": v.principle,
                "message": v.message,
                "severity": v.severity,
            }
            for v in result.violations
        ]
        violations.extend(violation_snapshots)

        valid = (
            token_present and result.is_compliant and not violation_snapshots
        )
        return GitHubIntegrationStatus(
            valid=valid,
            constitutional_score=result.overall_score,
            token_present=token_present,
            details={
                "resource_count": len(resources),
                "resources": resources,
            },
            violations=violations,
        )

    async def _soft_list_resources(self) -> list[dict[str, Any]]:
        """Attempt to list GitHub resources without letting failures bubble loudly."""

        # list_resources is synchronous; read_resource provides async validation.
        resources = self.tools.list_resources()
        if not resources:
            return []

        # Probe the first resource to ensure token validity.
        first = resources[0]["uri"]
        try:
            await asyncio.wait_for(self.tools.read_resource(first), timeout=8)
        except Exception:  # noqa: BLE001
            # If the probe fails we still return the metadata; higher layers
            # will record the failure so contributors can address credentials.
            pass
        return resources

    def _build_validation_summary(
        self,
        token_present: bool,
        resources: list[dict[str, Any]],
    ) -> str:
        resource_names = (
            ", ".join(r.get("name", "unknown") for r in resources) or "none"
        )
        return (
            "## GitHub MCP Constitutional Assessment\n"
            f"- token_present: {token_present}\n"
            f"- constitutional_threshold: {self.constitutional_threshold}\n"
            f"- declared_resources: {resource_names}\n"
            "- gate_requirements: integration-first, clarity, counterfactual\n"
            "Ensure GitHub interactions maintain ≥0.75 compliance and respect"
            " neural orchestration policies."
        )

    def _build_operation_summary(
        self,
        phase: str,
        operation: str,
        payload: Any,
    ) -> str:
        return (
            f"### GitHub MCP {phase} summary\n"
            f"operation: {operation}\n"
            f"payload_snapshot: {repr(payload)[:400]}\n"
            f"constitutional_threshold: {self.constitutional_threshold}\n"
            "articles: [I, II, III, IV, V, VI]\n"
            "focus: maintain event sourcing audit trail and avoid privileged writes"
        )

    def _as_report(
        self,
        *,
        success: bool,
        operation: str,
        payload: Any,
        reliability_snapshot: dict[str, Any],
        entry_result: ConstitutionalResult,
        exit_result: ConstitutionalResult | None,
    ) -> GitHubOperationReport:
        exit_score = exit_result.overall_score if exit_result else 0.0
        violations = []
        for source in (
            entry_result.violations,
            exit_result.violations if exit_result else [],
        ):
            for item in source:
                violations.append(
                    {
                        "article": item.article,
                        "principle": item.principle,
                        "message": item.message,
                        "severity": item.severity,
                    }
                )
        return GitHubOperationReport(
            success=success,
            operation=operation,
            data=payload,
            constitutional_score=min(entry_result.overall_score, exit_score),
            entry_score=entry_result.overall_score,
            exit_score=exit_score,
            reliability_snapshot=reliability_snapshot,
            violations=violations,
        )


__all__ = [
    "GitHubMCPIntegration",
    "GitHubIntegrationStatus",
    "GitHubOperationReport",
]
