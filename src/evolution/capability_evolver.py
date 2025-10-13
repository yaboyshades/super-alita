"""Capability evolution engine for autonomous self-improvement."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Iterable, Optional

from src.governance import ConstitutionalReasoner

logger = logging.getLogger(__name__)


class PerformanceMonitorLike:
    """Protocol subset for performance monitors used by the evolver."""

    async def get_recent_failures(self) -> list[dict[str, Any]]:  # pragma: no cover - protocol
        ...

    async def track_capability(self, capability_id: Any, deployment_context: dict[str, Any]) -> None:  # pragma: no cover - protocol
        ...


class CapabilityGeneratorLike:
    """Protocol subset for capability generators."""

    async def propose(self, gap: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


class CapabilityEvolutionEngine:
    """Iteratively propose, validate, and deploy new capabilities."""

    def __init__(
        self,
        *,
        constitutional_reasoner: ConstitutionalReasoner | None = None,
        performance_monitor: PerformanceMonitorLike | None = None,
        capability_generator: CapabilityGeneratorLike | None = None,
    ) -> None:
        self._reasoner = constitutional_reasoner or ConstitutionalReasoner()
        self._performance_monitor = performance_monitor
        self._capability_generator = capability_generator
        self._lock = asyncio.Lock()

    async def evolve_capabilities(self) -> list[dict[str, Any]]:
        """Run a single evolution cycle and return deployed capabilities."""

        async with self._lock:
            gaps = await self._identify_capability_gaps()
            deployed: list[dict[str, Any]] = []
            for gap in gaps:
                proposal = await self._propose_capability(gap)
                if not proposal:
                    continue
                approved, reasoning = await self._reasoner.evaluate_action(
                    proposed_action=proposal,
                    current_context={"type": "capability_evolution", "gap": gap},
                )
                proposal["constitutional_reasoning"] = reasoning
                if not approved:
                    logger.info("Capability proposal rejected: %s", reasoning)
                    continue
                test_results = await self._test_capability_safely(proposal)
                if not (test_results.get("safe") and test_results.get("effective")):
                    logger.info("Capability %s failed safe deployment test", proposal.get("id"))
                    continue
                await self._deploy_capability(proposal)
                await self._track_performance(proposal, test_results)
                deployed.append(proposal)
            return deployed

    async def _identify_capability_gaps(self) -> list[dict[str, Any]]:
        if not self._performance_monitor:
            return []
        try:
            failures = await self._performance_monitor.get_recent_failures()
        except Exception:
            logger.exception("Failed to fetch recent failures; skipping evolution cycle")
            return []
        gaps: list[dict[str, Any]] = []
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            if failure.get("suggested_capability"):
                gaps.append({
                    "description": failure.get("suggested_capability"),
                    "evidence": failure,
                })
        return gaps

    async def _propose_capability(self, gap: dict[str, Any]) -> dict[str, Any] | None:
        if not self._capability_generator:
            return {
                "id": gap.get("description", "unknown_capability"),
                "definition": gap,
                "ability": gap.get("description", "capability"),
            }
        try:
            proposal = await self._capability_generator.propose(gap)
        except Exception:
            logger.exception("Capability generator failed; skipping gap")
            return None
        if not isinstance(proposal, dict):
            return None
        proposal.setdefault("id", proposal.get("name") or gap.get("description"))
        proposal.setdefault("ability", proposal.get("id"))
        return proposal

    async def _test_capability_safely(self, proposal: dict[str, Any]) -> dict[str, Any]:
        # In lieu of a full sandbox, ensure the proposal declares conservative defaults.
        safe = not proposal.get("requires_root_access")
        effective = bool(proposal.get("success_metrics")) or bool(
            proposal.get("definition", {}).get("evidence")
        )
        return {"safe": safe, "effective": effective}

    async def _deploy_capability(self, proposal: dict[str, Any]) -> None:
        logger.info("Deploying capability %s", proposal.get("id"))

    async def _track_performance(
        self, proposal: dict[str, Any], deployment_context: dict[str, Any]
    ) -> None:
        if not self._performance_monitor:
            return
        try:
            await self._performance_monitor.track_capability(
                capability_id=proposal.get("id"), deployment_context=deployment_context
            )
        except Exception:
            logger.exception("Failed to record capability performance")
