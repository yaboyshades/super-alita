"""Distributed cognition helpers for coordinating specialised agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass(slots=True)
class AgentCallbacks:
    """Container for agent-specific coordination hooks."""

    analyze: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
    share: Callable[[dict[str, dict[str, Any]], dict[str, Any]], Awaitable[None]]
    synthesize: Callable[[dict[str, dict[str, Any]], dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class AgentState:
    """Track recent contributions for auditing and debugging."""

    contributions: dict[str, Any] = field(default_factory=dict)
    last_result: Optional[dict[str, Any]] = None


class DistributedCognitionOrchestrator:
    """Coordinate multi-agent collaboration via a shared workspace.

    The structure mirrors blackboard-style coordination architectures used in
    cooperative multi-agent planning (Durfee & Montgomery, 1991).
    """

    def __init__(self) -> None:
        self.shared_workspace: dict[str, Any] = {}
        self.agent_states: dict[str, AgentState] = {}
        self._callbacks: dict[str, AgentCallbacks] = {}
        self._lock = asyncio.Lock()

    def register_agent(self, agent_id: str, callbacks: AgentCallbacks) -> None:
        """Register callbacks enabling the orchestrator to coordinate an agent."""

        if not agent_id:
            raise ValueError("agent_id must be provided")
        self._callbacks[agent_id] = callbacks
        self.agent_states.setdefault(agent_id, AgentState())

    async def coordinate_problem_solving(
        self,
        problem: str,
        agents_available: List[str],
    ) -> dict[str, Any]:
        """Execute a three-phase collaborative problem-solving protocol."""

        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("problem must be a non-empty string")
        if not agents_available:
            raise ValueError("agents_available must contain at least one agent")

        analyses = await self._phase_independent_analysis(problem, agents_available)
        await self._phase_cross_pollination(analyses, agents_available)
        solution = await self._phase_synthesis(analyses)
        return solution

    async def _phase_independent_analysis(
        self, problem: str, agents: List[str]
    ) -> dict[str, dict[str, Any]]:
        tasks = []
        for agent_id in agents:
            callbacks = self._callbacks.get(agent_id)
            if callbacks is None:
                raise KeyError(f"Agent '{agent_id}' is not registered")
            task = asyncio.create_task(
                callbacks.analyze(problem, dict(self.shared_workspace))
            )
            tasks.append((agent_id, task))
        analyses: dict[str, dict[str, Any]] = {}
        for agent_id, task in tasks:
            result = await task
            if not isinstance(result, dict):
                raise TypeError("Agent analyses must return dictionaries")
            analyses[agent_id] = result
            # Update shared workspace with declared contributions
            contributions = result.get("contributions", {})
            if isinstance(contributions, dict):
                self.shared_workspace.update(contributions)
                self.agent_states[agent_id].contributions.update(contributions)
            self.agent_states[agent_id].last_result = result
        return analyses

    async def _phase_cross_pollination(
        self, analyses: dict[str, dict[str, Any]], agents: List[str]
    ) -> None:
        for agent_id in agents:
            callbacks = self._callbacks.get(agent_id)
            if callbacks is None:
                raise KeyError(f"Agent '{agent_id}' is not registered")
            contribution = await callbacks.share(analyses, dict(self.shared_workspace))
            if isinstance(contribution, dict):
                new_items = contribution.get("contributions", contribution)
                if isinstance(new_items, dict):
                    self.shared_workspace.update(new_items)
                    self.agent_states[agent_id].contributions.update(new_items)

    async def _phase_synthesis(
        self, analyses: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        if not analyses:
            return {"summary": "No analyses provided"}
        # Select agent with highest contribution footprint to lead synthesis
        agent_id = max(
            analyses.keys(),
            key=lambda aid: len(self.agent_states[aid].contributions),
        )
        callbacks = self._callbacks[agent_id]
        return await callbacks.synthesize(analyses, dict(self.shared_workspace))
