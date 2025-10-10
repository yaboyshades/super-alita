from __future__ import annotations

"""Agent wrapper that coordinates GitHub MCP operations for Super-Alita."""

from dataclasses import dataclass
from typing import Any

from src.constitutional.scorer import ConstitutionalScorer
from src.mcp.integrations.github_mcp_integration import (
    GitHubMCPIntegration,
    GitHubOperationReport,
)


@dataclass(slots=True)
class GitHubAgentContext:
    """Container describing the inputs passed to the GitHub agent."""

    repository: str | None = None
    issue_number: int | None = None
    workflow_id: str | None = None
    analysis_type: str = "architecture"
    focus_area: str | None = None


class GitHubAgent:
    """Single-responsibility agent for orchestrating GitHub MCP interactions."""

    def __init__(
        self,
        *,
        constitutional_threshold: float = 0.75,
        integration: GitHubMCPIntegration,
    ) -> None:
        self.integration = integration
        self.constitutional_threshold = constitutional_threshold
        self.scorer = ConstitutionalScorer(
            compliance_threshold=constitutional_threshold
        )

    async def analyze_issue(
        self, context: dict[str, Any]
    ) -> GitHubOperationReport:
        repo = context.get("repository")
        summary = self._summarize("issue_analysis", context)
        self._assert_constitutional_intent(summary)
        payload = {
            "uri": "github://issues",
        }
        report = await self.integration.execute_operation(
            "read_resource", payload
        )
        report.data = {
            "repository": repo,
            "issues": report.data,
        }
        return report

    async def review_pull_request(
        self, context: dict[str, Any]
    ) -> GitHubOperationReport:
        repo = context.get("repository")
        workflow_id = context.get("workflow_id")
        summary = self._summarize("pull_request_review", context)
        self._assert_constitutional_intent(summary)
        payload = {
            "repo": repo,
            "workflow_id": workflow_id,
        }
        return await self.integration.execute_operation(
            "get_workflow_runs", payload
        )

    async def analyze_repository(
        self, context: dict[str, Any]
    ) -> GitHubOperationReport:
        summary = self._summarize("repository_analysis", context)
        self._assert_constitutional_intent(summary)
        payload = {
            "analysis_type": context.get("analysis_type", "architecture"),
            "focus_area": context.get("focus_area"),
        }
        return await self.integration.execute_operation(
            "analyze_context", payload
        )

    async def inspect_actions_health(
        self, context: dict[str, Any]
    ) -> GitHubOperationReport:
        summary = self._summarize("workflow_health", context)
        self._assert_constitutional_intent(summary)
        payload = {
            "repo": context.get("repository"),
            "status": context.get("status"),
        }
        return await self.integration.execute_operation(
            "get_workflow_runs", payload
        )

    def _summarize(self, workflow: str, context: dict[str, Any]) -> str:
        return (
            f"GitHub workflow: {workflow}. "
            f"Context keys: {sorted(context.keys())}. "
            f"Threshold: {self.constitutional_threshold}."
        )

    def _assert_constitutional_intent(self, text: str) -> None:
        result = self.scorer.score_specification(text)
        if not result.is_compliant:
            raise RuntimeError(
                "Requested GitHub workflow violates constitutional intent gates"
            )


__all__ = ["GitHubAgent", "GitHubAgentContext"]
