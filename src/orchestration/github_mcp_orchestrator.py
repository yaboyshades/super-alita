"""High-level coordinator for GitHub MCP workflows inside Super-Alita."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.github_agent import GitHubAgent
from src.mcp.integrations.github_mcp_integration import (
    GitHubMCPIntegration,
    GitHubOperationReport,
)


@dataclass(slots=True)
class GitHubWorkflowResult:
    """Structured response describing a coordinated GitHub workflow."""

    workflow_type: str
    report: GitHubOperationReport
    context: dict[str, Any]


class GitHubMCPOrchestrator:
    """Coordinates multi-agent workflows that rely on GitHub MCP operations."""

    def __init__(
        self,
        *,
        constitutional_threshold: float = 0.75,
        integration: GitHubMCPIntegration | None = None,
    ) -> None:
        self.threshold = constitutional_threshold
        self.integration = integration or GitHubMCPIntegration(
            constitutional_threshold=constitutional_threshold
        )
        self.github_agent = GitHubAgent(
            constitutional_threshold=constitutional_threshold,
            integration=self.integration,
        )
        self._initialized = False

    async def ensure_initialized(self) -> None:
        if not self._initialized:
            await self.integration.ensure_initialized()
            self._initialized = True

    async def coordinate_workflow(
        self,
        workflow_type: str,
        context: dict[str, Any],
    ) -> GitHubWorkflowResult:
        """Entry point invoked by higher-level orchestrators."""

        await self.ensure_initialized()
        workflow_map = {
            "issue_analysis": self.github_agent.analyze_issue,
            "pull_request_review": self.github_agent.review_pull_request,
            "repository_analysis": self.github_agent.analyze_repository,
            "workflow_health": self.github_agent.inspect_actions_health,
        }
        if workflow_type not in workflow_map:
            raise ValueError(f"Unknown GitHub workflow: {workflow_type}")

        handler = workflow_map[workflow_type]
        report = await handler(context)
        return GitHubWorkflowResult(
            workflow_type=workflow_type,
            report=report,
            context=context,
        )


__all__ = ["GitHubMCPOrchestrator", "GitHubWorkflowResult"]
