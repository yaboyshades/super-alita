"""GitHub API integration for Super Alita cognitive agent.

Provides REST and GraphQL API integration for richer telemetry and data collection.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx

from src.core.schemas import (
    AttentionLevel,
    GitHubApiRequest,
    GitHubApiResponse,
    GitHubEventSchema,
    GitHubEventType,
    GitHubPriorityMetrics,
)


class GitHubApiClient:
    """GitHub API client with cognitive agent integration."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://api.github.com",
        timeout: float = 30.0,
        rate_limit_buffer: int = 100,
    ):
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_buffer = rate_limit_buffer

        # Rate limiting state
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = None

        # Request headers
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Super-Alita-Cognitive-Agent/1.0",
        }

        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

    async def make_request(self, request: GitHubApiRequest) -> GitHubApiResponse:
        """Make GitHub API request with rate limiting and error handling."""
        start_time = datetime.now(UTC).timestamp()

        try:
            # Check rate limits if enabled
            if request.rate_limit_aware and not self._can_make_request():
                await self._wait_for_rate_limit()

            # Prepare request
            url = f"{self.base_url}/{request.endpoint.lstrip('/')}"
            headers = {**self.headers, **request.headers}

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Make request
                if request.method.upper() == "GET":
                    response = await client.get(
                        url, headers=headers, params=request.parameters
                    )
                elif request.method.upper() == "POST":
                    response = await client.post(
                        url, headers=headers, json=request.parameters
                    )
                elif request.method.upper() == "PUT":
                    response = await client.put(
                        url, headers=headers, json=request.parameters
                    )
                elif request.method.upper() == "DELETE":
                    response = await client.delete(
                        url, headers=headers, params=request.parameters
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {request.method}")

                # Update rate limit info
                self._update_rate_limit_info(response)

                # Parse response
                execution_time = datetime.now(UTC).timestamp() - start_time

                if response.status_code < 400:
                    data = response.json() if response.content else None
                    return GitHubApiResponse(
                        success=True,
                        status_code=response.status_code,
                        data=data,
                        rate_limit_remaining=self.rate_limit_remaining,
                        rate_limit_reset=self.rate_limit_reset,
                        execution_time=execution_time,
                    )
                else:
                    error_msg = f"HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("message", error_msg)
                    except:
                        error_msg = response.text or error_msg

                    return GitHubApiResponse(
                        success=False,
                        status_code=response.status_code,
                        error=error_msg,
                        rate_limit_remaining=self.rate_limit_remaining,
                        rate_limit_reset=self.rate_limit_reset,
                        execution_time=execution_time,
                    )

        except httpx.TimeoutException:
            execution_time = datetime.now(UTC).timestamp() - start_time
            return GitHubApiResponse(
                success=False,
                status_code=408,
                error=f"Request timed out after {self.timeout} seconds",
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = datetime.now(UTC).timestamp() - start_time
            return GitHubApiResponse(
                success=False,
                status_code=500,
                error=f"Request failed: {str(e)}",
                execution_time=execution_time,
            )

    def _can_make_request(self) -> bool:
        """Check if we can make a request based on rate limits."""
        if self.rate_limit_remaining <= self.rate_limit_buffer:
            if self.rate_limit_reset:
                return datetime.now(UTC) >= self.rate_limit_reset
            return False
        return True

    async def _wait_for_rate_limit(self):
        """Wait for rate limit to reset."""
        if self.rate_limit_reset:
            wait_time = (
                self.rate_limit_reset - datetime.now(UTC)
            ).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(min(wait_time, 3600))  # Max 1 hour wait

    def _update_rate_limit_info(self, response: httpx.Response):
        """Update rate limit information from response headers."""
        try:
            self.rate_limit_remaining = int(
                response.headers.get("x-ratelimit-remaining", 5000)
            )
            reset_timestamp = response.headers.get("x-ratelimit-reset")
            if reset_timestamp:
                self.rate_limit_reset = datetime.fromtimestamp(
                    int(reset_timestamp), UTC
                )
        except (ValueError, TypeError):
            pass

    async def get_repository_info(self, repository: str) -> GitHubApiResponse:
        """Get repository information."""
        request = GitHubApiRequest(endpoint=f"repos/{repository}", method="GET")
        return await self.make_request(request)

    async def list_issues(
        self,
        repository: str,
        state: str = "open",
        labels: list[str] | None = None,
        limit: int = 30,
    ) -> GitHubApiResponse:
        """List repository issues."""
        params = {"state": state, "per_page": min(limit, 100)}
        if labels:
            params["labels"] = ",".join(labels)

        request = GitHubApiRequest(
            endpoint=f"repos/{repository}/issues", method="GET", parameters=params
        )
        return await self.make_request(request)

    async def list_pull_requests(
        self, repository: str, state: str = "open", limit: int = 30
    ) -> GitHubApiResponse:
        """List repository pull requests."""
        params = {"state": state, "per_page": min(limit, 100)}

        request = GitHubApiRequest(
            endpoint=f"repos/{repository}/pulls", method="GET", parameters=params
        )
        return await self.make_request(request)

    async def get_pull_request_files(
        self, repository: str, pr_number: int
    ) -> GitHubApiResponse:
        """Get files changed in a pull request."""
        request = GitHubApiRequest(
            endpoint=f"repos/{repository}/pulls/{pr_number}/files", method="GET"
        )
        return await self.make_request(request)

    async def list_commits(
        self, repository: str, since: datetime | None = None, limit: int = 30
    ) -> GitHubApiResponse:
        """List repository commits."""
        params = {"per_page": min(limit, 100)}
        if since:
            params["since"] = since.isoformat()

        request = GitHubApiRequest(
            endpoint=f"repos/{repository}/commits", method="GET", parameters=params
        )
        return await self.make_request(request)

    async def get_workflow_runs(
        self, repository: str, status: str | None = None, limit: int = 30
    ) -> GitHubApiResponse:
        """Get workflow runs for repository."""
        params = {"per_page": min(limit, 100)}
        if status:
            params["status"] = status

        request = GitHubApiRequest(
            endpoint=f"repos/{repository}/actions/runs", method="GET", parameters=params
        )
        return await self.make_request(request)

    async def extract_priority_metrics(
        self, repository: str, item_type: str, item_number: int
    ) -> GitHubPriorityMetrics:
        """Extract GitHub-specific priority metrics for an item."""

        metrics = GitHubPriorityMetrics()

        try:
            if item_type == "pull_request":
                # Get PR details
                pr_response = await self.make_request(
                    GitHubApiRequest(
                        endpoint=f"repos/{repository}/pulls/{item_number}", method="GET"
                    )
                )

                if pr_response.success and pr_response.data:
                    pr_data = pr_response.data

                    # Extract basic metrics
                    metrics.comment_count = pr_data.get("comments", 0)
                    metrics.review_count = pr_data.get("review_comments", 0)
                    metrics.lines_changed = pr_data.get("additions", 0) + pr_data.get(
                        "deletions", 0
                    )
                    metrics.file_changes_count = pr_data.get("changed_files", 0)

                    # Check for merge conflicts
                    metrics.merge_conflicts = pr_data.get("mergeable", True) is False

                    # Extract labels
                    labels = [label["name"] for label in pr_data.get("labels", [])]
                    metrics.issue_labels = labels

                    # Check for security/critical labels
                    security_labels = ["security", "critical", "urgent", "hotfix"]
                    metrics.has_security_alert = any(
                        label.lower() in security_labels for label in labels
                    )

                    # Check for stakeholder mentions in body
                    body = pr_data.get("body", "")
                    stakeholder_keywords = [
                        "@team",
                        "@admin",
                        "@maintainer",
                        "urgent",
                        "critical",
                    ]
                    metrics.has_stakeholder_mention = any(
                        keyword in body.lower() for keyword in stakeholder_keywords
                    )

                    # Get CI status
                    status_response = await self.make_request(
                        GitHubApiRequest(
                            endpoint=f"repos/{repository}/commits/{pr_data['head']['sha']}/status",
                            method="GET",
                        )
                    )

                    if status_response.success and status_response.data:
                        metrics.ci_status = status_response.data.get("state", "unknown")

                # Get files to check dependencies
                files_response = await self.get_pull_request_files(
                    repository, item_number
                )
                if files_response.success and files_response.data:
                    # Check if this PR blocks others by modifying critical files
                    critical_files = [
                        "package.json",
                        "requirements.txt",
                        "Dockerfile",
                        "Makefile",
                    ]
                    modified_files = [f["filename"] for f in files_response.data]

                    metrics.blocks_other_prs = any(
                        any(critical in file for critical in critical_files)
                        for file in modified_files
                    )

            elif item_type == "issue":
                # Get issue details
                issue_response = await self.make_request(
                    GitHubApiRequest(
                        endpoint=f"repos/{repository}/issues/{item_number}",
                        method="GET",
                    )
                )

                if issue_response.success and issue_response.data:
                    issue_data = issue_response.data

                    metrics.comment_count = issue_data.get("comments", 0)

                    # Extract labels
                    labels = [label["name"] for label in issue_data.get("labels", [])]
                    metrics.issue_labels = labels

                    # Check for security/critical labels
                    security_labels = ["bug", "security", "critical", "urgent"]
                    metrics.has_security_alert = any(
                        label.lower() in security_labels for label in labels
                    )

                    # Check for stakeholder mentions
                    body = issue_data.get("body", "")
                    stakeholder_keywords = [
                        "@team",
                        "@admin",
                        "@maintainer",
                        "urgent",
                        "critical",
                    ]
                    metrics.has_stakeholder_mention = any(
                        keyword in body.lower() for keyword in stakeholder_keywords
                    )

        except Exception:
            # Return default metrics on error
            pass

        return metrics

    async def create_github_event_from_api_data(
        self, repository: str, api_data: dict[str, Any], event_type: GitHubEventType
    ) -> GitHubEventSchema:
        """Create GitHub event schema from API data."""

        return GitHubEventSchema(
            event_type=event_type,
            repository=repository,
            actor=api_data.get("user", {}).get("login", "unknown"),
            payload=api_data,
            event_id=f"api-{api_data.get('id', 'unknown')}-{datetime.now(UTC).isoformat()}",
            attention_level=self._calculate_attention_level(api_data),
            processing_status="api_captured",
        )

    def _calculate_attention_level(self, api_data: dict[str, Any]) -> AttentionLevel:
        """Calculate attention level based on API data."""

        # Check for critical indicators
        labels = [label["name"].lower() for label in api_data.get("labels", [])]
        critical_labels = ["critical", "urgent", "security", "hotfix"]

        if any(label in critical_labels for label in labels):
            return AttentionLevel.CRITICAL

        # Check for high priority indicators
        high_priority_labels = ["bug", "enhancement", "important"]
        if any(label in high_priority_labels for label in labels):
            return AttentionLevel.HIGH

        # Check comment/activity level
        comment_count = api_data.get("comments", 0)
        if comment_count > 10:
            return AttentionLevel.HIGH
        elif comment_count > 3:
            return AttentionLevel.MEDIUM

        return AttentionLevel.LOW
