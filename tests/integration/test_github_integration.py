"""Tests for GitHub integration components in Super Alita cognitive agent."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.github_priority_calculator import (
    EnhancedPriorityMetrics,
    GitHubPriorityCalculator,
)
from src.core.schemas import (
    AttentionLevel,
    GitHubEventSchema,
    GitHubEventType,
    GitHubPriorityMetrics,
)
from src.integration.github_api import GitHubApiClient
from src.tools.github_cli_tool import GitHubCliInput, GitHubCliTool


class TestGitHubEventSchema:
    """Test GitHub event schema validation."""

    def test_github_event_creation(self):
        """Test creating GitHub event schema."""
        event = GitHubEventSchema(
            event_type=GitHubEventType.PR_OPENED,
            repository="owner/repo",
            actor="testuser",
            payload={"pr_number": 123},
            event_id="test-event-1",
        )

        assert event.event_type == GitHubEventType.PR_OPENED
        assert event.repository == "owner/repo"
        assert event.actor == "testuser"
        assert event.payload["pr_number"] == 123
        assert event.attention_level == AttentionLevel.MEDIUM
        assert event.processing_status == "pending"

    def test_github_priority_metrics_creation(self):
        """Test creating GitHub priority metrics."""
        metrics = GitHubPriorityMetrics(
            has_security_alert=True,
            blocks_other_prs=True,
            comment_count=5,
            file_changes_count=10,
            issue_labels=["bug", "critical"],
        )

        assert metrics.has_security_alert is True
        assert metrics.blocks_other_prs is True
        assert metrics.comment_count == 5
        assert metrics.file_changes_count == 10
        assert "bug" in metrics.issue_labels
        assert "critical" in metrics.issue_labels


class TestGitHubCliTool:
    """Test GitHub CLI tool functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = GitHubCliTool()

    @pytest.mark.asyncio
    async def test_github_cli_dry_run(self):
        """Test GitHub CLI tool in dry-run mode."""
        input_data = GitHubCliInput(
            command="gh issue create --title 'Test Issue' --body 'Test body'",
            dry_run=True,
            repository="owner/repo",
        )

        result = await self.tool.execute(input_data)

        assert result.success is True
        assert result.dry_run is True
        assert "Would create issue" in result.output
        assert result.github_event is not None
        assert result.github_event.event_type == GitHubEventType.ISSUE_CREATED

    @pytest.mark.asyncio
    async def test_github_cli_validation(self):
        """Test GitHub CLI command validation."""
        # Test invalid command (not starting with 'gh')
        input_data = GitHubCliInput(command="git status", dry_run=True)

        result = await self.tool.execute(input_data)

        assert result.success is False
        assert "Invalid or unsafe" in result.error

    @pytest.mark.asyncio
    async def test_github_cli_security_validation(self):
        """Test GitHub CLI security validation."""
        # Test command with shell injection patterns
        input_data = GitHubCliInput(
            command="gh issue list; rm -rf /", dry_run=True
        )

        result = await self.tool.execute(input_data)

        assert result.success is False
        assert "Invalid or unsafe" in result.error

    def test_github_cli_metadata(self):
        """Test GitHub CLI tool metadata."""
        metadata = self.tool.get_metadata()

        assert metadata["name"] == "github_cli"
        assert "github" in metadata["tags"]
        assert "input_schema" in metadata
        assert "output_schema" in metadata
        assert "safe_commands" in metadata


class TestGitHubApiClient:
    """Test GitHub API client functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.client = GitHubApiClient(token="test-token")

    @pytest.mark.asyncio
    async def test_github_api_request_structure(self):
        """Test GitHub API request structure."""
        with patch("httpx.AsyncClient") as mock_client:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": 123,
                "title": "Test Issue",
            }
            mock_response.headers = {"x-ratelimit-remaining": "4999"}

            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            from src.core.schemas import GitHubApiRequest

            request = GitHubApiRequest(
                endpoint="repos/owner/repo/issues",
                method="GET",
                parameters={"state": "open"},
            )

            result = await self.client.make_request(request)

            assert result.success is True
            assert result.status_code == 200
            assert result.data["id"] == 123
            assert result.rate_limit_remaining == 4999

    @pytest.mark.asyncio
    async def test_github_api_error_handling(self):
        """Test GitHub API error handling."""
        with patch("httpx.AsyncClient") as mock_client:
            # Mock error response
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.json.return_value = {"message": "Not Found"}
            mock_response.text = "Not Found"

            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            from src.core.schemas import GitHubApiRequest

            request = GitHubApiRequest(
                endpoint="repos/nonexistent/repo", method="GET"
            )

            result = await self.client.make_request(request)

            assert result.success is False
            assert result.status_code == 404
            assert "Not Found" in result.error

    @pytest.mark.asyncio
    async def test_priority_metrics_extraction(self):
        """Test GitHub priority metrics extraction."""
        with patch("httpx.AsyncClient") as mock_client:
            # Mock PR response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": 123,
                "number": 456,
                "comments": 5,
                "review_comments": 3,
                "additions": 100,
                "deletions": 20,
                "changed_files": 8,
                "mergeable": False,
                "labels": [{"name": "bug"}, {"name": "critical"}],
                "body": "This is urgent and needs @team review",
                "head": {"sha": "abc123"},
            }
            mock_response.headers = {"x-ratelimit-remaining": "4999"}

            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            metrics = await self.client.extract_priority_metrics(
                "owner/repo", "pull_request", 456
            )

            assert metrics.comment_count == 5
            assert metrics.review_count == 3
            assert metrics.lines_changed == 120
            assert metrics.file_changes_count == 8
            assert metrics.merge_conflicts is True
            assert "bug" in metrics.issue_labels
            assert "critical" in metrics.issue_labels
            assert metrics.has_security_alert is True  # "critical" label
            assert metrics.has_stakeholder_mention is True  # "@team" mention


class TestGitHubPriorityCalculator:
    """Test GitHub priority calculation system."""

    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = GitHubPriorityCalculator()

    def test_base_priority_calculation(self):
        """Test base priority calculation."""
        metrics = EnhancedPriorityMetrics(impact=8.0, urgency=6.0, effort=4.0)

        priority = self.calculator.calculate_priority(metrics)

        # Base priority should be (8 * 6) / 4 = 12.0
        assert priority == 12.0

    def test_github_priority_adjustments(self):
        """Test GitHub-specific priority adjustments."""
        github_metrics = GitHubPriorityMetrics(
            has_security_alert=True,
            blocks_other_prs=True,
            has_stakeholder_mention=True,
            issue_labels=["critical", "security"],
        )

        metrics = EnhancedPriorityMetrics(
            impact=5.0, urgency=5.0, effort=5.0, github_metrics=github_metrics
        )

        priority = self.calculator.calculate_priority(metrics)

        # Should be significantly higher than base priority (5*5/5 = 5.0)
        assert priority > 10.0  # GitHub adjustments should boost priority

    def test_temporal_priority_adjustments(self):
        """Test temporal priority adjustments."""
        metrics = EnhancedPriorityMetrics(
            impact=5.0,
            urgency=5.0,
            effort=5.0,
            age_hours=72.0,  # 3 days old
            deadline_hours=12.0,  # Deadline in 12 hours
        )

        priority = self.calculator.calculate_priority(metrics)

        # Should be higher than base priority due to age and approaching deadline
        assert priority > 5.0

    def test_priority_explanation(self):
        """Test priority calculation explanation."""
        github_metrics = GitHubPriorityMetrics(
            has_security_alert=True, blocks_other_prs=True
        )

        metrics = EnhancedPriorityMetrics(
            impact=6.0,
            urgency=7.0,
            effort=4.0,
            github_metrics=github_metrics,
            age_hours=48.0,
        )

        priority = self.calculator.calculate_priority(metrics)
        explanation = self.calculator.get_priority_explanation(
            metrics, priority
        )

        assert "final_priority" in explanation
        assert "contributing_factors" in explanation
        assert "priority_category" in explanation
        assert any(
            "Security alert" in factor
            for factor in explanation["contributing_factors"]
        )
        assert any(
            "Blocks other PRs" in factor
            for factor in explanation["contributing_factors"]
        )

    def test_priority_categorization(self):
        """Test priority score categorization."""
        test_cases = [
            (25.0, "CRITICAL"),
            (15.0, "HIGH"),
            (7.0, "MEDIUM"),
            (3.0, "LOW"),
            (1.0, "MINIMAL"),
        ]

        for priority_score, expected_category in test_cases:
            category = self.calculator._categorize_priority(priority_score)
            assert category == expected_category

    def test_github_event_to_priority_metrics(self):
        """Test converting GitHub event to priority metrics."""
        event = GitHubEventSchema(
            event_type=GitHubEventType.SECURITY_ALERT,
            repository="owner/repo",
            actor="testuser",
            payload={"severity": "high"},
            event_id="test-event",
            attention_level=AttentionLevel.CRITICAL,
        )

        metrics = self.calculator.create_priority_metrics_from_github_event(
            event
        )

        assert metrics.impact == 9.0  # Security alerts have high impact
        assert (
            metrics.urgency == 9.0
        )  # Critical attention level maps to high urgency
        assert metrics.age_hours >= 0.0


class TestGitHubIntegration:
    """Test integrated GitHub functionality."""

    @pytest.mark.asyncio
    async def test_end_to_end_github_workflow(self):
        """Test end-to-end GitHub integration workflow."""
        # Create GitHub event
        event = GitHubEventSchema(
            event_type=GitHubEventType.PR_OPENED,
            repository="owner/repo",
            actor="developer",
            payload={
                "pr_number": 123,
                "labels": [{"name": "enhancement"}],
                "comments": 2,
            },
            event_id="e2e-test-event",
        )

        # Calculate priority
        calculator = GitHubPriorityCalculator()
        metrics = calculator.create_priority_metrics_from_github_event(event)
        priority = calculator.calculate_priority(metrics)

        # Verify workflow
        assert event.event_type == GitHubEventType.PR_OPENED
        assert priority > 0.0
        assert metrics.impact > 0.0
        assert metrics.urgency > 0.0

    def test_github_schemas_integration(self):
        """Test integration between different GitHub schemas."""
        # Create related schemas
        github_metrics = GitHubPriorityMetrics(
            has_security_alert=True,
            comment_count=5,
            issue_labels=["bug", "urgent"],
        )

        priority_metrics = EnhancedPriorityMetrics(
            impact=7.0, urgency=8.0, effort=5.0, github_metrics=github_metrics
        )

        event = GitHubEventSchema(
            event_type=GitHubEventType.ISSUE_CREATED,
            repository="owner/repo",
            actor="reporter",
            payload={"issue_number": 456},
            event_id="schema-integration-test",
        )

        # Verify integration
        assert priority_metrics.github_metrics.has_security_alert is True
        assert event.event_type == GitHubEventType.ISSUE_CREATED
        assert len(github_metrics.issue_labels) == 2


# Integration test fixtures and utilities
@pytest.fixture
def sample_github_event():
    """Sample GitHub event for testing."""
    return GitHubEventSchema(
        event_type=GitHubEventType.PR_OPENED,
        repository="test/repo",
        actor="testuser",
        payload={"pr_number": 123},
        event_id="test-fixture-event",
    )


@pytest.fixture
def sample_priority_metrics():
    """Sample priority metrics for testing."""
    return EnhancedPriorityMetrics(
        impact=6.0,
        urgency=7.0,
        effort=4.0,
        github_metrics=GitHubPriorityMetrics(
            has_security_alert=True,
            comment_count=3,
            issue_labels=["enhancement"],
        ),
    )


@pytest.mark.integration
class TestGitHubIntegrationE2E:
    """End-to-end integration tests for GitHub functionality."""

    @pytest.mark.asyncio
    async def test_github_cli_to_event_flow(self, sample_github_event):
        """Test flow from GitHub CLI to event generation."""
        tool = GitHubCliTool()

        input_data = GitHubCliInput(
            command="gh pr create --title 'Test PR' --body 'Test description'",
            dry_run=True,
            repository="test/repo",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.github_event is not None
        assert result.github_event.event_type == GitHubEventType.PR_OPENED

    def test_priority_calculation_with_github_data(
        self, sample_priority_metrics
    ):
        """Test priority calculation with real GitHub data structure."""
        calculator = GitHubPriorityCalculator()

        priority = calculator.calculate_priority(sample_priority_metrics)
        explanation = calculator.get_priority_explanation(
            sample_priority_metrics, priority
        )

        assert priority > 0.0
        assert explanation["priority_category"] in [
            "MINIMAL",
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        ]
        assert len(explanation["contributing_factors"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
