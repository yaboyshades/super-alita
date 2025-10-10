"""Enhanced priority calculation system with GitHub-specific metrics.

Extends the existing priority system to incorporate GitHub-specific metrics
like security alerts, blocking relationships, and stakeholder involvement.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from src.core.schemas import (
    AttentionLevel,
    GitHubEventSchema,
    GitHubPriorityMetrics,
    TaskRequest,
)


class EnhancedPriorityMetrics(BaseModel):
    """Enhanced priority metrics combining base and GitHub-specific factors."""

    # Base priority factors
    impact: float = Field(
        default=5.0, ge=1.0, le=10.0, description="Task impact score"
    )
    urgency: float = Field(
        default=5.0, ge=1.0, le=10.0, description="Task urgency score"
    )
    effort: float = Field(
        default=5.0, ge=1.0, le=10.0, description="Required effort estimate"
    )

    # GitHub-specific factors
    github_metrics: GitHubPriorityMetrics | None = Field(
        default=None, description="GitHub-specific metrics"
    )

    # Temporal factors
    age_hours: float = Field(default=0.0, ge=0.0, description="Age in hours")
    deadline_hours: float | None = Field(
        default=None, description="Hours until deadline"
    )

    # Social factors
    stakeholder_count: int = Field(
        default=0, ge=0, description="Number of stakeholders"
    )
    community_interest: float = Field(
        default=1.0, ge=0.0, description="Community interest level"
    )

    # Technical factors
    complexity_score: float = Field(
        default=5.0, ge=1.0, le=10.0, description="Technical complexity"
    )
    risk_score: float = Field(
        default=5.0, ge=1.0, le=10.0, description="Risk assessment"
    )


class GitHubPriorityCalculator:
    """Calculator for enhanced priority with GitHub integration."""

    def __init__(self):
        # Base priority weights
        self.base_weights = {
            "impact": 0.4,
            "urgency": 0.3,
            "effort": -0.2,  # Negative because higher effort reduces priority
            "complexity": -0.1,
        }

        # GitHub-specific adjustment weights
        self.github_weights = {
            "security_alert": 1.5,
            "blocks_others": 1.3,
            "stakeholder_mention": 1.2,
            "critical_label": 1.4,
            "ci_failure": 1.1,
            "merge_conflicts": 1.15,
            "high_activity": 1.1,
        }

        # Temporal decay factors
        self.temporal_weights = {
            "age_penalty": 0.02,  # Priority increases with age
            "deadline_urgency": 2.0,  # Multiplier near deadline
        }

    def calculate_priority(
        self,
        metrics: EnhancedPriorityMetrics,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Calculate enhanced priority score with GitHub metrics."""

        # Base priority calculation
        base_priority = self._calculate_base_priority(metrics)

        # GitHub-specific adjustments
        github_adjustments = self._calculate_github_adjustments(metrics)

        # Temporal adjustments
        temporal_adjustments = self._calculate_temporal_adjustments(metrics)

        # Social/community adjustments
        social_adjustments = self._calculate_social_adjustments(metrics)

        # Technical risk adjustments
        risk_adjustments = self._calculate_risk_adjustments(metrics)

        # Combine all factors
        final_priority = (
            base_priority
            * github_adjustments
            * temporal_adjustments
            * social_adjustments
            * risk_adjustments
        )

        # Clamp to reasonable range
        return max(0.1, min(100.0, final_priority))

    def _calculate_base_priority(
        self, metrics: EnhancedPriorityMetrics
    ) -> float:
        """Calculate base priority using impact, urgency, effort formula."""
        if metrics.effort <= 0:
            metrics.effort = 1.0  # Avoid division by zero

        return (metrics.impact * metrics.urgency) / metrics.effort

    def _calculate_github_adjustments(
        self, metrics: EnhancedPriorityMetrics
    ) -> float:
        """Calculate GitHub-specific priority adjustments."""

        if not metrics.github_metrics:
            return 1.0

        github_metrics = metrics.github_metrics
        adjustment = 1.0

        # Security alerts have highest priority
        if github_metrics.has_security_alert:
            adjustment *= self.github_weights["security_alert"]

        # Blocking other PRs increases priority
        if github_metrics.blocks_other_prs:
            adjustment *= self.github_weights["blocks_others"]

        # Stakeholder mentions increase priority
        if github_metrics.has_stakeholder_mention:
            adjustment *= self.github_weights["stakeholder_mention"]

        # Critical/urgent labels
        critical_labels = ["critical", "urgent", "hotfix", "security"]
        if any(
            label.lower() in critical_labels
            for label in github_metrics.issue_labels
        ):
            adjustment *= self.github_weights["critical_label"]

        # CI failures need attention
        if github_metrics.ci_status in ["failure", "error"]:
            adjustment *= self.github_weights["ci_failure"]

        # Merge conflicts need resolution
        if github_metrics.merge_conflicts:
            adjustment *= self.github_weights["merge_conflicts"]

        # High activity (comments/reviews) indicates importance
        total_activity = (
            github_metrics.comment_count + github_metrics.review_count
        )
        if total_activity > 10:
            adjustment *= self.github_weights["high_activity"]

        return adjustment

    def _calculate_temporal_adjustments(
        self, metrics: EnhancedPriorityMetrics
    ) -> float:
        """Calculate time-based priority adjustments."""

        adjustment = 1.0

        # Age increases priority (older items get more urgent)
        if metrics.age_hours > 0:
            age_factor = 1 + (
                metrics.age_hours * self.temporal_weights["age_penalty"] / 24
            )
            adjustment *= age_factor

        # Deadline proximity dramatically increases priority
        if metrics.deadline_hours is not None:
            if metrics.deadline_hours <= 0:
                adjustment *= 3.0  # Past deadline
            elif metrics.deadline_hours <= 24:
                adjustment *= self.temporal_weights[
                    "deadline_urgency"
                ]  # Within 24 hours
            elif metrics.deadline_hours <= 72:
                adjustment *= 1.5  # Within 3 days

        return adjustment

    def _calculate_social_adjustments(
        self, metrics: EnhancedPriorityMetrics
    ) -> float:
        """Calculate social/community factor adjustments."""

        adjustment = 1.0

        # More stakeholders = higher priority
        if metrics.stakeholder_count > 0:
            stakeholder_factor = 1 + (metrics.stakeholder_count * 0.1)
            adjustment *= stakeholder_factor

        # Community interest multiplier
        if metrics.community_interest > 1.0:
            adjustment *= metrics.community_interest

        return adjustment

    def _calculate_risk_adjustments(
        self, metrics: EnhancedPriorityMetrics
    ) -> float:
        """Calculate technical risk adjustments."""

        adjustment = 1.0

        # High risk items need more attention
        if metrics.risk_score > 7.0:
            adjustment *= 1.3
        elif metrics.risk_score > 5.0:
            adjustment *= 1.1

        # High complexity items might need more careful handling
        if metrics.complexity_score > 8.0:
            adjustment *= 0.9  # Slightly lower priority for very complex items

        return adjustment

    def create_priority_metrics_from_github_event(
        self,
        event: GitHubEventSchema,
        base_impact: float = 5.0,
        base_urgency: float = 5.0,
        base_effort: float = 5.0,
    ) -> EnhancedPriorityMetrics:
        """Create priority metrics from GitHub event."""

        # Calculate age from event timestamp
        age_hours = (
            datetime.now(UTC) - event.timestamp
        ).total_seconds() / 3600

        # Map attention level to urgency
        urgency_mapping = {
            AttentionLevel.LOW: 3.0,
            AttentionLevel.MEDIUM: 5.0,
            AttentionLevel.HIGH: 7.0,
            AttentionLevel.CRITICAL: 9.0,
        }

        calculated_urgency = urgency_mapping.get(
            event.attention_level, base_urgency
        )

        # Estimate impact based on event type
        impact_mapping = {
            "security_alert": 9.0,
            "pr_merged": 6.0,
            "issue_created": 5.0,
            "commit_pushed": 4.0,
            "review_submitted": 4.0,
        }

        calculated_impact = impact_mapping.get(
            event.event_type.value, base_impact
        )

        # Create basic GitHub metrics from event payload
        github_metrics = GitHubPriorityMetrics()

        payload = event.payload
        if isinstance(payload, dict):
            # Extract metrics from payload
            if "labels" in payload:
                labels = [label.get("name", "") for label in payload["labels"]]
                github_metrics.issue_labels = labels

                security_labels = ["security", "critical", "urgent", "bug"]
                github_metrics.has_security_alert = any(
                    label.lower() in security_labels for label in labels
                )

            if "comments" in payload:
                github_metrics.comment_count = payload["comments"]

            if "mergeable" in payload:
                github_metrics.merge_conflicts = payload["mergeable"] is False

        return EnhancedPriorityMetrics(
            impact=calculated_impact,
            urgency=calculated_urgency,
            effort=base_effort,
            github_metrics=github_metrics,
            age_hours=age_hours,
            complexity_score=5.0,
            risk_score=5.0,
        )

    def create_priority_metrics_from_task(
        self,
        task: TaskRequest,
        github_metrics: GitHubPriorityMetrics | None = None,
    ) -> EnhancedPriorityMetrics:
        """Create priority metrics from task request."""

        # Map task priority to impact/urgency
        task.priority / 10.0  # Normalize to 0-1

        # Calculate age if task has timestamp in metadata
        age_hours = 0.0
        if "created_at" in task.metadata:
            try:
                created_at = datetime.fromisoformat(
                    task.metadata["created_at"]
                )
                age_hours = (
                    datetime.now(UTC) - created_at
                ).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

        # Estimate effort based on task type and description length
        base_effort = 5.0
        if task.description:
            # Longer descriptions might indicate more complex tasks
            desc_length = len(task.description)
            if desc_length > 500:
                base_effort = 7.0
            elif desc_length > 200:
                base_effort = 6.0

        return EnhancedPriorityMetrics(
            impact=task.priority,
            urgency=task.priority,
            effort=base_effort,
            github_metrics=github_metrics,
            age_hours=age_hours,
            complexity_score=base_effort,  # Use effort as complexity proxy
            risk_score=5.0,
        )

    def get_priority_explanation(
        self, metrics: EnhancedPriorityMetrics, final_priority: float
    ) -> dict[str, Any]:
        """Get explanation of priority calculation."""

        base_priority = self._calculate_base_priority(metrics)
        github_adj = self._calculate_github_adjustments(metrics)
        temporal_adj = self._calculate_temporal_adjustments(metrics)
        social_adj = self._calculate_social_adjustments(metrics)
        risk_adj = self._calculate_risk_adjustments(metrics)

        factors = []

        # GitHub factors
        if metrics.github_metrics:
            if metrics.github_metrics.has_security_alert:
                factors.append("Security alert detected (+50%)")
            if metrics.github_metrics.blocks_other_prs:
                factors.append("Blocks other PRs (+30%)")
            if metrics.github_metrics.has_stakeholder_mention:
                factors.append("Stakeholder mentioned (+20%)")
            if metrics.github_metrics.merge_conflicts:
                factors.append("Has merge conflicts (+15%)")

        # Temporal factors
        if metrics.age_hours > 48:
            factors.append(
                f"Old item ({metrics.age_hours:.1f}h) - increasing urgency"
            )
        if metrics.deadline_hours and metrics.deadline_hours <= 24:
            factors.append("Approaching deadline - high urgency")

        # Social factors
        if metrics.stakeholder_count > 3:
            factors.append(
                f"High stakeholder involvement ({metrics.stakeholder_count})"
            )

        return {
            "final_priority": final_priority,
            "base_priority": base_priority,
            "github_adjustment": github_adj,
            "temporal_adjustment": temporal_adj,
            "social_adjustment": social_adj,
            "risk_adjustment": risk_adj,
            "contributing_factors": factors,
            "priority_category": self._categorize_priority(final_priority),
        }

    def _categorize_priority(self, priority: float) -> str:
        """Categorize priority score into human-readable categories."""

        if priority >= 20.0:
            return "CRITICAL"
        elif priority >= 10.0:
            return "HIGH"
        elif priority >= 5.0:
            return "MEDIUM"
        elif priority >= 2.0:
            return "LOW"
        else:
            return "MINIMAL"
