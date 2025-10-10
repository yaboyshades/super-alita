"""Meta-Constitutional Framework for evolvable governance.

Implements self-modifying constitutional system with:
- Amendment proposal and voting
- Consensus gathering for changes
- Backward compatibility tracking
- Constitutional evolution history
- Article lifecycle management
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any


class AmendmentStatus(str, Enum):
    """Amendment lifecycle states."""

    PROPOSED = "proposed"
    UNDER_REVIEW = "under_review"
    VOTING = "voting"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    DEPRECATED = "deprecated"


class ArticleStatus(str, Enum):
    """Constitutional article states."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


@dataclass
class Vote:
    """Vote on an amendment."""

    voter_id: str
    approve: bool
    weight: float = 1.0
    rationale: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Amendment:
    """Proposed change to the constitution."""

    id: str
    title: str
    description: str
    proposed_by: str
    proposed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Content
    article_id: str | None = None  # If modifying existing article
    new_article_content: dict[str, Any] | None = None
    action: str = "add"  # add, modify, deprecate

    # Status
    status: AmendmentStatus = AmendmentStatus.PROPOSED

    # Voting
    votes: list[Vote] = field(default_factory=list)
    approval_threshold: float = 0.75  # 75% approval needed
    voting_deadline: datetime | None = None

    # Impact
    backward_compatible: bool = True
    breaking_changes: list[str] = field(default_factory=list)
    migration_guide: str = ""


@dataclass
class ConstitutionalArticle:
    """A single article in the constitution."""

    id: str
    name: str
    content: dict[str, Any]
    version: int = 1
    status: ArticleStatus = ArticleStatus.ACTIVE

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Validation
    validator_fn: Callable[[Any], float] | None = None
    threshold: float = 0.75

    # History
    amendment_history: list[str] = field(default_factory=list)
    superseded_by: str | None = None


@dataclass
class ConstitutionalVersion:
    """A version of the constitution."""

    version: int
    articles: dict[str, ConstitutionalArticle]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    hash: str = ""

    def __post_init__(self):
        """Generate hash of this version."""
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of all articles."""
        content = json.dumps(
            {aid: art.content for aid, art in self.articles.items()},
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class MetaConstitution:
    """Self-modifying constitutional framework.

    Manages constitutional amendments with:
    - Proposal and voting system
    - Consensus-based approval
    - Backward compatibility tracking
    - Full evolution history
    """

    def __init__(
        self,
        initial_articles: dict[str, ConstitutionalArticle] | None = None,
        default_approval_threshold: float = 0.75,
        default_voting_period_hours: int = 168,  # 1 week
    ):
        """Initialize meta-constitutional framework.

        Args:
            initial_articles: Starting articles
            default_approval_threshold: Default approval percentage
            default_voting_period_hours: Default voting period
        """
        self.default_approval_threshold = default_approval_threshold
        self.default_voting_period_hours = default_voting_period_hours

        # Current constitution
        self.current_version = 1
        self.articles = initial_articles or {}

        # History
        self.versions: list[ConstitutionalVersion] = []
        self._save_version()

        # Amendments
        self.amendments: dict[str, Amendment] = {}

        # Statistics
        self.total_amendments = 0
        self.approved_amendments = 0
        self.rejected_amendments = 0

    def _save_version(self) -> None:
        """Save current state as a version."""
        version = ConstitutionalVersion(
            version=self.current_version,
            articles=dict(self.articles),
        )
        self.versions.append(version)

    def propose_amendment(
        self,
        title: str,
        description: str,
        proposed_by: str,
        article_id: str | None = None,
        new_article_content: dict[str, Any] | None = None,
        action: str = "add",
        backward_compatible: bool = True,
        breaking_changes: list[str] | None = None,
    ) -> Amendment:
        """Propose a constitutional amendment.

        Args:
            title: Amendment title
            description: Detailed description
            proposed_by: Proposer identifier
            article_id: Article being modified (if applicable)
            new_article_content: New/modified article content
            action: 'add', 'modify', or 'deprecate'
            backward_compatible: Whether change is backward compatible
            breaking_changes: List of breaking changes

        Returns:
            Created Amendment
        """
        amendment_id = hashlib.sha256(
            f"{title}_{datetime.now(UTC).isoformat()}".encode()
        ).hexdigest()[:16]

        voting_deadline = datetime.now(UTC) + timedelta(
            hours=self.default_voting_period_hours
        )

        amendment = Amendment(
            id=amendment_id,
            title=title,
            description=description,
            proposed_by=proposed_by,
            article_id=article_id,
            new_article_content=new_article_content,
            action=action,
            approval_threshold=self.default_approval_threshold,
            voting_deadline=voting_deadline,
            backward_compatible=backward_compatible,
            breaking_changes=breaking_changes or [],
        )

        self.amendments[amendment_id] = amendment
        self.total_amendments += 1

        return amendment

    def vote_on_amendment(
        self,
        amendment_id: str,
        voter_id: str,
        approve: bool,
        weight: float = 1.0,
        rationale: str = "",
    ) -> Amendment:
        """Cast a vote on an amendment.

        Args:
            amendment_id: Amendment identifier
            voter_id: Voter identifier
            approve: Whether to approve (True) or reject (False)
            weight: Vote weight (for weighted voting)
            rationale: Reason for vote

        Returns:
            Updated Amendment
        """
        if amendment_id not in self.amendments:
            raise ValueError(f"Amendment {amendment_id} not found")

        amendment = self.amendments[amendment_id]

        if amendment.status not in {
            AmendmentStatus.PROPOSED,
            AmendmentStatus.VOTING,
        }:
            raise ValueError(f"Amendment {amendment_id} not open for voting")

        # Check if voter already voted
        existing_vote = next(
            (v for v in amendment.votes if v.voter_id == voter_id), None
        )
        if existing_vote:
            # Update existing vote
            existing_vote.approve = approve
            existing_vote.rationale = rationale
            existing_vote.timestamp = datetime.now(UTC)
        else:
            # Add new vote
            vote = Vote(
                voter_id=voter_id,
                approve=approve,
                weight=weight,
                rationale=rationale,
            )
            amendment.votes.append(vote)

        amendment.status = AmendmentStatus.VOTING

        return amendment

    def tally_votes(self, amendment_id: str) -> dict[str, Any]:
        """Tally votes for an amendment.

        Args:
            amendment_id: Amendment identifier

        Returns:
            Vote tally results
        """
        if amendment_id not in self.amendments:
            raise ValueError(f"Amendment {amendment_id} not found")

        amendment = self.amendments[amendment_id]

        if not amendment.votes:
            return {
                "total_votes": 0,
                "approval_percentage": 0.0,
                "approved": False,
            }

        # Weighted tally
        total_weight = sum(v.weight for v in amendment.votes)
        approve_weight = sum(v.weight for v in amendment.votes if v.approve)

        approval_percentage = (
            (approve_weight / total_weight) * 100.0
            if total_weight > 0
            else 0.0
        )

        approved = approval_percentage >= (
            amendment.approval_threshold * 100.0
        )

        return {
            "total_votes": len(amendment.votes),
            "total_weight": total_weight,
            "approve_weight": approve_weight,
            "approval_percentage": approval_percentage,
            "threshold": amendment.approval_threshold * 100.0,
            "approved": approved,
        }

    def finalize_amendment(self, amendment_id: str) -> Amendment:
        """Finalize voting and approve or reject amendment.

        Args:
            amendment_id: Amendment identifier

        Returns:
            Updated Amendment
        """
        if amendment_id not in self.amendments:
            raise ValueError(f"Amendment {amendment_id} not found")

        amendment = self.amendments[amendment_id]

        if amendment.status not in {
            AmendmentStatus.VOTING,
            AmendmentStatus.UNDER_REVIEW,
        }:
            raise ValueError(f"Amendment {amendment_id} not ready to finalize")

        tally = self.tally_votes(amendment_id)

        if tally["approved"]:
            amendment.status = AmendmentStatus.APPROVED
            self.approved_amendments += 1
        else:
            amendment.status = AmendmentStatus.REJECTED
            self.rejected_amendments += 1

        return amendment

    def implement_amendment(
        self, amendment_id: str
    ) -> ConstitutionalArticle | None:
        """Implement an approved amendment.

        Args:
            amendment_id: Amendment identifier

        Returns:
            Updated or created ConstitutionalArticle, or None if deprecation
        """
        if amendment_id not in self.amendments:
            raise ValueError(f"Amendment {amendment_id} not found")

        amendment = self.amendments[amendment_id]

        if amendment.status != AmendmentStatus.APPROVED:
            raise ValueError(f"Amendment {amendment_id} not approved")

        # Perform action
        if amendment.action == "add":
            # Add new article
            if not amendment.new_article_content:
                raise ValueError("No article content provided")

            article_id = (
                amendment.article_id or f"Article_{len(self.articles) + 1}"
            )
            article = ConstitutionalArticle(
                id=article_id,
                name=amendment.title,
                content=amendment.new_article_content,
                version=1,
            )
            article.amendment_history.append(amendment_id)

            self.articles[article_id] = article

            amendment.status = AmendmentStatus.IMPLEMENTED
            self._increment_version()
            return article

        elif amendment.action == "modify":
            # Modify existing article
            if not amendment.article_id:
                raise ValueError("No article_id specified for modification")

            if amendment.article_id not in self.articles:
                raise ValueError(f"Article {amendment.article_id} not found")

            article = self.articles[amendment.article_id]
            article.content = amendment.new_article_content or article.content
            article.version += 1
            article.updated_at = datetime.now(UTC)
            article.amendment_history.append(amendment_id)

            amendment.status = AmendmentStatus.IMPLEMENTED
            self._increment_version()
            return article

        elif amendment.action == "deprecate":
            # Deprecate article
            if not amendment.article_id:
                raise ValueError("No article_id specified for deprecation")

            if amendment.article_id not in self.articles:
                raise ValueError(f"Article {amendment.article_id} not found")

            article = self.articles[amendment.article_id]
            article.status = ArticleStatus.DEPRECATED
            article.updated_at = datetime.now(UTC)
            article.amendment_history.append(amendment_id)

            amendment.status = AmendmentStatus.IMPLEMENTED
            self._increment_version()
            return None

        return None

    def _increment_version(self) -> None:
        """Increment constitution version and save."""
        self.current_version += 1
        self._save_version()

    def get_article(self, article_id: str) -> ConstitutionalArticle | None:
        """Get an article by ID.

        Args:
            article_id: Article identifier

        Returns:
            ConstitutionalArticle or None if not found
        """
        return self.articles.get(article_id)

    def get_active_articles(self) -> dict[str, ConstitutionalArticle]:
        """Get all active articles.

        Returns:
            Dictionary of active articles
        """
        return {
            aid: art
            for aid, art in self.articles.items()
            if art.status == ArticleStatus.ACTIVE
        }

    def get_version_history(self) -> list[dict[str, Any]]:
        """Get constitutional version history.

        Returns:
            List of version summaries
        """
        return [
            {
                "version": ver.version,
                "hash": ver.hash,
                "articles": len(ver.articles),
                "created_at": ver.created_at.isoformat(),
            }
            for ver in self.versions
        ]

    def get_amendment_history(self, article_id: str) -> list[Amendment] | None:
        """Get amendment history for an article.

        Args:
            article_id: Article identifier

        Returns:
            List of amendments affecting this article
        """
        article = self.get_article(article_id)
        if not article:
            return None

        return [
            self.amendments[aid]
            for aid in article.amendment_history
            if aid in self.amendments
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get meta-constitutional statistics."""
        active_amendments = sum(
            1
            for a in self.amendments.values()
            if a.status in {AmendmentStatus.PROPOSED, AmendmentStatus.VOTING}
        )

        return {
            "current_version": self.current_version,
            "total_articles": len(self.articles),
            "active_articles": len(self.get_active_articles()),
            "total_amendments": self.total_amendments,
            "approved_amendments": self.approved_amendments,
            "rejected_amendments": self.rejected_amendments,
            "active_amendments": active_amendments,
            "versions": len(self.versions),
        }

    def export_constitution(self) -> dict[str, Any]:
        """Export current constitution.

        Returns:
            Constitution as dictionary
        """
        return {
            "version": self.current_version,
            "hash": self.versions[-1].hash if self.versions else "",
            "articles": {
                aid: {
                    "id": art.id,
                    "name": art.name,
                    "content": art.content,
                    "version": art.version,
                    "status": art.status,
                    "created_at": art.created_at.isoformat(),
                    "updated_at": art.updated_at.isoformat(),
                }
                for aid, art in self.articles.items()
            },
        }


# Example usage
def example_meta_constitution() -> None:
    """Example demonstrating MetaConstitution."""
    # Initialize with base articles
    base_articles = {
        "Article_I": ConstitutionalArticle(
            id="Article_I",
            name="Test-First Development",
            content={"rule": "All features must have tests"},
            threshold=0.8,
        ),
        "Article_II": ConstitutionalArticle(
            id="Article_II",
            name="Code Simplicity",
            content={"rule": "Keep functions small and focused"},
            threshold=0.75,
        ),
    }

    meta = MetaConstitution(initial_articles=base_articles)

    print(f"Initial version: {meta.current_version}")
    print(f"Articles: {len(meta.articles)}")

    # Propose amendment
    amendment = meta.propose_amendment(
        title="Add Documentation Standard",
        description="All public APIs must have docstrings",
        proposed_by="developer_1",
        new_article_content={"rule": "Document all public APIs"},
        action="add",
    )

    print(f"\nProposed amendment: {amendment.id}")

    # Vote on amendment
    meta.vote_on_amendment(
        amendment.id,
        "developer_1",
        True,
        rationale="Critical for maintainability",
    )
    meta.vote_on_amendment(
        amendment.id, "developer_2", True, rationale="Agree, very important"
    )
    meta.vote_on_amendment(
        amendment.id, "developer_3", True, rationale="Essential for onboarding"
    )

    # Tally votes
    tally = meta.tally_votes(amendment.id)
    print(f"\nVote tally: {tally['approval_percentage']:.1f}% approval")

    # Finalize
    meta.finalize_amendment(amendment.id)
    print(f"Amendment status: {amendment.status}")

    # Implement
    new_article = meta.implement_amendment(amendment.id)
    print(f"New article: {new_article.id if new_article else 'None'}")  # type: ignore[union-attr]
    print(f"New version: {meta.current_version}")

    # Stats
    print("\nMeta-Constitutional Stats:")
    print(json.dumps(meta.get_stats(), indent=2))

    # Export
    print("\nCurrent Constitution:")
    print(json.dumps(meta.export_constitution(), indent=2))


if __name__ == "__main__":
    example_meta_constitution()
