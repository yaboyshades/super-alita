"""Collective intelligence integration for multi-user learning."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)


class SharedKnowledgeGraphLike:
    async def integrate_learning(
        self,
        *,
        learning: dict[str, Any],
        source_hash: int,
        validation_score: float,
    ) -> None:  # pragma: no cover - protocol
        ...


class PrivacyControllerLike:
    async def anonymize(
        self, *, learning: dict[str, Any], privacy_level: str
    ) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


class ContributionValidatorLike:
    async def validate(
        self, *, contribution: dict[str, Any], contributor: str
    ) -> Any:  # pragma: no cover - protocol
        ...


class CollectiveIntelligenceNetwork:
    """Accept and broadcast learning updates from multiple users."""

    def __init__(
        self,
        *,
        shared_knowledge_graph: SharedKnowledgeGraphLike | None = None,
        privacy_controller: PrivacyControllerLike | None = None,
        contribution_validator: ContributionValidatorLike | None = None,
    ) -> None:
        self.shared_knowledge_graph = shared_knowledge_graph
        self.privacy_controller = privacy_controller
        self.contribution_validator = contribution_validator

    async def contribute_learning(
        self,
        *,
        user_id: str,
        learning: dict[str, Any],
        privacy_level: str = "public",
    ) -> dict[str, Any]:
        """Integrate user learning into the collective knowledge graph."""

        if not user_id:
            raise ValueError("user_id must be provided")
        if not isinstance(learning, dict):
            raise ValueError("learning must be a dictionary")
        validation = await self._validate_contribution(user_id, learning)
        if not validation.get("approved", False):
            return {"accepted": False, "reason": validation.get("reason", "rejected")}
        anonymized_learning = await self._apply_privacy(learning, privacy_level)
        await self._store_learning(
            anonymized_learning,
            user_id=user_id,
            validation_score=float(validation.get("confidence", 1.0)),
        )
        await self._broadcast_collective_learning(anonymized_learning)
        return {"accepted": True, "anonymized_learning": anonymized_learning}

    async def _validate_contribution(
        self, user_id: str, learning: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.contribution_validator:
            return {"approved": True, "confidence": 1.0}
        try:
            result = await self.contribution_validator.validate(
                contribution=learning, contributor=user_id
            )
        except Exception:
            logger.exception("Contribution validator failed; rejecting input")
            return {"approved": False, "reason": "validator_error"}
        if not isinstance(result, dict):
            return {"approved": False, "reason": "invalid_validator_response"}
        return result

    async def _apply_privacy(
        self, learning: dict[str, Any], privacy_level: str
    ) -> dict[str, Any]:
        if not self.privacy_controller:
            return dict(learning)
        try:
            anonymized = await self.privacy_controller.anonymize(
                learning=learning, privacy_level=privacy_level
            )
        except Exception:
            logger.exception("Privacy controller failed; defaulting to raw learning")
            return dict(learning)
        if not isinstance(anonymized, dict):
            return dict(learning)
        return anonymized

    async def _store_learning(
        self,
        anonymized_learning: dict[str, Any],
        *,
        user_id: str,
        validation_score: float,
    ) -> None:
        if not self.shared_knowledge_graph:
            return
        try:
            await self.shared_knowledge_graph.integrate_learning(
                learning=anonymized_learning,
                source_hash=hash(user_id),
                validation_score=validation_score,
            )
        except Exception:
            logger.exception("Failed to integrate learning into shared graph")

    async def _broadcast_collective_learning(self, learning: dict[str, Any]) -> None:
        relevant_users = await self._find_relevant_users(learning)
        for user in relevant_users:
            await self._send_learning_update(user_id=user, learning=learning)

    async def _find_relevant_users(self, learning: dict[str, Any]) -> list[str]:
        audience = learning.get("audience")
        return [str(uid) for uid in audience] if isinstance(audience, list) else []

    async def _send_learning_update(
        self, *, user_id: str, learning: dict[str, Any]
    ) -> None:
        logger.debug("Dispatching learning update to %s", user_id)
