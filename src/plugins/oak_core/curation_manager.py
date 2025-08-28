from __future__ import annotations
import logging
import re
from collections import defaultdict
from typing import Any

from src.core.plugin_interface import PluginInterface

"""Stable minimal implementation of the Oak CurationManager.

The previous version of this module had duplicated methods, inconsistent
indentation and unreachable code sections that caused an IndentationError
on import. This simplified version preserves the public surface expected
by the plugin system while removing experimental / broken logic.

Responsibilities (minimum viable):
  * Maintain simple counters for semantic vs syntactic errors coming from
    tool results.
  * Emit a lightweight feedback event so downstream components can adapt.
  * Provide hooks for prediction error signals.

All advanced utility aggregation and feature weighting can be reintroduced
later in a dedicated refactor; here we prioritise reliability so that
plugin discovery (and tests that import this module) succeed.
"""

logger = logging.getLogger(__name__)


class CurationManager(PluginInterface):
    """Curate basic feedback signals for Oak core.

    Emits:
      - oak.curation_feedback
    Subscribes:
      - tool_result
      - prediction_error / oak.prediction_error
    """

    def __init__(self) -> None:  # noqa: D401 - simple init
        super().__init__()
        self.error_counts: dict[str, int] = defaultdict(int)
        # Simple tuning constants (could become configurable)
        self.play_weight: float = 0.1
        self.semantic_error_penalty: float = -0.2
        self.syntactic_error_penalty: float = -0.1

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "oak_curation_manager"

    async def setup(
        self, event_bus: Any, store: Any, config: dict[str, Any]
    ) -> None:  # noqa: D401
        await super().setup(event_bus, store, config)
        self.cfg: dict[str, float] = {
            "play_weight": self.get_config("play_weight", 0.1),
            "planning_weight": self.get_config("planning_weight", 0.2),
            "semantic_error_penalty": self.get_config("semantic_error_penalty", -0.2),
            "syntactic_error_penalty": self.get_config("syntactic_error_penalty", -0.1),
        }
        self.error_counts: dict[str, int] = defaultdict(int)
        self._required_features = {"global_play", "global_planning"}

    async def start(self) -> None:
        # Allow light config overrides
        self.play_weight = float(config.get("play_weight", self.play_weight))
        self.semantic_error_penalty = float(
            config.get("semantic_error_penalty", self.semantic_error_penalty)
        )
        self.syntactic_error_penalty = float(
            config.get("syntactic_error_penalty", self.syntactic_error_penalty)
        )

    async def start(self) -> None:  # noqa: D401
        await super().start()
        await self.subscribe("tool_result", self.handle_tool_result)
        await self.subscribe("prediction_error", self.handle_prediction_error)
        await self.subscribe("oak.prediction_error", self.handle_prediction_error)

    def _feature_exists(self, feature_id: str) -> bool:
        if not self.store:
            return False
        try:
            if hasattr(self.store, "has_feature"):
                return bool(self.store.has_feature(feature_id))  # type: ignore[attr-defined]
            if hasattr(self.store, "get_feature"):
                return self.store.get_feature(feature_id) is not None  # type: ignore[attr-defined]
            if hasattr(self.store, "features"):
                return feature_id in self.store.features  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            return False
        return False

    async def _emit_utility_update(
        self,
        feature_id: str,
        signal_type: str,
        value: float,
        components: dict[str, float],
    ) -> None:
        if not self._feature_exists(feature_id):
            logger.warning(
                "CurationManager skipping utility update for missing feature '%s'",
                feature_id,
            )
            return
        await self.emit_event(
            "oak.feature_utility_update",
            feature_id=feature_id,
            signal_type=signal_type,
            value=value,
            components=components,
        )

    async def start(self) -> None:  # type: ignore[override]
        await super().start()

    async def shutdown(self) -> None:  # type: ignore[override]
    async def shutdown(self) -> None:  # noqa: D401
        await super().shutdown()

    async def handle_tool_result(self, event: Any) -> None:
        """Process a tool_result event and emit curation feedback."""
        try:
            success = bool(event.get("success", False))
            if success:
                await self.emit_event(
                    "oak.curation_feedback",
                    category="play",
                    success=True,
                    weight=self.play_weight,
                )
                return

            error_msg = str(event.get("error", ""))
            if re.search(r"(schema|validation|type|required)", error_msg, re.I):
                category = "syntactic"
                signal = self.cfg["syntactic_error_penalty"]
            else:
                category = "semantic"
                signal = self.cfg["semantic_error_penalty"]

            self.error_counts[category] += 1

            # Emit process feedback (planning utility impact)
                weight = self.syntactic_error_penalty
            else:
                category = "semantic"
                weight = self.semantic_error_penalty
            self.error_counts[category] += 1
            await self.emit_event(
                "oak.curation_feedback",
                category=category,
                success=False,
                error=error_msg[:256],
                weight=weight,
            )

            # Global planning utility nudge (no specific feature_id attached)

            await self.emit_event(
                "oak.feature_utility_updated",
                feature_id="global_planning",
                signal_type="planning",
                value=signal,
                components={"planning": signal},
            )

            await self._emit_utility_update(
                "global_planning", "planning", signal, {"planning": signal}
            )
        else:
            signal = float(self.cfg["play_weight"])
        except Exception:  # pragma: no cover - defensive
            logger.exception("CurationManager.handle_tool_result failed")

    async def handle_prediction_error(self, event: Any) -> None:
        """Translate a prediction error into a planning feedback signal."""
        try:
            err_val = float(getattr(event, "error", event.get("error", 0.0)))  # type: ignore[arg-type]
            # Basic confidence transform (smaller error -> higher weight)
            weight = 1.0 / (1.0 + max(err_val, 0.0))
            await self.emit_event(
                "oak.feature_utility_updated",
                feature_id="global_play",
                signal_type="play",
                value=signal,
                components={"play": signal},
            )

            await self._emit_utility_update(
                "global_play", "play", signal, {"play": signal}
                "oak.curation_feedback",
                category="planning",
                success=True,
                weight=weight,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("CurationManager.handle_prediction_error failed")

    async def handle_prediction_error(self, event: Any) -> None:

        err = float(event.get("error", 0.0))
        signal = max(0.0, 1.0 - err) * float(self.cfg["planning_weight"])

        # Route prediction confidence as a positive signal for planning utility
        err = float(getattr(event, "error", 0.0))
        signal = 1.0 / (1.0 + err)

        await self.emit_event(
            "oak.feature_utility_updated",
            feature_id="global_planning",
            signal_type="planning",
            value=signal,
            components={"planning": signal},
        )

        await self._emit_utility_update(
            "global_planning", "planning", signal, {"planning": signal}
        )
