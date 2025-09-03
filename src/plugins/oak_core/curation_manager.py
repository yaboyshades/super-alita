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
        except Exception:  # pragma: no cover - defensive
            logger.exception("CurationManager.handle_tool_result failed")

    async def handle_prediction_error(self, event: Any) -> None:
        """Translate a prediction error into a planning feedback signal."""
        try:
            err_val = float(getattr(event, "error", event.get("error", 0.0)))  # type: ignore[arg-type]
            # Basic confidence transform (smaller error -> higher weight)
            weight = 1.0 / (1.0 + max(err_val, 0.0))
            await self.emit_event(
                "oak.curation_feedback",
                category="planning",
                success=True,
                weight=weight,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("CurationManager.handle_prediction_error failed")
