from __future__ import annotations
import logging
import re
from collections import defaultdict
from typing import Any, Dict

from src.core.plugin_interface import PluginInterface


logger = logging.getLogger(__name__)


class CurationManager(PluginInterface):
    """Curates growth/retire signals using play/prove and error diagnostics.

    Emits:
      - oak.curation_feedback
      - oak.feature_utility_update (for play/planning weights)
    Subscribes:
      - tool_result (existing event)
      - oak.prediction_error
    Alignment: incorporates LiveMCP-style diagnostics (syntactic vs semantic errors)
    and process-focused signals highlighted in helpful_oak_info.md.
    """

    @property
    def name(self) -> str:  # type: ignore[override]
        return "oak_curation_manager"

    def __init__(self) -> None:
        super().__init__()
        self.cfg: dict[str, Any] = {
            "play_weight": 0.1,
            "planning_weight": 0.2,
            "semantic_error_penalty": -0.2,
            "syntactic_error_penalty": -0.1,
        }
        self.error_counts: Dict[str, int] = defaultdict(int)
        self._required_features = {"global_play", "global_planning"}

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:  # type: ignore[override]
        await super().setup(event_bus, store, config)
        self.cfg.update(config or {})
        await self.subscribe("tool_result", self.handle_tool_result)
        await self.subscribe("oak.prediction_error", self.handle_prediction_error)
        await self._ensure_global_features()

    async def start(self) -> None:  # type: ignore[override]
        await super().start()

    async def _ensure_global_features(self) -> None:
        """Ensure required global features exist in the shared store."""
        if not self.store:
            return
        for fid in self._required_features:
            if self._feature_exists(fid):
                continue
            created = False
            try:
                if hasattr(self.store, "create_feature"):
                    self.store.create_feature(fid)  # type: ignore[attr-defined]
                    created = True
                elif hasattr(self.store, "features"):
                    self.store.features[fid] = {}  # type: ignore[index]
                    created = True
            except Exception:  # pragma: no cover - defensive
                created = False
            if not created:
                logger.warning(
                    "CurationManager missing feature %s and could not create it", fid
                )

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
        self, feature_id: str, signal_type: str, value: float, components: dict[str, float]
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

    async def handle_tool_result(self, event: Any) -> None:
        success = bool(getattr(event, "success", False))
        error_msg = getattr(event, "error", "") or ""
        conv_id = getattr(event, "conversation_id", None)
        # Heuristic classification per helpful_oak_info.md categories
        category = None
        if not success:
            if re.search(r"schema|validation|type|required", error_msg, re.I):
                category = "syntactic"
                signal = self.cfg["syntactic_error_penalty"]
            else:
                category = "semantic"
                signal = self.cfg["semantic_error_penalty"]
            self.error_counts[category] += 1  # type: ignore[index]
            # Emit process feedback (planning utility impact)
            await self.emit_event(
                "oak.curation_feedback",
                category=category or "unknown",
                success=False,
                error=str(error_msg)[:256],
            )
            # Global planning utility nudge (no specific feature_id attached)
            await self._emit_utility_update(
                "global_planning", "planning", signal, {"planning": signal}
            )
        else:
            # Positive play signal on successful tool usage
            signal = float(self.cfg["play_weight"])
            await self._emit_utility_update(
                "global_play", "play", signal, {"play": signal}
            )

    async def handle_prediction_error(self, event: Any) -> None:
        # Route prediction confidence as a positive signal for planning utility
        err = float(getattr(event, "error", 0.0))
        signal = 1.0 / (1.0 + err)
        await self._emit_utility_update(
            "global_planning", "planning", signal, {"planning": signal}
        )

