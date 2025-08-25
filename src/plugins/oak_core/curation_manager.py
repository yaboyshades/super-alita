from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict

from src.core.plugin_interface import PluginInterface


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

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:  # type: ignore[override]
        await super().setup(event_bus, store, config)
        self.cfg.update(config or {})
        await self.subscribe("tool_result", self.handle_tool_result)
        await self.subscribe("oak.prediction_error", self.handle_prediction_error)

    async def start(self) -> None:  # type: ignore[override]
        await super().start()

    async def shutdown(self) -> None:  # type: ignore[override]
        await super().shutdown()

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
            await self.emit_event(
                "oak.feature_utility_update",
                feature_id="global_planning",
                signal_type="planning",
                value=signal,
                components={"planning": signal},
            )
        else:
            # Positive play signal on successful tool usage
            signal = float(self.cfg["play_weight"])
            await self.emit_event(
                "oak.feature_utility_update",
                feature_id="global_play",
                signal_type="play",
                value=signal,
                components={"play": signal},
            )

    async def handle_prediction_error(self, event: Any) -> None:
        # Route prediction confidence as a positive signal for planning utility
        err = float(getattr(event, "error", 0.0))
        signal = 1.0 / (1.0 + err)
        await self.emit_event(
            "oak.feature_utility_update",
            feature_id="global_planning",
            signal_type="planning",
            value=signal,
            components={"planning": signal},
        )

