#!/usr/bin/env python3
"""Bridge DeepCode outputs to Puter file writes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.core.plugin_interface import PluginInterface


class DeepCodePuterBridgePlugin(PluginInterface):
    """Subscribe to DeepCode results and mirror files to Puter."""

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "deepcode_puter_bridge"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:  # pragma: no cover - simple
        await super().setup(event_bus, store, config)

    async def start(self) -> None:
        await super().start()
        await self.subscribe("deepcode_ready_for_apply", self._on_ready)

    async def _on_ready(self, event: Any) -> None:
        # Collect all proposed files (diffs, tests, docs)
        files: list[tuple[str, str]] = []

        for diff in getattr(event, "diffs", []) or []:
            files.append((diff.get("path", ""), diff.get("new_content", "")))
        for test in getattr(event, "tests", []) or []:
            files.append((test.get("path", ""), test.get("content", "")))
        for doc in getattr(event, "docs", []) or []:
            files.append((doc.get("path", ""), doc.get("content", "")))

        for path, content in files:
            if not path:
                continue
            await self.emit_event(
                "puter_file_write",
                file_path=path,
                content=content,
                conversation_id=getattr(event, "conversation_id", None),
                correlation_id=getattr(event, "correlation_id", None),
                request_id=getattr(event, "request_id", None),
                proposal_id=getattr(event, "proposal_id", None),
                timestamp=datetime.now(timezone.utc),
            )
