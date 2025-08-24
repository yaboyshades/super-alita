#!/usr/bin/env python3
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class DeepCodeGeneratorBridgePlugin(PluginInterface):
    """
    Bridge plugin accepting legacy Super Alita requests and forwarding them
    into the normalized DeepCode orchestrator pipeline as `deepcode_request`.

    Listens:
      - code_generation_request
      - repository_analysis_request

    Emits:
      - deepcode_request
      - cognitive_turn (breadcrumb)
    """
    @property
    def name(self) -> str:
        return "deepcode_generator"

    async def setup(self, event_bus: Any, store: Any, config: Dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        logger.info("DeepCodeGeneratorBridge setup complete")

    async def start(self) -> None:
        await super().start()
        await self.subscribe("code_generation_request", self._handle_generation)
        await self.subscribe("repository_analysis_request", self._handle_analysis)
        logger.info("DeepCodeGeneratorBridge started (listening for legacy events)")

    async def shutdown(self) -> None:
        logger.info("DeepCodeGeneratorBridge shutting down")
        await super().shutdown()

    async def _handle_generation(self, event: Dict[str, Any]) -> None:
        if not self.is_running:
            return
        prompt = event.get("prompt") or event.get("data", {}).get("prompt") or ""
        repo_path = event.get("repo_path") or event.get("data", {}).get("repo_path") or "."
        conversation_id = event.get("conversation_id")

        await self.emit_event(
            "cognitive_turn",
            source_plugin=self.name,
            stage="deepcode_bridge_processing",
            confidence=0.85,
            conversation_id=conversation_id,
            timestamp=_utcnow(),
        )
        await self.emit_event(
            "deepcode_request",
            source_plugin=self.name,
            task_kind="text2backend",
            requirements=prompt,
            repo_path=repo_path,
            conversation_id=conversation_id,
            timestamp=_utcnow(),
        )

    async def _handle_analysis(self, event: Dict[str, Any]) -> None:
        if not self.is_running:
            return
        repo_path = event.get("repo_path") or "."
        conversation_id = event.get("conversation_id")
        await self.emit_event(
            "deepcode_request",
            source_plugin=self.name,
            task_kind="analyze",
            requirements=f"Analyze repository at {repo_path}",
            repo_path=repo_path,
            conversation_id=conversation_id,
            timestamp=_utcnow(),
        )

def create_plugin():
    return DeepCodeGeneratorBridgePlugin()
