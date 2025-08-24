#!/usr/bin/env python3
from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import aiohttp
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DifyConfig:
    api_url: str
    api_key: Optional[str] = None
    workflow_id: Optional[str] = None
    enable_streaming: bool = True
    use_codegen_for_code: bool = True
    enabled: bool = False


class DifyAdapterPlugin(PluginInterface):
    """Bridge Dify requests to generic codegen events."""

    def __init__(self) -> None:
        super().__init__()
        self._session: aiohttp.ClientSession | None = None
        self._cfg = DifyConfig(
            api_url=os.getenv("DIFY_API_URL", "http://localhost:3000/api"),
            api_key=os.getenv("DIFY_API_KEY"),
            workflow_id=os.getenv("DIFY_WORKFLOW_ID") or None,
            enable_streaming=os.getenv("DIFY_STREAMING", "true").lower() == "true",
            use_codegen_for_code=os.getenv("DIFY_USE_CODEGEN", "true").lower()
            == "true",
            enabled=os.getenv("DIFY_ENABLED", "false").lower() == "true",
        )

    @property
    def name(self) -> str:
        return "dify_adapter"

    async def setup(
        self, event_bus: Any, store: Any, config: Dict[str, Any]
    ) -> None:
        await super().setup(event_bus, store, config)
        logger.info("DifyAdapter setup (enabled=%s)", self._cfg.enabled)

    async def start(self) -> None:
        await super().start()
        if not self._cfg.enabled:
            logger.info("DifyAdapter disabled; not subscribing.")
            return
        self._session = aiohttp.ClientSession()
        await self.subscribe("dify_request", self._on_dify_request)
        await self.subscribe("codegen_implementation_proposed", self._on_codegen_result)
        await self.subscribe("codegen_ready_for_apply", self._on_codegen_result)
        logger.info("DifyAdapter started")

    async def shutdown(self) -> None:
        logger.info("DifyAdapter shutting down")
        if self._session:
            await self._session.close()
        await super().shutdown()

    async def _on_dify_request(self, event: Dict[str, Any]) -> None:
        if not self.is_running or not self._cfg.enabled:
            return
        action = (event.get("action") or "generate_code").lower()
        if action == "generate_code" and self._cfg.use_codegen_for_code:
            await self.emit_event(
                "codegen_request",
                source_plugin=self.name,
                task_kind="text2backend",
                requirements=event.get("prompt") or "",
                repo_path=event.get("repo_path") or ".",
                context_files=event.get("context_files") or [],
                dify_callback_id=event.get("dify_callback_id"),
                conversation_id=event.get("conversation_id"),
                timestamp=_utcnow(),
            )
        else:
            await self.emit_event(
                "dify_callback_completed",
                source_plugin=self.name,
                info=f"action {action} routed internally (not codegen)",
                timestamp=_utcnow(),
            )

    async def _on_codegen_result(self, event: Dict[str, Any]) -> None:
        if not self._session:
            return
        callback_id = event.get("dify_callback_id") or event.get("data", {}).get(
            "dify_callback_id"
        )
        if not callback_id:
            return
        payload = {
            "callback_id": callback_id,
            "result": {
                "event_type": event.get("event_type"),
                "proposal_id": event.get("proposal_id"),
                "diffs": event.get("diffs", []),
                "tests": event.get("tests", []),
                "docs": event.get("docs", []),
                "validation": event.get("validation"),
                "success": event.get("success", None),
                "confidence": event.get("confidence", None),
                "timestamp": _utcnow(),
            },
        }
        try:
            async with self._session.post(
                f"{self._cfg.api_url}/callbacks/{callback_id}",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._cfg.api_key}"
                }
                if self._cfg.api_key
                else None,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                await resp.text()
                await self.emit_event(
                    "dify_callback_completed",
                    source_plugin=self.name,
                    status=resp.status,
                    callback_id=callback_id,
                    timestamp=_utcnow(),
                )
        except Exception as e:
            logger.warning("Dify callback failed: %s", e)
