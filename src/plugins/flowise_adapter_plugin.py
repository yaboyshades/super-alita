#!/usr/bin/env python3
from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import asyncio
import contextlib
import websockets
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FlowiseConfig:
    ws_url: str
    api_key: Optional[str] = None
    enabled: bool = False


class FlowiseAdapterPlugin(PluginInterface):
    """Flowise <-> Super Alita adapter for codegen events."""

    def __init__(self) -> None:
        super().__init__()
        self._cfg = FlowiseConfig(
            ws_url=os.getenv("FLOWISE_WS_URL", "ws://localhost:3000"),
            api_key=os.getenv("FLOWISE_API_KEY"),
            enabled=os.getenv("FLOWISE_ENABLED", "false").lower() == "true",
        )
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._listener_task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return "flowise_adapter"

    async def start(self) -> None:
        await super().start()
        if not self._cfg.enabled:
            logger.info("FlowiseAdapter disabled; not starting.")
            return
        await self.subscribe("codegen_implementation_proposed", self._on_codegen_event)
        await self.subscribe("codegen_ready_for_apply", self._on_codegen_event)
        self._listener_task = asyncio.create_task(self._connect_and_listen())
        logger.info("FlowiseAdapter started")

    async def shutdown(self) -> None:
        logger.info("FlowiseAdapter shutting down")
        if self._listener_task:
            self._listener_task.cancel()
            with contextlib.suppress(Exception):
                await self._listener_task
        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
        await super().shutdown()

    async def _connect_and_listen(self) -> None:
        headers = {"Authorization": f"Bearer {self._cfg.api_key}"} if self._cfg.api_key else None
        while self.is_running:
            try:
                async with websockets.connect(self._cfg.ws_url, extra_headers=headers) as ws:
                    self._ws = ws
                    await self._register_node(ws)
                    async for raw in ws:
                        try:
                            data = json.loads(raw)
                        except Exception:
                            continue
                        await self._handle_inbound(data)
            except Exception as e:
                logger.warning("Flowise WS error: %s; retrying in 2s", e)
                await asyncio.sleep(2)

    async def _register_node(self, ws: websockets.WebSocketClientProtocol) -> None:
        node_def = {
            "type": "register_node",
            "data": {
                "name": "Codegen Generator",
                "class": "CodegenNode",
                "description": "Repo-level code generation via Gemini",
                "inputs": [
                    {"label": "Prompt", "name": "prompt", "type": "string"},
                    {
                        "label": "Repository Path",
                        "name": "repoPath",
                        "type": "string",
                        "optional": True,
                        "default": ".",
                    },
                    {
                        "label": "Context Files",
                        "name": "contextFiles",
                        "type": "string",
                        "optional": True,
                    },
                ],
                "outputs": [
                    {"label": "Generated Diffs", "name": "diffs", "type": "json"},
                    {"label": "Tests", "name": "tests", "type": "json"},
                ],
            },
        }
        await ws.send(json.dumps(node_def))

    async def _handle_inbound(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "execute_node" and msg.get("node") == "CodegenNode":
            inputs = msg.get("inputs") or {}
            session_id = msg.get("sessionId")
            await self.emit_event(
                "codegen_request",
                source_plugin=self.name,
                task_kind="text2backend",
                requirements=inputs.get("prompt") or "",
                repo_path=inputs.get("repoPath") or ".",
                context_files=(inputs.get("contextFiles") or "").splitlines()
                if inputs.get("contextFiles")
                else [],
                flowise_session_id=session_id,
                timestamp=_utcnow(),
            )
        elif mtype == "chat_message":
            message = msg.get("message") or ""
            session_id = msg.get("sessionId")
            if any(k in message.lower() for k in ("generate", "create", "implement", "write code")):
                await self.emit_event(
                    "codegen_request",
                    source_plugin=self.name,
                    task_kind="text2backend",
                    requirements=message,
                    flowise_session_id=session_id,
                    timestamp=_utcnow(),
                )

    async def _on_codegen_event(self, event: Dict[str, Any]) -> None:
        sess = event.get("flowise_session_id")
        if not sess or not self._ws:
            return
        out = {
            "type": "node_result",
            "sessionId": sess,
            "outputs": {
                "diffs": event.get("diffs", []),
                "tests": event.get("tests", []),
            },
            "metadata": {
                "proposal_id": event.get("proposal_id"),
                "event_type": event.get("event_type"),
                "confidence": event.get("confidence"),
                "timestamp": _utcnow(),
            },
        }
        try:
            await self._ws.send(json.dumps(out))
        except Exception as e:
            logger.warning("Flowise send failed: %s", e)
