"""Codex Adapter - bridges VS Code/Codex extension to Super Alita.

Handles:
- Code generation requests from Codex
- File system operations
- LSP (Language Server Protocol) interactions
- Extension commands and events
"""

from __future__ import annotations

import logging
from typing import Any

from src.contracts import Adapter, HealthStatus, UnifiedEvent

logger = logging.getLogger(__name__)


class CodexAdapter(Adapter):
    """Adapter for Codex VS Code extension integration.

    Translates between Codex extension events and Super Alita unified events.
    """

    name = "codex"

    def __init__(self, bus: Any):
        """Initialize Codex adapter.

        Args:
            bus: EventBus instance
        """
        super().__init__(bus)
        self.last_health_check: float = 0.0
        self.requests_handled = 0

    async def handle(self, evt: UnifiedEvent) -> None:
        """Handle incoming events from orchestrator.

        Args:
            evt: Event to handle
        """
        # Route events to appropriate handlers
        handlers = {
            "code_generate": self._handle_code_generate,
            "code_review": self._handle_code_review,
            "sdd_command": self._handle_sdd_command,
        }

        handler = handlers.get(evt.event_type)
        if handler:
            await handler(evt)
            self.requests_handled += 1
        else:
            logger.debug(
                f"Codex adapter ignoring event type: {evt.event_type}"
            )

    async def _handle_code_generate(self, evt: UnifiedEvent) -> None:
        """Handle code generation request.

        Args:
            evt: Code generation event
        """
        logger.info(f"Codex: Generating code for {evt.corr_id}")

        payload = evt.payload
        prompt = payload.get("prompt", "")
        context = payload.get("context", {})
        language = payload.get("language", "python")

        # Emit code generation started event
        await self.emit(
            evt_type="code_generate",
            payload={
                "status": "started",
                "prompt": prompt,
                "language": language,
                "context": context,
            },
            corr=evt.corr_id,
            target="orchestrator",
        )

        # Simulate code generation (in real impl, call LLM/codegen service)
        generated_code = (
            f"# Generated code for: {prompt}\n# Language: {language}\n"
        )

        # Emit completion event
        await self.emit(
            evt_type="code_generate",
            payload={
                "status": "completed",
                "code": generated_code,
                "prompt": prompt,
                "language": language,
            },
            corr=evt.corr_id,
            target="orchestrator",
        )

    async def _handle_code_review(self, evt: UnifiedEvent) -> None:
        """Handle code review request.

        Args:
            evt: Code review event
        """
        logger.info(f"Codex: Reviewing code for {evt.corr_id}")

        code = evt.payload.get("code", "")

        # Emit review result
        await self.emit(
            evt_type="code_review",
            payload={
                "status": "completed",
                "code": code,
                "issues": [],  # In real impl, run linters/analyzers
                "suggestions": [],
            },
            corr=evt.corr_id,
            target="orchestrator",
        )

    async def _handle_sdd_command(self, evt: UnifiedEvent) -> None:
        """Handle SDD workflow commands.

        Args:
            evt: SDD command event
        """
        command = evt.payload.get("command", "")
        logger.info(f"Codex: SDD command '{command}' for {evt.corr_id}")

        # Route to appropriate SDD stage
        if command == "specify":
            await self.emit(
                evt_type="sdd_specify",
                payload=evt.payload,
                corr=evt.corr_id,
                target="super_alita",
            )
        elif command == "plan":
            await self.emit(
                evt_type="sdd_plan",
                payload=evt.payload,
                corr=evt.corr_id,
                target="super_alita",
            )
        elif command == "tasks":
            await self.emit(
                evt_type="sdd_tasks",
                payload=evt.payload,
                corr=evt.corr_id,
                target="super_alita",
            )

    async def health_check(self) -> HealthStatus:
        """Check health of Codex integration.

        Returns:
            Current health status
        """
        import time

        self.last_health_check = time.time()

        # In real implementation, check:
        # - VS Code connection status
        # - Extension responsiveness
        # - File system access

        return HealthStatus(
            component="codex",
            status="healthy",
            message=f"Handled {self.requests_handled} requests",
            details={
                "requests_handled": self.requests_handled,
                "last_check": self.last_health_check,
            },
        )
