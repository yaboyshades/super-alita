#!/usr/bin/env python3
"""
___AbilityName___ Ability for Super-Alita

This ability implements ___abilityDescription___ following Super-Alita's
Constitutional principles:
- Article I (Library-First): standalone, reusable
- Article II (Test-First): implementation follows tests
- Article III (Simplicity Gate): minimal necessary complexity
- Article VI (Knowledge Codification): documented patterns
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

try:  # runtime imports are optional in template stage
    from src.core.events import create_event  # type: ignore
except Exception:  # pragma: no cover
    def create_event(event_type: str, **data: Any) -> dict[str, Any]:  # type: ignore
        ev = {"type": event_type}
        ev.update(data)
        return ev

try:
    from src.abilities.base_ability import BaseAbility
except Exception:
    class BaseAbility:  # type: ignore
        pass


class ___AbilityName___Ability(BaseAbility):
    """___AbilityName___ Ability Implementation."""

    # Metadata
    name = "___abilityName___"
    description = "___abilityDescription___"
    version = "___version___"
    author = "___author___"

    # Simple JSON schemas (advisory)
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "___inputField___": {
                "type": "string",
                "minLength": 1,
                "maxLength": 10000,
            },
            "options": {
                "type": "object",
                "properties": {
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
                    "format": {"type": "string", "enum": ["json", "text", "markdown"]},
                    "debug": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
        "required": ["___inputField___"],
        "additionalProperties": False,
    }

    output_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "data": {"type": ["object", "string", "array"]},
            "error": {"type": "string"},
            "metadata": {
                "type": "object",
                "properties": {
                    "execution_time": {"type": "number"},
                    "timestamp": {"type": "string"},
                    "version": {"type": "string"},
                    "execution_id": {"type": "string"},
                },
            },
        },
        "required": ["success"],
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.timeout: int = int(self.config.get("timeout", 30))
        self.max_retries: int = int(self.config.get("max_retries", 3))
        self.debug: bool = bool(self.config.get("debug", False))
        self.enable_logging: bool = bool(self.config.get("enable_logging", True))
        self.max_input_size: int = int(self.config.get("max_input_size", 1024 * 1024))

        self.is_initialized: bool = False
        self.execution_count: int = 0
        self.last_execution_time: datetime | None = None

        self.logger = logging.getLogger(f"super_alita.abilities.{self.name}")
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

    async def initialize(self, event_bus: Any) -> bool:
        """Wire event bus subscriptions and emit initialized event."""
        try:
            self.event_bus = event_bus
            if hasattr(event_bus, "subscribe"):
                await event_bus.subscribe("system_shutdown", self._handle_shutdown)
                await event_bus.subscribe("ability_health_check", self._handle_health_check)
            if hasattr(event_bus, "emit"):
                await event_bus.emit(
                    create_event(
                        "ability_initialized",
                        ability_name=self.name,
                        version=self.version,
                        timestamp=datetime.now(UTC).isoformat(),
                    )
                )
            self.is_initialized = True
            return True
        except Exception as e:  # pragma: no cover
            self.logger.error("Initialization failed: %s", e)
            return False

    def validate_input(self, input_data: Any) -> bool:
        if input_data is None or not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
        if "___inputField___" not in input_data:
            raise ValueError("Required field '___inputField___' is missing")
        val = input_data["___inputField___"]
        if not isinstance(val, str) or len(val) == 0:
            raise ValueError("'___inputField___' must be a non-empty string")
        if len(val.encode("utf-8")) > self.max_input_size:
            raise ValueError("Input size exceeds limit")
        # Security checks (basic)
        s = json.dumps(input_data).lower()
        if "<script" in s or "../" in s or "..\\" in s:
            raise ValueError("Potentially dangerous input detected")
        return True

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        execution_id = hashlib.md5(f"{time.time()}_{self.execution_count}".encode()).hexdigest()[:8]
        start = time.time()
        try:
            self.validate_input(input_data)
            if self.event_bus and hasattr(self.event_bus, "emit"):
                await self.event_bus.emit(
                    create_event(
                        "ability_started",
                        ability_name=self.name,
                        execution_id=execution_id,
                        timestamp=datetime.now(UTC).isoformat(),
                    )
                )
            options = input_data.get("options", {})
            timeout = int(options.get("timeout", self.timeout))
            data = await asyncio.wait_for(self._execute_core(input_data), timeout=timeout)
            elapsed = time.time() - start
            if self.event_bus and hasattr(self.event_bus, "emit"):
                await self.event_bus.emit(
                    create_event(
                        "ability_completed",
                        ability_name=self.name,
                        execution_id=execution_id,
                        success=True,
                        execution_time=elapsed,
                        timestamp=datetime.now(UTC).isoformat(),
                    )
                )
            self.execution_count += 1
            self.last_execution_time = datetime.now(UTC)
            return {
                "success": True,
                "data": data,
                "metadata": {
                    "execution_time": elapsed,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "version": self.version,
                    "execution_id": execution_id,
                },
            }
        except TimeoutError:
            return self._error_result(
                f"Execution timed out after {self.timeout} seconds",
                execution_id,
                time.time() - start,
            )
        except ValueError as e:
            return self._error_result(f"Input validation failed: {e}", execution_id, time.time() - start)
        except Exception as e:  # pragma: no cover
            self.logger.exception("Unexpected error during execution: %s", e)
            return self._error_result(f"Unexpected error: {e}", execution_id, time.time() - start)

    async def _execute_core(self, input_data: dict[str, Any]) -> Any:
        """Core logic placeholder – customize per ability."""
        _ = input_data["___inputField___"]
        await asyncio.sleep(0.1)
        return {
            "processed_input": input_data["___inputField___"],
            "processing_timestamp": datetime.now(UTC).isoformat(),
            "options_applied": input_data.get("options", {}),
            "status": "processed",
        }

    def _error_result(self, msg: str, execution_id: str, elapsed: float) -> dict[str, Any]:
        if self.event_bus and hasattr(self.event_bus, "emit"):
            # fire and forget
            asyncio.create_task(
                self.event_bus.emit(
                    create_event(
                        "ability_failed",
                        ability_name=self.name,
                        execution_id=execution_id,
                        error=msg,
                        execution_time=elapsed,
                        timestamp=datetime.now(UTC).isoformat(),
                    )
                )
            )
        return {
            "success": False,
            "error": msg,
            "metadata": {
                "execution_time": elapsed,
                "timestamp": datetime.now(UTC).isoformat(),
                "version": self.version,
                "execution_id": execution_id,
            },
        }

    async def health_check(self) -> dict[str, Any]:
        status = "healthy" if self.is_initialized else ("degraded" if self.event_bus else "unhealthy")
        return {
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
            "ability": self.name,
            "version": self.version,
            "details": {
                "is_initialized": self.is_initialized,
                "execution_count": self.execution_count,
                "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None,
                "event_bus_connected": bool(self.event_bus),
            },
        }

    async def _handle_shutdown(self, _event: dict[str, Any]) -> None:  # pragma: no cover
        await self.shutdown()

    async def _handle_health_check(self, _event: dict[str, Any]) -> None:  # pragma: no cover
        if self.event_bus and hasattr(self.event_bus, "emit"):
            await self.event_bus.emit(
                create_event(
                    "ability_health_response",
                    ability_name=self.name,
                    health_status=await self.health_check(),
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )

    async def shutdown(self) -> None:  # pragma: no cover
        if self.event_bus and hasattr(self.event_bus, "emit"):
            await self.event_bus.emit(
                create_event(
                    "ability_shutdown",
                    ability_name=self.name,
                    execution_count=self.execution_count,
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )
        self.is_initialized = False

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.name} v{self.version} ({'initialized' if self.is_initialized else 'not initialized'})"

    def __repr__(self) -> str:  # pragma: no cover
        return f"___AbilityName___Ability(name='{self.name}', version='{self.version}', executions={self.execution_count})"


__all__ = ["___AbilityName___Ability"]
