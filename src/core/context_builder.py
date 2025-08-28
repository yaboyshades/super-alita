from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .correlation import get_correlation_id, get_session_id

CONTEXT_VERSION = "1.0"

@dataclass
class ContextAssembler:
    """
    Builds normalized context payloads for decision, tool execution, and memory.
    Ensures stable shape + context_version across modules.
    """
    def __init__(self, *, user_input: str = "", recent_events: list[dict[str, Any]] | None = None,
                 memory_hits: list[dict[str, Any]] | None = None, active_goals: list[str] | None = None,
                 tool_inventory: list[dict[str, Any]] | None = None, extras: dict[str, Any] | None = None):
        self.user_input = user_input
        self.recent_events = recent_events or []
        self.memory_hits = memory_hits or []
        self.active_goals = active_goals or []
        self.tool_inventory = tool_inventory or []
        self.extras = extras or {}

    def _base(self) -> dict[str, Any]:
        return {
            "context_version": CONTEXT_VERSION,
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": get_session_id(),
            "correlation_id": get_correlation_id(),
            "user_input": self.user_input,
            "recent_events": self.recent_events,
            "memory_hits": self._normalize_memory_hits(self.memory_hits),
            "active_goals": self.active_goals,
            "tool_inventory": self.tool_inventory,
            "extras": self.extras,
        }

    def build_for_decision(self) -> dict[str, Any]:
        ctx = self._base()
        ctx["for"] = "decision"
        return ctx

    def build_for_tool_execution(self, tool_name: str) -> dict[str, Any]:
        ctx = self._base()
        ctx["for"] = "tool_execution"
        ctx["tool_name"] = tool_name
        return ctx

    def build_for_memory(self) -> dict[str, Any]:
        ctx = self._base()
        ctx["for"] = "memory"
        return ctx

    @staticmethod
    def _normalize_memory_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for h in hits:
            out.append({
                "atom_id": h.get("atom_id"),
                "score": float(h.get("score", 0.0)),
                "truncated_content_hash": h.get("truncated_content_hash"),
            })
        return out
