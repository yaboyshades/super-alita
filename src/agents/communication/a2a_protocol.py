"""Agent-to-agent (A2A) communication protocol helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

__all__ = [
    "A2AProtocol",
    "AgentMessageEvent",
    "create_security_context",
]

_DEFAULT_PROTOCOL_VERSION = "a2a-1.0"


def _now_iso() -> str:
    """Return a UTC ISO-8601 timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class AgentMessageEvent:
    """Envelope describing an agent-to-agent message event."""

    sender_id: str
    recipient_id: str
    message_type: str
    payload: dict[str, Any]
    security_context: dict[str, Any]
    protocol_version: str = _DEFAULT_PROTOCOL_VERSION
    priority: str = "medium"
    event_type: str = "agent_message"
    correlation_id: str | None = None
    timestamp: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the envelope to a serializable mapping."""

        data: dict[str, Any] = {
            "event_type": self.event_type,
            "protocol_version": self.protocol_version,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type,
            "payload": dict(self.payload),
            "security_context": dict(self.security_context),
            "priority": self.priority,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }
        if self.correlation_id is not None:
            data["correlation_id"] = self.correlation_id
        return data


def create_security_context(
    sender_id: str,
    recipient_id: str,
    *,
    correlation_id: str | None = None,
    channel: str = "agent_to_agent",
    claims: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a lightweight security context for agent messaging."""

    context: dict[str, Any] = {
        "issuer": sender_id,
        "audience": recipient_id,
        "channel": channel,
        "issued_at": _now_iso(),
        "nonce": str(uuid.uuid4()),
    }
    if correlation_id is not None:
        context["correlation_id"] = correlation_id
    if claims:
        context["claims"] = dict(claims)
    return context


class A2AProtocol:
    """Publish agent-to-agent messages through the runtime event bus."""

    def __init__(self, event_bus: Any, *, protocol_version: str = _DEFAULT_PROTOCOL_VERSION) -> None:
        self._event_bus = event_bus
        self._protocol_version = protocol_version

    async def agent_to_agent(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: str,
        payload: Mapping[str, Any],
        *,
        priority: str = "medium",
        correlation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create and publish an ``AgentMessageEvent`` envelope."""

        payload_dict = dict(payload)
        metadata_dict = dict(metadata or {})
        security_context = create_security_context(
            sender_id,
            recipient_id,
            correlation_id=correlation_id,
            claims=metadata_dict.get("claims"),
        )
        event = AgentMessageEvent(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload_dict,
            security_context=security_context,
            protocol_version=self._protocol_version,
            priority=priority,
            correlation_id=correlation_id,
            metadata=metadata_dict,
        )
        event_dict = event.to_dict()
        event_dict.setdefault("priority", priority)

        publisher = getattr(self._event_bus, "publish", None)
        if callable(publisher):
            await publisher(event_dict, priority=priority)
        else:
            emitter = getattr(self._event_bus, "emit", None)
            if not callable(emitter):  # pragma: no cover - defensive guard
                raise AttributeError("Event bus does not support publish or emit")
            await emitter(event_dict)
        return event_dict
