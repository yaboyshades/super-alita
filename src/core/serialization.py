"""
Serializer interface + built-in JSON / Protobuf adapters.
Protobuf schema lives in proto/event.proto (compiled by protoc).
Enhanced with structured event routing for proper type validation.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Import events at module level to avoid circular imports
try:
    from .events import EVENT_ALIASES, EVENT_TYPE_TO_MODEL, EVENT_TYPES, BaseEvent
except ImportError:
    # Fallback if events module isn't available yet
    EVENT_ALIASES = {}
    EVENT_TYPES = {}
    EVENT_TYPE_TO_MODEL = {}
    BaseEvent = None

# Timestamp epoch millisecond threshold
EPOCH_MS_THRESHOLD = 1e12


class EventSerializer:
    """Serializer that routes events by type for proper Pydantic validation."""

    def __init__(self):
        # Import here to avoid circular imports
        self.BaseEvent = BaseEvent
        self.EVENT_TYPE_TO_MODEL = EVENT_TYPE_TO_MODEL

    def encode(self, event: Any) -> dict:
        """Encode event to clean dictionary."""
        if hasattr(event, "model_dump"):
            return event.model_dump()
        if hasattr(event, "dict"):
            return event.dict()
        # Fallback for non-Pydantic objects
        return dict(event) if hasattr(event, "__dict__") else event

    def decode(self, payload: dict, default_cls: type | None = None) -> Any:
        """Decode payload to proper event type based on event_type field."""
        if default_cls is None:
            default_cls = self.BaseEvent or dict

        event_type = payload.get("event_type")
        if not event_type:
            logger.warning("Payload missing event_type, using default class")
            if hasattr(default_cls, "model_validate"):
                return default_cls.model_validate(payload)
            return default_cls(**payload) if default_cls is not dict else payload

        # Route to specific model if known
        model_cls = self.EVENT_TYPE_TO_MODEL.get(event_type, default_cls)

        if model_cls != default_cls:
            logger.debug(f"Routing event_type '{event_type}' to {model_cls.__name__}")
        else:
            logger.debug(
                f"Unknown event_type '{event_type}', using {default_cls.__name__ if hasattr(default_cls, '__name__') else str(default_cls)}"
            )

        if hasattr(model_cls, "model_validate"):
            return model_cls.model_validate(payload)
        if model_cls is dict:
            return payload
        return model_cls(**payload)


class Serializer(ABC):
    """Abstract serializer interface for pluggable wire formats."""

    @abstractmethod
    def encode(self, obj: Any) -> bytes:
        """Encode a Pydantic object to bytes."""

    @abstractmethod
    def decode(self, data: bytes, cls: type) -> Any:
        """Decode bytes back to a Pydantic object of the given class."""


# ---- JSON Implementation -----------------------------------------------
class JsonSerializer(Serializer):
    """JSON serializer using Pydantic's built-in JSON support with normalization."""

    def encode(self, obj) -> bytes:
        """Encode Pydantic object to JSON bytes."""
        if hasattr(obj, "model_dump_json"):
            return obj.model_dump_json().encode("utf-8")
        # Fallback for non-Pydantic objects
        return json.dumps(obj).encode("utf-8")

    def decode(self, data: bytes, cls):
        """
        Decode JSON bytes to Pydantic object with payload normalization.
        Back-compat, plus light normalization for conversation payloads so
        legacy producers ('conversation_message', 'user_message') don't explode.
        """
        raw = data.decode("utf-8") if isinstance(data, bytes | bytearray) else data
        try:
            payload = json.loads(raw)
        except Exception:
            # Fall back to pydantic JSON path if not plain JSON
            if hasattr(cls, "model_validate_json"):
                return cls.model_validate_json(raw)
            return cls(**json.loads(raw))

        # Event-type alias: conversation_message → conversation
        et = payload.get("event_type")
        if et == "conversation_message":
            payload["event_type"] = "conversation"

        # If target is ConversationEvent or we see its shape, ensure 'text'
        tn = getattr(cls, "__name__", "")
        looks_like_convo = et in ("conversation", "conversation_message")
        if (
            (tn == "ConversationEvent" or looks_like_convo)
            and "text" not in payload
            and "user_message" in payload
        ):
            payload["text"] = payload.get("user_message")

        # Normalize payload for event aliases and timestamp robustness
        normalized_dict = self._normalize_event_payload(payload)

        # Handle model class selection for aliased event types
        target_cls = self._select_model_class(normalized_dict, cls)

        if hasattr(target_cls, "model_validate"):
            return target_cls.model_validate(normalized_dict)
        # Fallback for non-Pydantic classes
        return target_cls(**normalized_dict)

    def _normalize_event_payload(self, payload: dict) -> dict:
        """Normalize event payload for aliases and timestamp robustness."""
        # Apply event type aliases
        event_type = payload.get("event_type", "")
        if event_type in EVENT_ALIASES:
            payload["event_type"] = EVENT_ALIASES[event_type]

        # ---- Conversation field compatibility shims ----
        # Some publishers send 'text', others 'user_message' - ensure both are present
        if "text" in payload and "user_message" not in payload:
            payload["user_message"] = payload["text"]
        if "user_message" in payload and "text" not in payload:
            payload["text"] = payload["user_message"]

        # Session ID variants
        if "session" in payload and "session_id" not in payload:
            payload["session_id"] = payload["session"]

        # Conversation ID variants
        if "conv_id" in payload and "conversation_id" not in payload:
            payload["conversation_id"] = payload["conv_id"]

        # Robust timestamp decoding (epoch ms/s or ISO → aware UTC)
        if "timestamp" in payload:
            ts_value = payload["timestamp"]
            try:
                if isinstance(ts_value, int | float):
                    # Heuristic: if looks like ms, convert to seconds
                    if ts_value > EPOCH_MS_THRESHOLD:
                        ts_value = ts_value / 1000.0
                    payload["timestamp"] = datetime.fromtimestamp(ts_value, tz=UTC)
                elif isinstance(ts_value, str):
                    # Accept 'Z' and naive ISO; coerce to UTC
                    ts_str = ts_value.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(ts_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    payload["timestamp"] = dt
            except Exception:
                # Leave timestamp to model default on parse failures
                payload.pop("timestamp", None)

        return payload

    def _select_model_class(self, payload: dict, default_cls: type) -> type:
        """Select appropriate model class based on event type."""
        event_type = payload.get("event_type", "")
        # Get the correct class for this event type, fallback to default
        return EVENT_TYPES.get(event_type, default_cls)


# ---- Protobuf Implementation (Optional) --------------------------------
try:
    # Try to import generated protobuf modules
    from .proto.event_pb2 import BaseEvent as PBEvent

    class ProtobufSerializer(Serializer):
        """Protobuf serializer with Pydantic <-> Proto mapping."""

        def encode(self, obj) -> bytes:
            """Convert Pydantic object to Protobuf and serialize."""
            pb = PBEvent()

            # Map common Pydantic fields to Protobuf
            if hasattr(obj, "event_type"):
                pb.event_type = obj.event_type
            if hasattr(obj, "source_plugin"):
                pb.source_plugin = obj.source_plugin
            if hasattr(obj, "timestamp"):
                pb.timestamp = obj.timestamp

            # Handle event-specific data as JSON in a data field
            if hasattr(obj, "model_dump"):
                pb.data = obj.model_dump_json()

            return pb.SerializeToString()

        def decode(self, data: bytes, cls):
            """Parse Protobuf and convert back to Pydantic object."""
            pb = PBEvent()
            pb.ParseFromString(data)

            # Convert protobuf fields back to dict for Pydantic
            event_data = {
                "event_type": pb.event_type,
                "source_plugin": pb.source_plugin,
                "timestamp": pb.timestamp,
            }

            # Parse JSON data field if present
            if pb.data:
                data_dict = json.loads(pb.data)
                event_data.update(data_dict)

            return cls(**event_data)

except ImportError:
    # Graceful fallback if protobuf modules aren't available
    ProtobufSerializer = type(None)  # Type placeholder


# ---- Factory Function --------------------------------------------------
def get_serializer(wire_format: str = "json"):
    """Factory function to get the appropriate serializer."""
    if wire_format == "json":
        return JsonSerializer()
    if wire_format == "protobuf":
        if ProtobufSerializer is None or ProtobufSerializer is type(None):
            raise ImportError(
                "Protobuf serializer requested but protobuf modules not available. "
                "Run: protoc -I proto --python_out src/core/proto event.proto"
            )
        return ProtobufSerializer()
    raise ValueError(f"Unknown wire format: {wire_format}")
