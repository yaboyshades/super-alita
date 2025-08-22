#!/usr/bin/env python3
"""
Event serialization for Redis-based EventBus
"""

import json
from datetime import datetime

from src.core.events import BaseEvent


class EventSerializerError(Exception):
    """Exception raised when event serialization/deserialization fails."""


class EventSerializer:
    """Handles serialization and deserialization of events for Redis transport."""

    def serialize(self, event: BaseEvent) -> bytes:
        """
        Serialize an event to bytes for Redis storage.

        Args:
            event: The event to serialize

        Returns:
            Serialized event as bytes

        Raises:
            EventSerializerError: If serialization fails
        """
        try:
            # Convert event to dictionary
            if hasattr(event, "model_dump"):
                event_dict = event.model_dump()
            elif hasattr(event, "dict"):
                event_dict = event.dict()
            else:
                # Fallback for basic events
                event_dict = {
                    "event_type": getattr(event, "event_type", "unknown"),
                    "source_plugin": getattr(event, "source_plugin", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                }
                # Add all other attributes
                for attr in dir(event):
                    if not attr.startswith("_") and not callable(getattr(event, attr)):
                        event_dict[attr] = getattr(event, attr)

            # Handle datetime objects
            for key, value in event_dict.items():
                if isinstance(value, datetime):
                    event_dict[key] = value.isoformat()

            # Add event class name for deserialization
            event_dict["_event_class"] = event.__class__.__name__
            event_dict["_event_module"] = event.__class__.__module__

            # Serialize to JSON and then to bytes
            json_str = json.dumps(event_dict, default=str)
            return json_str.encode("utf-8")

        except Exception as e:
            raise EventSerializerError(f"Failed to serialize event {event}: {e}") from e

    def deserialize(self, data: bytes) -> BaseEvent:
        """
        Deserialize bytes back to an event object.

        Args:
            data: Serialized event data

        Returns:
            Deserialized event object

        Raises:
            EventSerializerError: If deserialization fails
        """
        try:
            # Decode bytes to JSON string
            json_str = data.decode("utf-8")
            event_dict = json.loads(json_str)

            # Extract class information
            event_class_name = event_dict.pop("_event_class", "BaseEvent")
            event_module = event_dict.pop("_event_module", "src.core.events")

            # Convert timestamp strings back to datetime if needed
            if "timestamp" in event_dict and isinstance(event_dict["timestamp"], str):
                try:
                    event_dict["timestamp"] = datetime.fromisoformat(
                        event_dict["timestamp"]
                    )
                except (ValueError, TypeError):
                    pass  # Keep as string if conversion fails

            # Try to import and create the original event class
            try:
                module = __import__(event_module, fromlist=[event_class_name])
                event_class = getattr(module, event_class_name)
                return event_class(**event_dict)
            except (ImportError, AttributeError):
                # Fallback to BaseEvent
                from src.core.events import BaseEvent

                return BaseEvent(**event_dict)

        except Exception as e:
            raise EventSerializerError(f"Failed to deserialize event data: {e}") from e
