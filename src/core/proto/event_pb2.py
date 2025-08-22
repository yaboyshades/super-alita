# Stub protobuf module for development
# To generate real protobuf code, run:
# protoc -I proto --python_out src/core/proto event.proto


class BaseEvent:
    """Stub protobuf message for BaseEvent."""

    def __init__(self):
        self.event_type = ""
        self.source_plugin = ""
        self.timestamp = ""
        self.data = ""

    def SerializeToString(self) -> bytes:
        """Stub serialization - returns JSON as bytes."""
        import json

        data = {
            "event_type": self.event_type,
            "source_plugin": self.source_plugin,
            "timestamp": self.timestamp,
            "data": self.data,
        }
        return json.dumps(data).encode("utf-8")

    def ParseFromString(self, data: bytes):
        """Stub parsing - expects JSON bytes."""
        import json

        parsed = json.loads(data.decode("utf-8"))
        self.event_type = parsed.get("event_type", "")
        self.source_plugin = parsed.get("source_plugin", "")
        self.timestamp = parsed.get("timestamp", "")
        self.data = parsed.get("data", "")

    def __str__(self):
        return f"BaseEvent(type={self.event_type}, source={self.source_plugin})"
