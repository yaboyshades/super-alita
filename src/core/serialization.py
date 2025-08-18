from __future__ import annotations
"""Utility module for fast event serialization."""

from typing import Any, Type

try:  # pragma: no cover - import guard
    import orjson as _json
except Exception:  # pragma: no cover
    import json as _json  # type: ignore


class Serializer:
    """Fast (de)serialization for events."""

    def encode(self, obj: Any) -> bytes:
        """Encode an object to UTF-8 JSON bytes."""
        try:
            return _json.dumps(obj)
        except TypeError:
            if hasattr(obj, "model_dump"):
                return _json.dumps(obj.model_dump(by_alias=True))
            return _json.dumps(obj.__dict__)

    def decode(self, data: bytes | str, target_cls: Type[Any]) -> Any:
        """Decode JSON bytes/str into target class or raw payload."""
        if isinstance(data, bytes):
            payload = _json.loads(data)
        else:
            payload = _json.loads(data.encode("utf-8"))
        if hasattr(target_cls, "model_validate"):
            return target_cls.model_validate(payload)
        return payload
