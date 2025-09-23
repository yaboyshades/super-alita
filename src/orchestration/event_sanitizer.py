from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .canonical_events import CanonicalEvent

_SENSITIVE_SUBSTRINGS = ("key", "token", "secret", "password")


def sanitize_config_snapshot(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a sanitized copy of the config with sensitive values redacted."""

    if config is None:
        return {}
    return _sanitize_mapping(config)


def truncate_preview(value: str | None, limit: int = 200) -> str | None:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    suffix = "..." if limit >= 3 else ""
    head_len = max(limit - len(suffix), 0)
    return value[:head_len] + suffix


def compute_args_hash(args: Mapping[str, Any]) -> str:
    sanitized = _strip_sensitive(args)
    materialised = json.dumps(sanitized, sort_keys=True)
    return hashlib.sha256(materialised.encode("utf-8")).hexdigest()[:16]


def sanitize_event_for_ledger(event: CanonicalEvent) -> dict[str, Any]:
    payload = event.to_dict()
    data = payload.get("data")
    if isinstance(data, Mapping):
        # Special-case previews that must be truncated
        if "final_output_preview" in data:
            payload["data"]["final_output_preview"] = truncate_preview(
                data["final_output_preview"]
            )
        if "result_preview" in data:
            payload["data"]["result_preview"] = truncate_preview(
                data["result_preview"]
            )
        if payload["kind"] == "RunStarted" and "config" in data:
            payload["data"]["config"] = sanitize_config_snapshot(data["config"])
    if payload.get("meta"):
        payload["meta"] = sanitize_config_snapshot(payload.get("meta") or {})
    return payload


def _sanitize_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        lowered = key.lower()
        if any(token in lowered for token in _SENSITIVE_SUBSTRINGS):
            result[key] = "<redacted>"
            continue
        result[key] = _sanitize_value(value)
    return result


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _strip_sensitive(mapping: Mapping[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in mapping.items():
        lowered = key.lower()
        if any(token in lowered for token in _SENSITIVE_SUBSTRINGS):
            continue
        if isinstance(value, Mapping):
            clean[key] = _strip_sensitive(value)
        elif isinstance(value, list):
            clean[key] = [
                _strip_sensitive(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            clean[key] = value
    return clean


__all__ = [
    "compute_args_hash",
    "sanitize_config_snapshot",
    "sanitize_event_for_ledger",
    "truncate_preview",
]
