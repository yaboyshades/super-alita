"""Canonical error taxonomy helpers for orchestrator events."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Canonical error codes emitted via RunError / RunFailed."""

    TIMEOUT = "TIMEOUT"
    NETWORK_FAILURE = "NETWORK_FAILURE"
    RATE_LIMIT = "RATE_LIMIT"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    ABILITY_FAILURE = "ABILITY_FAILURE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNKNOWN = "UNKNOWN"


_TIMEOUT_TYPES = (
    TimeoutError,
    ConnectionError,
)


def classify_exception(exc: Exception | BaseException | None) -> ErrorCode:
    """Map Python exceptions onto canonical orchestrator error codes."""

    if exc is None:
        return ErrorCode.UNKNOWN

    if isinstance(exc, _TIMEOUT_TYPES):
        return ErrorCode.TIMEOUT

    name = exc.__class__.__name__.lower()

    if "ratelimit" in name or "quota" in name:
        return ErrorCode.RATE_LIMIT

    if any(token in name for token in ("http", "network", "socket")):
        return ErrorCode.NETWORK_FAILURE

    if any(token in name for token in ("validation", "value", "schema")):
        return ErrorCode.VALIDATION_ERROR

    if any(token in name for token in ("ability", "tool", "planner")):
        return ErrorCode.ABILITY_FAILURE

    return ErrorCode.INTERNAL_ERROR


def normalize_error_code(raw: Any) -> ErrorCode:
    """Normalize arbitrary values into the canonical :class:`ErrorCode`."""

    if isinstance(raw, ErrorCode):
        return raw
    if isinstance(raw, str):
        try:
            return ErrorCode(raw.upper())
        except ValueError:
            pass
    return ErrorCode.UNKNOWN


__all__ = ["ErrorCode", "classify_exception", "normalize_error_code"]
