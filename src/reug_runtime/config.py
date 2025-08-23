"""Configuration utilities for the REUG runtime.

This module centralizes environment driven settings and exposes a small
``Settings`` dataclass that other modules can rely on.  All helpers use a
predictable ``_getenv`` pattern and include Google style docstrings so the
expected arguments and return types are explicit.
"""

import os
from dataclasses import dataclass


def _getenv(name: str, default: str | None = None) -> str | None:
    """Return the value of an environment variable.

    Args:
        name: Name of the environment variable to read.
        default: Value returned when the variable is not set.

    Returns:
        The string value stored in the environment or ``default`` when the
        variable is missing.
    """
    v = os.getenv(name, default)
    return v


def _getenv_float(name: str, default: float) -> float:
    """Retrieve a floating point environment variable.

    Args:
        name: Environment variable name.
        default: Fallback value when the variable is unset or invalid.

    Returns:
        The parsed floating point value or ``default`` if conversion fails.
    """
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _getenv_int(name: str, default: int) -> int:
    """Retrieve an integer environment variable.

    Args:
        name: Environment variable name.
        default: Fallback value when the variable is unset or invalid.

    Returns:
        The parsed integer value or ``default`` if conversion fails.
    """
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _getenv_bool(name: str, default: bool) -> bool:
    """Retrieve a boolean environment variable.

    The function interprets a number of common truthy strings
    (``"1"``, ``"true"``, ``"yes"``, ``"on"``) as ``True``. All other values
    result in ``False``.

    Args:
        name: Environment variable name.
        default: Fallback value when the variable is unset.

    Returns:
        ``True`` or ``False`` depending on the variable content, or ``default``
        if the variable is missing.
    """
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """Runtime configuration resolved from environment variables.

    Attributes:
        max_tool_calls: Maximum number of tool invocations per turn.
        tool_timeout_s: Timeout for a single tool execution in seconds.
        model_stream_timeout_s: Timeout for model streaming in seconds.
        max_retries: Number of retry attempts after the initial try.
        retry_base_ms: Base delay in milliseconds used for exponential backoff.
        schema_enforce: Whether to enforce tool input schemas.
        event_log_dir: Optional directory for event log storage.
        tool_registry_dir: Optional directory for tool registration artifacts.
    """

    # Execution limits / guardrails
    max_tool_calls: int = _getenv_int("REUG_MAX_TOOL_CALLS", 5)
    tool_timeout_s: float = _getenv_float("REUG_EXEC_TIMEOUT_S", 20.0)
    model_stream_timeout_s: float = _getenv_float("REUG_MODEL_STREAM_TIMEOUT_S", 60.0)
    max_retries: int = _getenv_int("REUG_EXEC_MAX_RETRIES", 1)
    retry_base_ms: int = _getenv_int("REUG_RETRY_BASE_MS", 250)
    schema_enforce: bool = _getenv_bool("REUG_SCHEMA_ENFORCE", True)

    # Observability / storage (used indirectly by your EventBus/KG)
    event_log_dir: str | None = _getenv("REUG_EVENT_LOG_DIR")
    tool_registry_dir: str | None = _getenv("REUG_TOOL_REGISTRY_DIR")


SETTINGS = Settings()
