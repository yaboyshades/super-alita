import os
from dataclasses import dataclass


def _getenv(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v


def _getenv_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _getenv_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
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
