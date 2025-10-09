"""Regression tests for ``src.reug_runtime.config.Settings`` environment overrides."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest


MODULE_NAME = "src.reug_runtime.config"
ENV_VARS = (
    "REUG_MAX_TOOL_CALLS",
    "REUG_EXEC_TIMEOUT_S",
    "REUG_MODEL_STREAM_TIMEOUT_S",
    "REUG_MESSAGE_OPTIMIZER_ENABLED",
    "REUG_MESSAGE_OPTIMIZER_TELEMETRY",
    "REUG_MESSAGE_OPTIMIZER_MAX_LEN",
    "REUG_SCHEMA_ENFORCE",
)


@pytest.fixture(autouse=True)
def reset_settings_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test sees a clean module state and unset REUG env vars."""

    for key in ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    if MODULE_NAME in sys.modules:
        monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    yield
    if MODULE_NAME in sys.modules:
        monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)


@pytest.mark.parametrize(
    ("env_key", "env_value", "attribute", "expected"),
    (
        ("REUG_EXEC_TIMEOUT_S", "15.5", "tool_timeout_s", 15.5),
        ("REUG_MODEL_STREAM_TIMEOUT_S", "120", "model_stream_timeout_s", 120.0),
        ("REUG_MAX_TOOL_CALLS", "7", "max_tool_calls", 7),
        ("REUG_MESSAGE_OPTIMIZER_ENABLED", "off", "message_optimizer_enabled", False),
        ("REUG_MESSAGE_OPTIMIZER_TELEMETRY", "0", "message_optimizer_emit_telemetry", False),
        ("REUG_MESSAGE_OPTIMIZER_MAX_LEN", "4096", "message_optimizer_max_len", 4096),
        ("REUG_SCHEMA_ENFORCE", "false", "schema_enforce", False),
    ),
)
def test_settings_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    env_key: str,
    env_value: str,
    attribute: str,
    expected: Any,
) -> None:
    """``Settings`` must honour environment overrides for loop-critical toggles."""

    monkeypatch.setenv(env_key, env_value)
    module = importlib.import_module(MODULE_NAME)
    settings = getattr(module, "SETTINGS")
    assert getattr(settings, attribute) == expected
