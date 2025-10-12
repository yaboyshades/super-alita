"""Compatibility shim that routes pytest-asyncio to the REUG plugin."""
from __future__ import annotations

from importlib import import_module
from typing import Any

from reug_pytest_asyncio import plugin as _plugin

__all__ = [
    "__version__",
    "pytest_plugins",
    "plugin",
]

__version__ = "0.0.0-reug-shim"
pytest_plugins = ["reug_pytest_asyncio.plugin"]


def __getattr__(name: str) -> Any:
    if name == "plugin":
        return import_module("reug_pytest_asyncio.plugin")
    return getattr(_plugin, name)
