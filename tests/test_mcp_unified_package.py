"""Smoke tests for the unified MCP package layout."""

from __future__ import annotations

import importlib
import types
import warnings

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "src.mcp.server.mcp_server",
        "src.mcp.client.mcp_client",
        "src.mcp.client.router",
        "src.mcp.protocol.events",
        "src.mcp.integrations.super_alita.handlers",
    ],
)
def test_new_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert isinstance(module, types.ModuleType)


@pytest.mark.parametrize(
    "shim_name",
    [
        "src.mcp_server.server",
        "src.mcp_local.clients",
        "src.super_alita_mcp.super_alita",
    ],
)
def test_shims_raise_deprecation_warning(shim_name: str) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(shim_name)
        assert isinstance(module, types.ModuleType)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
