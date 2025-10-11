"""REUG-maintained asyncio plugin for pytest."""

from __future__ import annotations

import pytest

from ._compat import ensure_fixturedef_alias

ensure_fixturedef_alias(pytest)

pytest_plugins = ("reug_pytest_asyncio.plugin",)

__all__ = ["ensure_fixturedef_alias"]
