"""Compatibility helpers for the REUG pytest-asyncio shim."""

from __future__ import annotations

from types import ModuleType


def ensure_fixturedef_alias(pytest_module: ModuleType | None = None) -> bool:
    """Ensure ``pytest.FixtureDef`` is available for legacy plugins."""

    try:
        from _pytest.fixtures import FixtureDef
    except ImportError:  # pragma: no cover - pytest internals moved
        return False

    if pytest_module is None:
        try:
            import pytest as pytest_module  # type: ignore[assignment]
        except Exception:  # pragma: no cover - pytest not available yet
            return False

    if hasattr(pytest_module, "FixtureDef"):
        return True

    pytest_module.FixtureDef = FixtureDef  # type: ignore[attr-defined]
    return True
