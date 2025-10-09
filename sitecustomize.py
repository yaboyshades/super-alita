"""Compatibility shims for the Super Alita runtime.

This module is imported automatically by Python when it exists on the import
path.  We use it to install minimal compatibility patches that keep the test
suite and developer tooling operating across the supported Python versions.
"""

from __future__ import annotations

try:
    import pytest
except Exception:  # pragma: no cover - pytest may not be available.
    pytest = None  # type: ignore[assignment]
else:
    try:
        from _pytest.fixtures import FixtureDef
    except ImportError:  # pragma: no cover - internal API moved.
        FixtureDef = None  # type: ignore[assignment]
    else:
        if pytest is not None and not hasattr(pytest, "FixtureDef") and FixtureDef is not None:
            # pytest-asyncio<0.24 expects ``pytest.FixtureDef`` to exist.
            # Pytest 8 removed the re-export, which breaks plugin import before
            # the test session starts. Installing the alias here keeps the
            # runtime shippable without pinning older tooling.
            pytest.FixtureDef = FixtureDef  # type: ignore[attr-defined]
