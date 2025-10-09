"""Regression tests for the sitecustomize compatibility shims."""

from __future__ import annotations

import importlib

import pytest


def test_pytest_fixturedef_alias(monkeypatch):
    """Ensure ``pytest.FixtureDef`` is available for legacy plugins."""
    # Reloading ``sitecustomize`` gives us a fresh view of the alias logic in
    # case a previous test mutated the attribute.
    importlib.reload(importlib.import_module("sitecustomize"))

    from _pytest.fixtures import FixtureDef

    assert hasattr(pytest, "FixtureDef")
    assert pytest.FixtureDef is FixtureDef

    # The alias should be resilient to accidental removal; dropping the attribute
    # and reloading should re-install it.
    monkeypatch.delattr(pytest, "FixtureDef", raising=False)
    importlib.reload(importlib.import_module("sitecustomize"))
    assert pytest.FixtureDef is FixtureDef
