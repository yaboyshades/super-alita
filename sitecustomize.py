"""Compatibility shims for the Super Alita runtime.

This module is imported automatically by Python when it exists on the import
path.  We use it to install minimal compatibility patches that keep the test
suite and developer tooling operating across the supported Python versions.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from collections.abc import Sequence
from contextlib import suppress
from types import ModuleType

from reug_pytest_asyncio._compat import ensure_fixturedef_alias


def _install_fixturedef_alias(pytest_module: ModuleType) -> bool:
    """Install ``pytest.FixtureDef`` for legacy plugins."""

    return ensure_fixturedef_alias(pytest_module)


def _ensure_pytest_alias() -> bool:
    """Attempt to import pytest immediately and patch it."""

    try:
        import pytest as pytest_module
    except Exception:  # pragma: no cover - pytest may not be available yet.
        return False

    return _install_fixturedef_alias(pytest_module)


if not _ensure_pytest_alias():

    class _PytestAliasFinder(importlib.abc.MetaPathFinder):
        """Inject a loader that guarantees ``pytest.FixtureDef``."""

        class _Loader(importlib.abc.Loader):
            def __init__(
                self,
                original_loader: importlib.abc.Loader,
                finder: _PytestAliasFinder,
            ) -> None:
                self._original_loader = original_loader
                self._finder = finder

            def create_module(
                self, spec: importlib.machinery.ModuleSpec
            ) -> ModuleType | None:  # pragma: no cover - delegated call
                create_module = getattr(self._original_loader, "create_module", None)
                if create_module is not None:
                    return create_module(spec)
                return None

            def exec_module(self, module: ModuleType) -> None:
                exec_module = getattr(self._original_loader, "exec_module", None)
                if exec_module is not None:
                    exec_module(module)

                _install_fixturedef_alias(module)

                with suppress(ValueError):  # pragma: no cover - finder removed
                    sys.meta_path.remove(self._finder)

        def find_spec(
            self,
            fullname: str,
            path: Sequence[str] | None,
            _target: ModuleType | None = None,
        ) -> importlib.machinery.ModuleSpec | None:
            if fullname != "pytest":
                return None

            # Delegate to the default path finder so the real loader is used.
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec

            spec.loader = self._Loader(spec.loader, self)
            return spec

    sys.meta_path.insert(0, _PytestAliasFinder())
