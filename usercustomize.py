"""Ensure runtime compatibility shims load in environments that skip sitecustomize."""
from __future__ import annotations

from contextlib import suppress

with suppress(Exception):  # pragma: no cover - best effort shim.
    import sitecustomize  # noqa: F401 - imported for side effects.
