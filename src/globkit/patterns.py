from __future__ import annotations

import glob
from collections.abc import Iterator


def list_files(pattern: str, recursive: bool = False) -> list[str]:
    """Return a list of paths matching a glob pattern."""
    return glob.glob(pattern, recursive=recursive)


def iglob_iter(pattern: str, recursive: bool = False) -> Iterator[str]:
    """Yield paths matching a glob pattern (memory-efficient)."""
    return glob.iglob(pattern, recursive=recursive)
