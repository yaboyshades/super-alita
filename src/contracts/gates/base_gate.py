from __future__ import annotations

from typing import Any


class Gate:
    """Gate interface: returns (ok, info)."""

    def validate_latest(self, latest: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        raise NotImplementedError