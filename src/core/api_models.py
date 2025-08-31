from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def api_error(error: str, code: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "error": error,
        "code": code,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **({"details": details} if details else {}),
    }


__all__ = ["api_error"]

