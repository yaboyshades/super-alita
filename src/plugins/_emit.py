from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.core.plugins.helpers import asafe_publish, safe_publish


def emit_safe(
    event_bus: Any,
    channel: str,
    payload: Any,
    on_warn: Callable[[str], None] | None = None,
) -> bool:
    return safe_publish(event_bus, channel, payload, on_warn=on_warn)


async def aemit_safe(
    event_bus: Any,
    channel: str,
    payload: Any,
    on_warn: Callable[[str], None] | None = None,
) -> bool:
    return await asafe_publish(event_bus, channel, payload, on_warn=on_warn)
