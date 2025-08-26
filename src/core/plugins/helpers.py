"""Plugin helper utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def safe_publish(
    event_bus: Any,
    channel: str,
    payload: Any,
    on_warn: Callable[[str], None] | None = None,
) -> bool:
    """Synchronously publish if event_bus has 'emit'/'publish', else no-op.

    Returns True if an emit/publish was attempted, False if skipped.
    """
    if event_bus is None:
        if on_warn:
            on_warn(f"safe_publish: skipped (no event_bus) channel={channel}")
        return False
    pub = None
    if hasattr(event_bus, "emit") and callable(event_bus.emit):
        pub = event_bus.emit
    elif hasattr(event_bus, "publish") and callable(event_bus.publish):
        pub = event_bus.publish
    if pub is None:
        if on_warn:
            on_warn(f"safe_publish: no emit/publish on event_bus for {channel}")
        return False
    pub(channel, payload)
    return True


async def asafe_publish(
    event_bus: Any,
    channel: str,
    payload: Any,
    on_warn: Callable[[str], None] | None = None,
) -> bool:
    """Async-friendly publish if event_bus has async/sync emit/publish."""
    if event_bus is None:
        if on_warn:
            on_warn(f"asafe_publish: skipped (no event_bus) channel={channel}")
        return False
    cand = None
    for name in ("emit", "publish"):
        if hasattr(event_bus, name):
            cand = getattr(event_bus, name)
            break
    if cand is None:
        if on_warn:
            on_warn(f"asafe_publish: no emit/publish on event_bus for {channel}")
            return False
    if cand is None:  # double-check (static analyzers)
        if on_warn:
            on_warn(f"asafe_publish: internal error (no callable) for {channel}")
        return False
    try:
        res = cand(channel, payload)
        if hasattr(res, "__await__") or hasattr(res, "__aiter__"):
            await res  # type: ignore
    except TypeError:
        res = cand(payload)  # type: ignore
        if hasattr(res, "__await__") or hasattr(res, "__aiter__"):
            await res  # type: ignore
    return True
