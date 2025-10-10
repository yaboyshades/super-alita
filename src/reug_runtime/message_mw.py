from __future__ import annotations

"""Lightweight message middleware registry for pre-dispatch transforms.

This module allows registering simple functions that take a raw user message
and a small context object and return a possibly transformed message plus a
metadata dict. It is intentionally minimal to avoid coupling to the rest of
the runtime.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class MessageContext:
    session_id: str


MiddlewareFn = Callable[[str, MessageContext], tuple[str, dict[str, str]]]

_MIDDLEWARE: list[MiddlewareFn] = []


def register(fn: MiddlewareFn) -> None:
    """Register a message middleware function."""

    _MIDDLEWARE.append(fn)


def clear() -> None:
    """Clear registered middleware functions (useful in tests)."""

    _MIDDLEWARE.clear()


def apply_all(
    message: str, ctx: MessageContext
) -> tuple[str, list[dict[str, str]]]:
    """Run all registered middleware over the message in order.

    Returns the final message and a list of step metadata dicts.
    """

    meta: list[dict[str, str]] = []
    out = message
    for fn in _MIDDLEWARE:
        try:
            out, m = fn(out, ctx)
            if m:
                meta.append(m)
        except (
            Exception
        ) as e:  # keep robust; a faulty middleware should not break requests
            meta.append(
                {"step": getattr(fn, "__name__", "unknown"), "error": str(e)}
            )
    return out, meta
