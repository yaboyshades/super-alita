"""Utilities for turning validation messages into structured JSON payloads.

The REUG runtime surfaces validation or health-check messages in a variety of
places (shell scripts, CI jobs, orchestration tools).  Bash is convenient for
running the checks, but JSON is the lingua franca for telemetry and pipeline
consumers.  These helpers keep the transformation small, well-tested, and
reusable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

DEFAULT_STATUS = "info"


def _split_status(entry: str, default_status: str = DEFAULT_STATUS) -> tuple[str, str]:
    """Split a raw ``status::message`` entry.

    Args:
        entry: Raw message captured by a shell script.  Expected format is
            ``"<status>::<message>"`` but the status prefix is optional.
        default_status: Status value applied when ``entry`` has no explicit
            prefix.

    Returns:
        Tuple of ``(status, message)`` with surrounding whitespace trimmed and
        status normalised to lowercase.
    """

    if "::" in entry:
        status_part, message = entry.split("::", 1)
        status = status_part.strip().lower() or default_status
    else:
        status, message = default_status, entry
    return status, message.strip()


def normalise_messages(
    entries: Sequence[str], default_status: str = DEFAULT_STATUS
) -> list[dict[str, str | int]]:
    """Normalise raw validation entries into structured dictionaries.

    Args:
        entries: Ordered collection of raw messages (usually from Bash arrays).
        default_status: Default status assigned to entries without a status
            prefix.

    Returns:
        List of dictionaries in the form ``{"index": n, "status": s, "message": m}``
        with blank entries removed.
    """

    records: list[dict[str, str | int]] = []
    for idx, raw in enumerate(entries, start=1):
        trimmed = raw.strip()
        if not trimmed:
            continue
        status, message = _split_status(trimmed, default_status=default_status)
        records.append({"index": idx, "status": status, "message": message})
    return records


def build_payload(
    entries: Sequence[str], default_status: str = DEFAULT_STATUS
) -> dict[str, Any]:
    """Create a JSON-serialisable payload describing validation messages."""

    records = normalise_messages(entries, default_status=default_status)
    return {"count": len(records), "messages": records}


def to_json(
    entries: Sequence[str],
    *,
    default_status: str = DEFAULT_STATUS,
    indent: int = 2,
) -> str:
    """Render validation messages as JSON."""

    payload = build_payload(entries, default_status=default_status)
    return json.dumps(payload, indent=indent)


__all__ = ["build_payload", "normalise_messages", "to_json"]
