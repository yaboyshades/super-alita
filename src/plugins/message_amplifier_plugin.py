from __future__ import annotations

"""Heuristic message amplifier middleware.

Default-safe optimizer that normalizes whitespace, supports a `noopt:`
escape hatch, and emits simple metadata about the transformation. More
advanced enrichment (e.g., context lookup or model rewriting) can be added
behind config flags later without changing the public interface.
"""

import re

from src.reug_runtime.message_mw import MessageContext, register

_WS_RE = re.compile(r"\s+", re.MULTILINE)


def _normalize(msg: str) -> str:
    msg = msg.strip()
    msg = _WS_RE.sub(" ", msg)
    return msg


def amplify_message(message: str, ctx: MessageContext) -> tuple[str, dict[str, str]]:
    if message.lower().startswith("noopt:"):
        # Explicit user bypass
        return message[len("noopt:") :].lstrip(), {"step": "amplify", "bypass": "true"}

    original_len = len(message)
    normalized = _normalize(message)
    # Optionally, we could add a short structured header when message is long
    # For now, keep changes minimal to avoid surprising users/tests.

    meta = {
        "step": "amplify",
        "bypass": "false",
        "len_in": str(original_len),
        "len_out": str(len(normalized)),
    }
    return normalized, meta


# Register on import so router can just import the module when enabled.
register(amplify_message)  # type: ignore[arg-type]

