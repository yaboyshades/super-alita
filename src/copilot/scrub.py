"""Prompt scrubbing utilities for Copilot integration.

This module provides lightweight helpers to remove common secret patterns
from text and to clamp a string to a rough token budget.  Functions are kept
pure to simplify unit testing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from re import Pattern

# Common patterns that may reveal secrets.  These expressions intentionally keep
# the implementation simple – they are not a substitute for full secret
# scanning but provide a basic safeguard before prompts are sent to a model.
_SECRET_PATTERNS: tuple[Pattern[str], ...] = (
    # Generic API key or token assignments, e.g. ``API_KEY=abc123``
    re.compile(
        r"(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s]+", re.IGNORECASE
    ),
    # PEM formatted private keys
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----",
        re.IGNORECASE,
    ),
)


def scrub_prompt(
    text: str, patterns: Iterable[Pattern[str]] | None = None
) -> str:
    """Remove sensitive substrings from ``text``.

    Each configured pattern is replaced with ``"[REDACTED]"``.  The operation is
    purely functional and returns a new string.

    Parameters
    ----------
    text:
        Input string potentially containing secrets.
    patterns:
        Optional iterable of regular expression patterns to apply.  If ``None``
        the module's default patterns are used.
    """

    pats = tuple(patterns) if patterns is not None else _SECRET_PATTERNS
    scrubbed = text
    for pat in pats:
        scrubbed = pat.sub("[REDACTED]", scrubbed)
    return scrubbed


def clamp_tokens(text: str, max_tokens: int) -> str:
    """Clamp ``text`` to at most ``max_tokens`` whitespace-delimited tokens.

    The function performs a naïve whitespace split which is adequate for
    budgeting in tests and environments where an exact tokenizer is
    unnecessary.
    """

    if max_tokens <= 0:
        return ""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


__all__ = ["scrub_prompt", "clamp_tokens"]
