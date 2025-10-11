"""Utilities for sanitising sensitive content prior to storage."""
from __future__ import annotations

import re
from typing import Tuple

# Simple regular expressions covering common secrets/PII like API keys or emails.
SECRET_PATTERNS = [
    re.compile(r"(api(?:[_-]|\s)?key\s*(?:[:=]|is)\s*)([A-Za-z0-9-]{6,})", re.IGNORECASE),
    re.compile(r"(token\s*(?:[:=]|is)\s*)([A-Za-z0-9-]{6,})", re.IGNORECASE),
    re.compile(r"(password\s*(?:[:=]|is)\s*)(\S+)", re.IGNORECASE),
]

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"\b\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b")


def sanitize_text(text: str) -> Tuple[str, bool]:
    """Redact sensitive tokens from *text*.

    Returns the sanitised text and a boolean indicating whether redaction occurred.
    """

    was_redacted = False
    clean_text = text

    def _replace(match: re.Match[str]) -> str:
        nonlocal was_redacted
        was_redacted = True
        prefix = match.group(1)
        return f"{prefix}***REDACTED***"

    for pattern in SECRET_PATTERNS:
        clean_text = pattern.sub(_replace, clean_text)

    if EMAIL_PATTERN.search(clean_text):
        was_redacted = True
        clean_text = EMAIL_PATTERN.sub("***REDACTED_EMAIL***", clean_text)

    if PHONE_PATTERN.search(clean_text):
        was_redacted = True
        clean_text = PHONE_PATTERN.sub("***REDACTED_PHONE***", clean_text)

    return clean_text, was_redacted


def should_quarantine(text: str) -> bool:
    """Return True if the message should be quarantined rather than stored."""

    lowered = text.lower()
    if "do not store" in lowered or "confidential" in lowered:
        return True
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return True
    return False
