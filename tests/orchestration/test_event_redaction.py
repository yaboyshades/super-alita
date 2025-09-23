from __future__ import annotations

import hashlib
import json

import pytest

from src.orchestration.event_sanitizer import (
    compute_args_hash,
    sanitize_config_snapshot,
    truncate_preview,
)


def test_config_snapshot_redacts_sensitive_fields():
    original = {
        "API_KEY": "super-secret",
        "token": "abc123",
        "safe": True,
        "nested": {"password": "hidden", "keep": "value"},
    }
    sanitized = sanitize_config_snapshot(original)

    # Original dict should remain untouched
    assert original["API_KEY"] == "super-secret"

    assert sanitized["API_KEY"] == "<redacted>"
    assert sanitized["token"] == "<redacted>"
    assert sanitized["safe"] is True
    assert sanitized["nested"]["password"] == "<redacted>"
    assert sanitized["nested"]["keep"] == "value"


@pytest.mark.parametrize(
    "payload,expected",
    [
        ("short text", "short text"),
        ("x" * 500, "x" * 197 + "..."),
    ],
)
def test_truncate_preview_limits_length(payload: str, expected: str):
    assert truncate_preview(payload) == expected


def test_args_hash_ignores_sensitive_keys():
    args = {
        "api_key": "should-not-leak",
        "token": "secret",
        "query": "hello",
        "nested": {"password": "hidden", "arg": 3},
    }
    sanitized = {
        "nested": {"arg": 3},
        "query": "hello",
    }
    expected_hash = hashlib.sha256(
        json.dumps(sanitized, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    assert compute_args_hash(args) == expected_hash
