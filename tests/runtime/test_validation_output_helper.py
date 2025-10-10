"""Tests for validation message normalisation utilities."""

from __future__ import annotations

import json

from src.utils.validation_output import (
    build_payload,
    normalise_messages,
    to_json,
)


def test_normalise_messages_parses_status_prefixes() -> None:
    entries = [
        "error::Missing required field",  # explicit status
        " warning ::Leading whitespace trimmed",  # whitespace handling
        "Plain message without status",  # inherits default status
        "",  # skipped entirely
    ]

    records = normalise_messages(entries)
    assert [r["status"] for r in records] == ["error", "warning", "info"]
    assert [r["index"] for r in records] == [1, 2, 3]
    assert records[0]["message"] == "Missing required field"
    assert records[1]["message"] == "Leading whitespace trimmed"
    assert records[2]["message"] == "Plain message without status"


def test_build_payload_and_to_json_round_trip() -> None:
    entries = ["info::Router healthy", "warning::Schema enforcement disabled"]

    payload = build_payload(entries)
    assert payload["count"] == 2
    assert len(payload["messages"]) == 2

    json_output = to_json(entries)
    as_dict = json.loads(json_output)
    assert as_dict == payload
