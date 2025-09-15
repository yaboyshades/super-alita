"""Tests for helper utilities in :mod:`src.reug_runtime.router`."""

import json

from src.reug_runtime.router import parse_tool_calls


def test_parse_tool_calls_generates_dict_payload() -> None:
    text = '<tool_call>{"tool":"search","args":{"query":"plan"}}</tool_call>'

    calls = parse_tool_calls(text)

    assert len(calls) == 1
    call = calls[0]
    assert call["name"] == "search"
    fn = call["function"]
    assert isinstance(fn, dict)
    assert fn["name"] == "search"
    assert json.loads(fn["arguments"]) == {"query": "plan"}
    assert isinstance(call["id"], str) and call["id"]


def test_parse_tool_calls_ignores_invalid_payload() -> None:
    text = '<tool_call>{"tool":"search","args":[]}</tool_call>'

    calls = parse_tool_calls(text)

    assert calls == []
