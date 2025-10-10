import hashlib
from typing import Any
from unittest.mock import patch

from src.core.context_builder import ContextAssembler


def build_assembler(**kwargs: dict[str, Any]) -> ContextAssembler:
    return ContextAssembler(
        user_input=kwargs.get("user_input", ""),
        recent_events=kwargs.get("recent_events", []),
        memory_hits=kwargs.get("memory_hits", []),
        active_goals=kwargs.get("active_goals", []),
        tool_inventory=kwargs.get("tool_inventory", []),
        extras=kwargs.get("extras", {}),
    )


def test_extraction_and_chat_signals_pass_through():
    hits = [
        {"atom_id": "a1", "score": 0.5, "truncated_content_hash": "deadbeef"}
    ]
    extras = {"chat_signals": {"attention": True}}
    assembler = build_assembler(
        user_input="hello",
        recent_events=[{"type": "event"}],
        memory_hits=hits,
        active_goals=["g1"],
        tool_inventory=[{"name": "tool"}],
        extras=extras,
    )

    with (
        patch("src.core.context_builder.get_session_id", return_value="sess"),
        patch(
            "src.core.context_builder.get_correlation_id", return_value="corr"
        ),
    ):
        ctx = assembler.build_for_decision()

    assert ctx["session_id"] == "sess"
    assert ctx["correlation_id"] == "corr"
    assert ctx["user_input"] == "hello"
    assert ctx["recent_events"] == [{"type": "event"}]
    assert ctx["active_goals"] == ["g1"]
    assert ctx["tool_inventory"] == [{"name": "tool"}]
    assert ctx["extras"] == extras


def test_scrubbing_normalizes_memory_hits():
    raw_hits = [
        {
            "atom_id": "a1",
            "score": "0.85",
            "truncated_content_hash": "abcd1234",
            "extra": "ignore",
        }
    ]

    cleaned = ContextAssembler._normalize_memory_hits(raw_hits)

    assert cleaned == [
        {"atom_id": "a1", "score": 0.85, "truncated_content_hash": "abcd1234"}
    ]


def test_hash_stability_across_normalization():
    content = "some content to hash"
    truncated = hashlib.sha256(content.encode()).hexdigest()[:8]
    hit = {"atom_id": "a1", "score": 0.7, "truncated_content_hash": truncated}

    first = ContextAssembler._normalize_memory_hits([hit])
    second = ContextAssembler._normalize_memory_hits(first)

    assert first == second
    assert first[0]["truncated_content_hash"] == truncated
