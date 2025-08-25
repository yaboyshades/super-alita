"""Tests for knowledge graph adapters and router provenance bonds."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from reug_runtime.kg import FileKG, InMemoryKG
from reug_runtime.router import execute_turn
from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeLLM


@pytest.mark.asyncio
async def test_filekg_persistence(tmp_path):
    path = tmp_path / "kg.json"
    kg1 = FileKG(path)
    atom = await kg1.create_atom("NOTE", {"msg": "hi"})
    await kg1.create_bond("RELATES_TO", atom["id"], "goal_test")

    # Re-open and ensure data persisted
    kg2 = FileKG(path)
    assert await kg2.get_atom(atom["id"]) == atom
    bonds = await kg2.get_bonds(atom["id"])
    assert any(b["tgt"] == "goal_test" for b in bonds)


@pytest.mark.asyncio
async def test_used_tool_bond_creation():
    kg = InMemoryKG()
    bus = FakeEventBus()
    registry = FakeAbilityRegistry()
    llm = FakeLLM()

    gen = execute_turn("hello", "s1", bus, registry, kg, llm)
    async for _ in gen:
        pass

    assert any(b["type"] == "USED_TOOL" for b in kg.bonds)


class FailingRegistry(FakeAbilityRegistry):
    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        raise RuntimeError("boom")


class ErrorAwareLLM(FakeLLM):
    async def stream_chat(
        self, messages: list[dict[str, str]], timeout: float
    ) -> AsyncGenerator[dict[str, str], None]:  # type: ignore[override]
        # finalize after seeing a tool_error
        has_result = any(
            m["role"] == "assistant"
            and ("<tool_result" in m["content"] or "<tool_error" in m["content"])
            for m in messages
        )
        if not has_result:
            yield {"content": "Thinking... "}
            yield {
                "content": '<tool_call>{"tool":"echo","args":{"payload":"hi"}}</tool_call>'
            }
        else:
            yield {
                "content": '<final_answer>{"content":"done","citations":[]}</final_answer>'
            }


@pytest.mark.asyncio
async def test_failed_tool_bond_creation():
    kg = InMemoryKG()
    bus = FakeEventBus()
    registry = FailingRegistry()
    llm = ErrorAwareLLM()

    gen = execute_turn("hello", "s1", bus, registry, kg, llm)
    async for _ in gen:
        pass

    assert any(b["type"] == "FAILED_TOOL" for b in kg.bonds)
