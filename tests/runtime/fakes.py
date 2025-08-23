import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any


class FakeEventBus:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit(self, event: dict[str, Any]) -> None:
        # store a shallow copy so tests can assert
        self.events.append(dict(event))


class FakeAbilityRegistry:
    """Knows one tool: 'echo' which returns its args as result."""

    def __init__(self) -> None:
        self._known = {"echo"}
        self._calls: list[dict[str, Any]] = []

    def get_available_tools_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "tool_id": "echo",
                "description": "Echo back the provided payload",
                "input_schema": {
                    "type": "object",
                    "properties": {"payload": {"type": "string"}},
                },
                "output_schema": {
                    "type": "object",
                    "properties": {"echo": {"type": "string"}},
                },
            }
        ]

    def knows(self, tool_name: str) -> bool:
        return tool_name in self._known

    def validate_args(self, tool_name: str, args: dict[str, Any]) -> bool:
        if tool_name == "echo" and isinstance(args.get("payload"), str):
            return True
        # allow dynamic registration path to run in router (it will call register)
        return False

    async def health_check(self, contract: dict[str, Any]) -> bool:
        return True

    async def register(self, contract: dict[str, Any]) -> None:
        self._known.add(contract["tool_id"])

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self._calls.append({"tool": tool_name, "args": dict(args)})
        if tool_name == "echo":
            return {"echo": args["payload"]}
        # default fallback for dynamically-added tools
        return {"ok": True, "args": args}


class FakeKG:
    def __init__(self) -> None:
        self.atoms: list[dict[str, Any]] = []
        self.bonds: list[dict[str, Any]] = []

    async def retrieve_relevant_context(self, user_message: str) -> str:
        return f"(kg_ctx for: {user_message[:32]}...)"

    async def get_goal_for_session(self, session_id: str) -> dict[str, Any]:
        return {
            "id": f"goal_{session_id}",
            "description": f"answer user in session {session_id}",
        }

    async def create_atom(self, atom_type: str, content: Any) -> dict[str, Any]:
        atom = {
            "id": f"atom_{len(self.atoms)}",
            "type": atom_type,
            "content": content,
        }
        self.atoms.append(atom)
        return atom

    async def create_bond(self, bond_type: str, source_atom_id: str, target_atom_id: str) -> None:
        self.bonds.append({"type": bond_type, "src": source_atom_id, "tgt": target_atom_id})


class FakeLLM:
    """Two-phase stream for deterministic tests."""

    def __init__(self) -> None:
        self._phase = 1

    async def stream_chat(
        self, messages: list[dict[str, str]], timeout: float
    ) -> AsyncGenerator[dict[str, str], None]:
        # detect if a tool_result was injected
        if any(m["role"] == "assistant" and "<tool_result" in m["content"] for m in messages):
            self._phase = 2
        if self._phase == 1:
            # yield a bit of prose and a well-formed tool_call block
            yield {"content": "Thinking... "}
            call = '<tool_call>{"tool":"echo","args":{"payload":"hi"}}</tool_call>'
            # stream in small pieces to exercise parser
            for piece in ["<tool", "_call>", call[len("<tool_call>") :]]:
                await asyncio.sleep(0)
                yield {"content": piece}
        else:
            await asyncio.sleep(0)
            final = json.dumps({"content": "done: hi", "citations": []})
            yield {"content": f"<final_answer>{final}</final_answer>"}
