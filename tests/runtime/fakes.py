import asyncio
import inspect
import json
from collections.abc import AsyncGenerator
from typing import Any


class FakeEventBus:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit(self, event: dict[str, Any]) -> None:
        # store a shallow copy so tests can assert
        snapshot = dict(event)
        if "type" not in snapshot and "event_type" in snapshot:
            snapshot["type"] = snapshot["event_type"]
        self.events.append(snapshot)


class FakeAbilityRegistry:
    """Knows one tool: 'echo' which returns its args as result."""

    def __init__(self) -> None:
        self._known = {"echo"}
        self._calls: list[dict[str, Any]] = []

        class _EchoTool:
            async def aexecute(self, payload: str) -> dict[str, str]:
                return {"echo": payload}

        self._impls: dict[str, Any] = {"echo": _EchoTool()}

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
        tid = contract["tool_id"]
        self._known.add(tid)
        self._impls.setdefault(tid, lambda **kwargs: {"ok": True, "args": kwargs})

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self._calls.append({"tool": tool_name, "args": dict(args)})
        impl = self._impls.get(tool_name)
        if impl is None:
            return {"ok": True, "args": args}
        fn = getattr(impl, "aexecute", None) or getattr(impl, "run", None) or impl
        sig = inspect.signature(fn)
        params = sig.parameters
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        unexpected = [k for k in args if k not in params] if not accepts_var_kw else []
        if unexpected:
            raise TypeError(
                f"Unexpected argument(s): {', '.join(sorted(unexpected))} for tool '{tool_name}'"
            )
        filtered = {k: v for k, v in args.items() if accepts_var_kw or k in params}
        if inspect.iscoroutinefunction(fn):
            return await fn(**filtered)
        return fn(**filtered)


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

    async def create_bond(
        self, bond_type: str, source_atom_id: str, target_atom_id: str
    ) -> None:
        self.bonds.append(
            {"type": bond_type, "src": source_atom_id, "tgt": target_atom_id}
        )


class FakeLLM:
    """Two-phase stream for deterministic tests."""

    def __init__(self) -> None:
        self._phase = 1

    async def stream_chat(
        self, messages: list[dict[str, str]], timeout: float
    ) -> AsyncGenerator[dict[str, str], None]:
        # detect if a tool_result was injected
        if any(
            m["role"] == "assistant" and "<tool_result" in m["content"]
            for m in messages
        ):
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
