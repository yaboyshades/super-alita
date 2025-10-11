import asyncio
from typing import Any

from src.core.global_workspace import AttentionLevel
from src.core.neural_atom import NeuralStore
from src.core.schemas import MemoryRequest, MemoryType, WorkingMemoryUpdate
from src.plugins.memory_manager_plugin_unified import MemoryManagerPlugin


class DummyWorkspace:
    def __init__(self):
        self.subscribers: dict[str, Any] = {}
        self.updates: list[dict[str, Any]] = []

    def subscribe(self, subscriber_id: str, callback):
        self.subscribers[subscriber_id] = callback

    async def update(
        self,
        *,
        data,
        source,
        attention_level: AttentionLevel,
        broadcast: bool = True,
    ):
        self.updates.append(
            {
                "data": data,
                "source": source,
                "attention_level": attention_level,
                "broadcast": broadcast,
            }
        )


def test_memory_manager_working_memory_flow():
    workspace = DummyWorkspace()
    store = NeuralStore()
    plugin = MemoryManagerPlugin()

    async def run_flow():
        await plugin.setup(workspace, store, {"working_memory_capacity": 3})

        update = WorkingMemoryUpdate(
            memory_id="test-key",
            content={"message": "hello"},
            operation="update",
        )
        await plugin._update_working_memory(update)

        assert plugin.working_memory["test-key"] == {"message": "hello"}
        assert plugin.memory_stats["working_memory_size"] == 1

        request = MemoryRequest(
            operation="recall",
            session_id="session-1",
            query="test-key",
            metadata={"memory_type": MemoryType.WORKING.value, "request_id": "req-1"},
        )

        await plugin._handle_memory_request(request)

        assert plugin.memory_stats["total_queries"] == 1
        assert plugin.memory_stats["successful_retrievals"] == 1
        assert workspace.updates, "workspace should receive a memory result"

        payload = workspace.updates[-1]["data"]
        assert payload["type"] == "memory_result"
        assert payload["metadata"]["request_id"] == "req-1"
        assert payload["content"]["value"] == {"message": "hello"}

        await plugin._handle_memory_request(request)

        assert plugin.memory_stats["total_queries"] == 2
        assert plugin.memory_stats["successful_retrievals"] == 2
        follow_up_payload = workspace.updates[-1]["data"]
        assert follow_up_payload["metadata"]["success"] is True

    asyncio.run(run_flow())
