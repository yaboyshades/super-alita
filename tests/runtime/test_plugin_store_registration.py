import numpy as np
import pytest

from src.plugins.brainstorm_plugin import BrainstormPlugin, DynamicAtom as BrainstormAtom
from src.plugins.compose_plugin import ComposePlugin, DynamicAtom as ComposeAtom


class MockStore:
    def __init__(self):
        self.registered = []

    async def embed_text(self, texts: list[str]):
        return [np.zeros(1024, dtype=np.float32) for _ in texts]

    def register(self, atom):
        self.registered.append(atom)


@pytest.mark.asyncio
async def test_brainstorm_registers_memory_atom():
    store = MockStore()
    plugin = BrainstormPlugin()
    plugin.store = store
    atom = BrainstormAtom(tool="t", code="print('hi')", description="d")
    await plugin._store_atom("mem1", atom)
    assert store.registered and store.registered[0].key == "mem1"


@pytest.mark.asyncio
async def test_compose_registers_memory_atom():
    store = MockStore()
    plugin = ComposePlugin()
    plugin.store = store
    atom = ComposeAtom(tool="t", code="print('hi')", description="d")
    await plugin._store_atom("mem2", atom)
    assert store.registered and store.registered[0].key == "mem2"
