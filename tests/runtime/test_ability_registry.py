import importlib
from pathlib import Path

import pytest


def _registry_class(tmp_path, monkeypatch):
    import reug_runtime.config as config
    old_settings = config.SETTINGS
    monkeypatch.setenv("REUG_TOOL_REGISTRY_DIR", str(tmp_path))
    importlib.reload(config)
    import reug_runtime.ability_registry as ability_registry
    importlib.reload(ability_registry)
    AbilityRegistry = ability_registry.AbilityRegistry
    config.SETTINGS = old_settings
    return AbilityRegistry


@pytest.mark.asyncio
async def test_register_and_persist(tmp_path, monkeypatch):
    AbilityRegistry = _registry_class(tmp_path, monkeypatch)
    reg = AbilityRegistry()
    contract = {
        "tool_id": "hello",
        "description": "Return greeting",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        "output_schema": {"type": "object"},
    }
    await reg.register(contract)
    assert reg.knows("hello")
    assert reg.validate_args("hello", {"name": "world"})
    assert not reg.validate_args("hello", {"name": 1})
    file = Path(tmp_path) / "hello.json"
    assert file.exists()

    reg2 = AbilityRegistry()
    assert reg2.knows("hello")


@pytest.mark.asyncio
async def test_register_invalid_schema(tmp_path, monkeypatch):
    AbilityRegistry = _registry_class(tmp_path, monkeypatch)
    reg = AbilityRegistry()
    bad_contract = {
        "tool_id": "bad",
        "description": "Bad schema",
        "input_schema": {"type": "notatype"},
        "output_schema": {"type": "object"},
    }
    with pytest.raises(ValueError):
        await reg.register(bad_contract)
