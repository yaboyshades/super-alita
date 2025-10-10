from src.orchestrator.discovery import ComponentRegistry


def test_registry_lists_all_integrations() -> None:
    registry = ComponentRegistry()

    components = registry.list_components()

    assert "codex" in components
    assert "super_alita" in components
    assert "cma" in components
