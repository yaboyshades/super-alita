from src.core.unified_registry import Capability, UnifiedCapabilityRegistry


def test_unified_registry_collects_tools():
    reg = UnifiedCapabilityRegistry()
    reg.register_capability(
        Capability("fmt", "Format selection", "normal", {"description": "fmt"}),
        "normal",
    )
    reg.register_capability(
        Capability("mcp.echo", "Echo", "mcp", {"description": "mcp echo"}),
        "mcp",
    )
    caps = reg.get_all_capabilities()
    names = sorted([c.name for c in caps])
    assert names == ["fmt", "mcp.echo"]
