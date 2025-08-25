from mcp_server_wrapper import MCPBridge


def test_mcp_bridge_capability_shape():
    bridge = MCPBridge()
    cap = bridge._convert_mcp_to_capability(
        {"name": "echo", "description": "Echo", "inputSchema": {"type": "object"}}
    )
    assert cap["type"] == "mcp_tool"
    assert callable(bridge._create_mcp_executor("echo"))
