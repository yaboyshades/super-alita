# MCP Server & Tool Registry

## Registry

- `register(name, fn)`
- `register_from_code(name, code: str)` writes to `~/.alita_tools/<name>.py` and registers the function
- `invoke(name, args: dict) -> Awaitable[Any]`
- `list_tools() -> List[str]`

## MCP Methods (JSON-RPC)

- `initialize` -> `{tools: [...] }`
- `list_tools` -> `[names...]`
- `invoke` -> tool result (tool must return dict; atom tools return `{atoms,bonds}`).
- `register_tool` -> `{"ok": true}` on success.
- `shutdown`
