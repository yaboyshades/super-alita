# Super Alita - Full Integration Guide (Revised)

## Quick Start

1. **Setup Environment**
```bash
git clone https://github.com/yaboyshades/super-alita.git
cd super-alita
make dev-setup
```

2. **Configure API Keys**
Edit `.env` with your actual API keys for search providers and LLMs.

3. **Start ACP Server**
```bash
make run-acp
# Server runs on http://localhost:8000
```

4. **Test the Integration**
```bash
# Run client examples
python -m src.acp_app.client_examples

# Run tests
make test
```

## MCP Integration (for Claude Desktop)

1. **Install MCP Adapter**
```bash
pip install acp-mcp
```

2. **Configure Claude Desktop**
Add to your Claude Desktop config:
```json
{
  "mcpServers": {
    "super-alita": {
      "command": "uvx",
      "args": ["acp-mcp", "http://localhost:8000"]
    }
  }
}
```

3. **Use in Claude**
- Start your ACP server: `make run-acp`
- Restart Claude Desktop
- Your agents (echo, classify, router, search) are now available as MCP tools

## Available Agents

### Search Agent
AI-powered multi-modal search with reasoning and citations.

```python
messages = [Message(parts=[
    MessagePart(text="explain quantum computing", metadata={"mode": "academic"})
])]
response = await client.run_sync("search", messages)
```

Modes: `web`, `academic`, `news`, `video`, `images`, `reddit`, `shopping`, `wolfram`

### Echo Agent
Simple round-trip testing.

### Classify Agent
Returns structured classification with confidence.

### Router Agent
Chains multiple agents with streaming output.

## Docker Deployment

```bash
# Start all services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

## Testing

```bash
# All tests
make test

# Specific test suites
pytest tests/plugins/test_perplexica_search_plugin.py -v
pytest tests/acp/test_acp_agents.py -v
```

## Production Checklist

- [ ] Set real API keys in `.env`
- [ ] Enable auth: `ACP_REQUIRE_AUTH=true`
- [ ] Configure rate limits in `src/config/perplexica.yaml`
- [ ] Set up Redis for distributed caching
- [ ] Deploy behind reverse proxy with SSL
- [ ] Set up monitoring (Prometheus metrics at `/metrics`)
- [ ] Configure log aggregation
- [ ] Set appropriate resource limits in Docker

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Claude    │────▶│ MCP Adapter │────▶│ ACP Server  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
              ┌─────▼─────┐           ┌───────▼───────┐         ┌────────▼────────┐
              │   Agents  │           │ Perplexica    │         │ Tool Registry   │
              └───────────┘           │ Search Tool   │         └─────────────────┘
                                      └───────────────┘
```

## Troubleshooting

**No agents showing in MCP:**
- Verify ACP server is running: `curl http://localhost:8000/health` (or your MCP client’s tool list)
- Check MCP adapter can reach server
- Restart Claude Desktop after config changes

**Search returns empty results:**
- Check API keys in `.env`
- Verify rate limits aren't exceeded
- Check `docker logs` if using Docker

**Connection refused:**
- Ensure port 8000 is not in use
- Check firewall settings
- Use `0.0.0.0` instead of `localhost` in Docker
