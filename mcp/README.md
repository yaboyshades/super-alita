#
# /mcp/README.md
#
# Description: Runbook for configuring and running the MCP server.
#

# MCP Server (Search + Fetch) — Runbook

This server exposes **search** and **fetch** tools compliant with the Model Context Protocol (MCP), designed for use with ChatGPT Connectors & Deep Research. It reads from a configured **OpenAI Vector Store** to make your private data available to AI models.

## 1. Configure Environment

Before running the server, you need to set the following environment variables. You can place them in a `.env` file for automatic loading if you use a library like `python-dotenv`.

```bash
# Required: Your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Required: The ID of the OpenAI Vector Store to use
export VECTOR_STORE_ID="vs_..."

# --- Optional Settings ---

# Server network configuration
export MCP_HOST="0.0.0.0"
export MCP_PORT="8000"

# Set to "true" to disable token authentication (useful for local development)
export MCP_ALLOW_NO_AUTH="false"

# Comma-separated list of allowed Bearer tokens if auth is enabled
export MCP_ALLOWLIST="secret-token-1,secret-token-2"
```

### Vector Store Setup
If you haven't already, upload your documents to an OpenAI Vector Store via the [OpenAI Dashboard](https://platform.openai.com/storage/vector_stores) or the API. Note the `Vector store ID` to use for `VECTOR_STORE_ID`.

## 2. Run the Server

First, install the required dependencies:

```bash
pip install "fastmcp>=0.2.2" "openai>=1.40"
```

Then, run the server from your terminal:

```bash
python -m mcp.fastmcp_server
```

The server will start listening on `0.0.0.0:8000` (or as configured) using the SSE transport.

## 3. Connect in ChatGPT or via API

You can now connect this server to ChatGPT or use it in the Deep Research API.

### In ChatGPT
Go to **Settings → Connectors → Add remote MCP** and configure it:
- **Server URL (SSE):** `http://<your-server-ip>:8000/sse/` (If running locally, you may need a tool like `ngrok` to expose it to the internet).
- **Allowed tools:** `["search", "fetch"]`
- **Approval:** `never`

### Via Deep Research API
Use the server URL in the `tools` array of your API request:

```json
{
  "tools": [
    {
      "type": "mcp",
      "server_url": "http://<your-server-ip>:8000/sse/",
      "allowed_tools": ["search", "fetch"],
      "require_approval": "never"
    }
  ]
}
```

## 4. Risks & Safety

- This server exposes data from your private vector store. Ensure it is deployed in a secure environment.
- Use the built-in Bearer token authentication (`MCP_ALLOWLIST`) for production use. For more robust security, consider placing the server behind an authentication gateway that handles OAuth 2.0.
- The server sees all search queries sent to it. Do not send sensitive data in queries if the server's security posture is not sufficient.

## 5. Testing

To run the unit tests for the server, install `pytest` and run it from the root of the project:

```bash
pip install pytest pytest-asyncio
pytest mcp/tests/
```