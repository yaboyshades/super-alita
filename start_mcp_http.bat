@echo off
REM Start Super Alita MCP Server via HTTP/SSE
set MCP_TRANSPORT=sse
set MCP_PORT=8001
set MCP_HOST=0.0.0.0
set MCP_ALLOW_NO_AUTH=true
set MCP_SERVER_NAME=Super Alita MCP Server

echo Starting Super Alita MCP Server on http://localhost:8001
echo Press Ctrl+C to stop

python d:\Coding_Projects\super-alita-clean\mcp_server_wrapper.py