#
# /mcp/fastmcp_server.py
#
# Description: This server implements the Model Context Protocol (MCP) with `search` and `fetch`
# capabilities, designed to work with ChatGPT connectors and deep research features.
# It connects to an OpenAI Vector Store as its data source.
#

from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastmcp import FastMCP
from openai import OpenAI

# --- Logging Configuration -----------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# --- Server Configuration ------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
VECTOR_STORE_ID = os.environ.get("VECTOR_STORE_ID", "")
ALLOW_NO_AUTH = os.environ.get("MCP_ALLOW_NO_AUTH", "false").lower() == "true"
ALLOWLIST = {s.strip() for s in os.environ.get("MCP_ALLOWLIST", "").split(",") if s.strip()}
HOST = os.environ.get("MCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("MCP_PORT", "8000"))
TRANSPORT = os.environ.get("MCP_TRANSPORT", "sse")  # SSE is recommended for remote servers
SERVER_NAME = os.environ.get("MCP_SERVER_NAME", "Sample MCP Server")

# --- OpenAI Client Initialization ----------------------------------------------
_openai_client: OpenAI | None = None
def get_openai_client() -> OpenAI:
    """Initializes and returns a singleton OpenAI client."""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

# --- Authentication Hook -------------------------------------------------------
def _check_auth(headers: dict[str, str]) -> None:
    """
    Checks for a valid Bearer token in the request headers if authentication is enabled.
    Raises PermissionError if authentication fails.
    """
    if ALLOW_NO_AUTH:
        return
    token = headers.get("authorization", "").replace("Bearer ", "").strip()
    if not token:
        raise PermissionError("Missing Bearer token in authorization header.")
    if ALLOWLIST and token not in ALLOWLIST:
        raise PermissionError("Provided token is not in the allowlist.")

# --- Server Instructions -------------------------------------------------------
server_instructions = """
This MCP server provides search and document retrieval capabilities for chat and deep research.
Use the `search` tool to find relevant documents, then `fetch` to retrieve full text for citation.
"""

def create_server() -> FastMCP:
    """Creates and configures the FastMCP server with search and fetch tools."""
    mcp = FastMCP(name=SERVER_NAME, instructions=server_instructions)

    @mcp.tool()
    async def search(query: str, __headers: dict[str, str] | None = None) -> dict[str, list[dict[str, Any]]]:
        """
        Search for documents using OpenAI Vector Store search.
        
        Args:
            query: Natural language search query string.
            __headers: Request headers for authentication.

        Returns:
            A dictionary with a 'results' key containing a list of matching documents.
            Each result includes id, title, text snippet, and a URL.
        """
        _check_auth(__headers or {})
        t0 = time.time()
        logger.info(f"Received search query: '{query}'")

        if not (query or "").strip():
            return {"results": []}

        client = get_openai_client()
        response = client.vector_stores.search(vector_store_id=VECTOR_STORE_ID, query=query)

        results: list[dict[str, Any]] = []
        if hasattr(response, "data"):
            for i, item in enumerate(response.data):
                file_id = getattr(item, "file_id", f"unknown_{i}")
                filename = getattr(item, "filename", f"Document {i+1}")
                content_list = getattr(item, "content", [])
                
                text_content = ""
                if content_list:
                    first_content = content_list[0]
                    if hasattr(first_content, 'text'):
                        text_content = first_content.text
                    elif isinstance(first_content, dict):
                        text_content = first_content.get('text', '')

                snippet = (text_content[:200] + "...") if len(text_content) > 200 else text_content
                
                results.append({
                    "id": file_id,
                    "title": filename,
                    "text": snippet or "No content available",
                    "url": f"https://platform.openai.com/storage/files/{file_id}",
                })

        latency_ms = int((time.time() - t0) * 1000)
        logger.info(f"Search completed in {latency_ms}ms, found {len(results)} results.")
        return {"results": results}

    @mcp.tool()
    async def fetch(id: str, __headers: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Retrieve complete document content by its unique ID.

        Args:
            id: The file ID from a search result (e.g., 'file-xxx').
            __headers: Request headers for authentication.

        Returns:
            A dictionary containing the full document details: id, title, text, url, and metadata.
        """
        _check_auth(__headers or {})
        t0 = time.time()
        logger.info(f"Fetching document with ID: {id}")

        if not (id or "").strip():
            raise ValueError("Document ID is required.")

        client = get_openai_client()
        
        content_response = client.vector_stores.files.content(vector_store_id=VECTOR_STORE_ID, file_id=id)
        file_info = client.vector_stores.files.retrieve(vector_store_id=VECTOR_STORE_ID, file_id=id)
        
        content_parts: list[str] = []
        if hasattr(content_response, "data"):
            for chunk in content_response.data:
                if hasattr(chunk, "text") and chunk.text:
                    content_parts.append(chunk.text)
        
        full_text = "\n".join(content_parts) if content_parts else "No content available"
        title = getattr(file_info, "filename", f"Document {id}")

        result: dict[str, Any] = {
            "id": id,
            "title": title,
            "text": full_text,
            "url": f"https://platform.openai.com/storage/files/{id}",
            "metadata": None,
        }
        
        if hasattr(file_info, "attributes") and file_info.attributes:
            result["metadata"] = file_info.attributes

        latency_ms = int((time.time() - t0) * 1000)
        logger.info(f"Fetch for ID {id} completed in {latency_ms}ms.")
        return result

    return mcp

def main():
    """Main function to configure and start the MCP server."""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not found.")
        raise ValueError("OpenAI API key is required.")
    if not VECTOR_STORE_ID:
        logger.warning("VECTOR_STORE_ID environment variable not set.")

    server = create_server()
    logger.info(f"Starting MCP server '{SERVER_NAME}' on {HOST}:{PORT} via {TRANSPORT} transport.")
    
    try:
        server.run(transport=TRANSPORT, host=HOST, port=PORT)
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()