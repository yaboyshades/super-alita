"""Helpers for interacting with MCP-enabled models."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from typing import Any

import httpx


async def sample_vscode(
    *,
    model: str,
    messages: Iterable[Mapping[str, str]],
    base_url: str,
    http_client: httpx.AsyncClient | None = None,
) -> str:
    """Sample text from a `vscode.lm` model via MCP.

    Parameters
    ----------
    model:
        Identifier of the model to invoke.
    messages:
        Conversation history in MCP format.
    base_url:
        Endpoint of the MCP sampling service.
    http_client:
        Optional pre-configured HTTP client.
    """

    client = http_client or httpx.AsyncClient()
    request = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "vscode.lm/sample",
        "params": {"model": model, "messages": list(messages)},
    }
    response = await client.post(base_url, json=request)
    response.raise_for_status()
    data: Any = response.json()
    # MCP responses place the payload under `result`.
    return data["result"]["text"]
