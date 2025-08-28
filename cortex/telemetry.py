"""
Lightweight telemetry emit hook used by MCPTelemetryBroadcaster.

This module intentionally avoids heavy dependencies. It supports:

- HTTP POST: set env MCP_HTTP_URL to receive JSON payloads.
- File JSONL: set env MCP_EMIT_FILE to append JSON lines for debugging.

Both transports are best-effort and non-fatal on errors. If neither is
configured, `emit` becomes a no-op.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any


def _post_http(url: str, payload: dict[str, Any], timeout: float = 0.8) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Best-effort: ignore response body, swallow errors
    try:
        with urllib.request.urlopen(req, timeout=timeout):  # nosec B310
            pass
    except Exception:
        # Intentionally silent: telemetry should never break the app
        return


def _append_jsonl(path: str, payload: dict[str, Any]) -> None:
    try:
        line = json.dumps(payload, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        return


def emit(payload: dict[str, Any]) -> None:
    """Best-effort emit of telemetry payload.

    Honors the following env vars:
    - MCP_HTTP_URL: HTTP endpoint to POST JSON payloads to
    - MCP_EMIT_FILE: path to append JSONL lines
    """
    # Add a generic timestamp if not present
    if "emitted_at" not in payload:
        payload = {**payload, "emitted_at": time.time()}

    http_url = os.getenv("MCP_HTTP_URL", "").strip()
    sink_file = os.getenv("MCP_EMIT_FILE", "").strip()

    if http_url:
        _post_http(http_url, payload)
    if sink_file:
        _append_jsonl(sink_file, payload)

