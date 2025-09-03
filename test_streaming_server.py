#!/usr/bin/env python3
"""
Simple streaming test server to validate connection handling.
"""

import asyncio
import json
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI(title="Streaming Test Server")


@app.get("/test/stream")
async def test_stream():
    """Test streaming endpoint."""

    async def generate_stream():
        for i in range(5):
            data = {"chunk": i, "timestamp": time.time(), "message": f"Test chunk {i}"}
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)

        # Final chunk
        final_data = {"finished": True, "total_chunks": 5}
        yield f"data: {json.dumps(final_data)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/test/health")
async def test_health():
    """Simple health check."""
    return {"status": "ok", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="127.0.0.1", port=8081, timeout_keep_alive=120, access_log=True
    )
