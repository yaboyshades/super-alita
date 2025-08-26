#!/usr/bin/env python3
"""Simple test server to validate DeepCode HTTP endpoint"""

from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="DeepCode Test Server")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "deepcode-test"}


@app.post("/deepcode/request")
async def deepcode_request(request: Request):
    """Test DeepCode request endpoint"""
    body = await request.json()

    print(f"📥 DeepCode request received: {body}")

    # Simulate processing
    task_kind = body.get("task_kind", "unknown")
    repo_path = body.get("repo_path", str(Path.cwd()))
    conversation_id = body.get("conversation_id", "test")

    response = {
        "status": "accepted",
        "request": {
            "task_kind": task_kind,
            "repo_path": repo_path,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
        },
        "message": f"DeepCode {task_kind} request queued for processing",
    }

    print(f"📤 DeepCode response: {response}")

    # In a real implementation, this would trigger the plugin orchestrator
    # For now, we just return acceptance
    return JSONResponse(content=response, status_code=202)


if __name__ == "__main__":
    print("🚀 Starting DeepCode Test Server on http://127.0.0.1:8080")
    print("📋 Available endpoints:")
    print("  - GET  /health")
    print("  - POST /deepcode/request")
    print("")
    print("🔗 Test with VS Code commands:")
    print("  - Alita: DeepCode — Analyze Workspace")
    print("  - Alita: DeepCode — Generate From Prompt")
    print("")
    print("⚙️  Make sure VS Code setting 'alita.runtime.host' = 'http://127.0.0.1:8080'")

    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
