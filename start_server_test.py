#!/usr/bin/env python3
"""Start the Super Alita server on port 8081 for testing."""

import uvicorn

from src.main import create_app

if __name__ == "__main__":
    app = create_app()
    print("🚀 Starting Super Alita server on http://localhost:8081")
    print("📍 Health endpoint: http://localhost:8081/health")
    print("📍 Healthz endpoint: http://localhost:8081/healthz")
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="info")