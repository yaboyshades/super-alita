#!/usr/bin/env python3
"""Simple startup script for Super Alita with chat interface."""

import uvicorn

from app import app

if __name__ == "__main__":
    print("🚀 Starting Super Alita with Chat Interface...")
    print("💬 Access the chat interface at: http://127.0.0.1:8080")
    print("📋 API documentation at: http://127.0.0.1:8080/docs")
    print("🔧 Health check at: http://127.0.0.1:8080/healthz")
    print()

    try:
        uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info", access_log=True)
    except KeyboardInterrupt:
        print("\n👋 Super Alita chat interface stopped.")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
