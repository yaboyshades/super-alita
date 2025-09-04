#!/usr/bin/env python3
"""
Unified Super Alita startup script.
Usage:
  python start.py --mode web --port 8080
  python start.py --mode chat
  python start.py --mode mcp
  python start.py --mode consensus --model gpt-oss:20b
"""
import argparse
import asyncio
import os
import sys

from pathlib import Path

# add src/ to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["web", "chat", "mcp", "consensus"], default="web")
    parser.add_argument("--port", type=int, default=int(os.getenv("APP_PORT", "8080")))
    parser.add_argument("--host", default=os.getenv("APP_HOST", "0.0.0.0"))
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
    args = parser.parse_args()

    if args.mode == "web":
        import uvicorn
        from src.main import create_app
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)
        return

    elif args.mode == "chat":
        print("💬 Chat mode not implemented in this scaffold.")
        return

    elif args.mode == "mcp":
        print("🔌 MCP mode not implemented in this scaffold.")
        return

    elif args.mode == "consensus":
        print(f"🤝 Consensus mode stub for model={args.model}.")
        return

if __name__ == "__main__":
    main()