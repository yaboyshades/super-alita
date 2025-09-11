#!/usr/bin/env python3
"""
Unified Launcher (template-generated)

Variables:
- modelName: default LLM model (e.g., gpt-4o)
- mode: SUPER_ALITA_MODE (shadow|act|batch)
- port: server port
- logLevel: INFO|DEBUG|WARNING
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_sys_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Super-Alita Unified Launcher")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=___port___)
    p.add_argument("--mode", choices=["shadow", "act", "batch"], default="___mode___")
    p.add_argument("--model", default="___modelName___")
    p.add_argument("--log-level", default="___logLevel___")
    p.add_argument("--reload", action="store_true")
    return p.parse_args()


def _apply_env(args: argparse.Namespace) -> None:
    # Core run-mode
    os.environ["SUPER_ALITA_MODE"] = args.mode

    # Model configuration
    os.environ.setdefault("LLM_PROVIDER", os.getenv("LLM_PROVIDER", "openai"))
    os.environ["LLM_MODEL"] = args.model

    # Logging
    os.environ["LOG_LEVEL"] = args.log_level
    os.environ["REUG_LOG_LEVEL"] = args.log_level

    # Port for external tools that read env
    os.environ["SUPER_ALITA_PORT"] = str(args.port)


def main() -> None:
    _ensure_sys_path()
    args = _parse_args()
    _apply_env(args)

    try:
        import uvicorn  # type: ignore

        from src.main import app  # type: ignore
    except Exception as e:  # pragma: no cover - import-time guard
        print(f"[launcher] Failed to import server: {e}")
        sys.exit(1)

    print(
        f"[launcher] Starting Super-Alita on {args.host}:{args.port} | "
        f"mode={args.mode} model={args.model} log={args.log_level}"
    )
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)  # type: ignore


if __name__ == "__main__":
    main()

