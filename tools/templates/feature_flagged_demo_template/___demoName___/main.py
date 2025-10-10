#!/usr/bin/env python3
"""
Demo: ___demoName___

___demoDescription___

Guarded by environment variable ENABLE_DEMO___DEMO_NAME_UPPER___ == 'true'.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def _enabled() -> bool:
    env_name = "ENABLE_DEMO___DEMO_NAME_UPPER___"
    return os.getenv(env_name, "false").lower() == "true"


def _ensure_sys_path() -> None:
    root = (
        Path(__file__).resolve().parents[2]
    )  # demos/<name>/main.py -> repo root
    src = root / "src"
    for p in (str(root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)


async def _run_demo() -> bool:
    print("[demo] ___demoName___ starting...")
    # TODO: Import your real demo module and invoke its async entrypoint
    # Example:
    #   from ___demoName___ import main
    #   return await main()
    await asyncio.sleep(0.01)
    print("[demo] ___demoName___ placeholder completed")
    return True


def main() -> None:
    if not _enabled():
        print(
            "[demo] Skipping ___demoName___ (set ENABLE_DEMO___DEMO_NAME_UPPER___=true to run)"
        )
        return
    _ensure_sys_path()
    result = asyncio.run(_run_demo())
    print(f"[demo] ___demoName___ result: {result}")


if __name__ == "__main__":
    main()
