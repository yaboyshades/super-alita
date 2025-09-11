#!/usr/bin/env python3
"""
Demo: complete_agent_demo

Demonstrates agent init + run using the VS Code integration and status checks.

Guarded by environment variable ENABLE_DEMO_COMPLETE_AGENT_DEMO == 'true'.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def _enabled() -> bool:
    return os.getenv("ENABLE_DEMO_COMPLETE_AGENT_DEMO", "false").lower() == "true"


def _ensure_sys_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    for p in (str(root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)


async def _run_demo() -> bool:
    # Reuse the existing demo's entrypoint for minimal refactor risk
    from complete_agent_demo import final_complete_demo

    return await final_complete_demo()


def main() -> None:
    if not _enabled():
        print(
            "[demo] Skipping complete_agent_demo (set ENABLE_DEMO_COMPLETE_AGENT_DEMO=true to run)"
        )
        return
    _ensure_sys_path()
    result = asyncio.run(_run_demo())
    print(f"[demo] complete_agent_demo result: {result}")


if __name__ == "__main__":
    main()

