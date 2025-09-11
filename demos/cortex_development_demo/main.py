#!/usr/bin/env python3
"""
Demo: cortex_development_demo

Interactive Cortex development demo for debugging and exploratory development.

Guarded by environment variable ENABLE_DEMO_CORTEX_DEVELOPMENT_DEMO == 'true'.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def _enabled() -> bool:
    return os.getenv("ENABLE_DEMO_CORTEX_DEVELOPMENT_DEMO", "false").lower() == "true"


def _ensure_sys_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    for p in (str(root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)


async def _run_demo() -> bool:
    # Reuse the existing demo's entrypoint for minimal refactor risk
    from cortex_development_demo import run_agent_cortex_demo

    return await run_agent_cortex_demo()


def main() -> None:
    if not _enabled():
        print(
            "[demo] Skipping cortex_development_demo (set ENABLE_DEMO_CORTEX_DEVELOPMENT_DEMO=true to run)"
        )
        return
    _ensure_sys_path()
    result = asyncio.run(_run_demo())
    print(f"[demo] cortex_development_demo result: {result}")


if __name__ == "__main__":
    main()

