#!/usr/bin/env python3
"""Utility script to run DeepCode analyze_current_file from CLI.

Usage (PowerShell):
  set DEEPCODE_ANALYSIS_LEVEL=DEEP; \
    .\\.venv\\Scripts\\python.exe scripts\\deepcode_run_file.py path\\to\\file.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import src.deepcode as deepcode


async def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: deepcode_run_file.py <file_path>")
        return 1
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return 1
    result = await deepcode.analyze_current_file(file_path)
    print("Analysis Level:", result.get("analysis_level"))
    print("Issues Count:", result.get("issues_count"))
    issues = result.get("issues") or []
    for issue in issues:
        print(
            f"- {issue['severity']}:{issue['category']} line {issue['line_number']} - "
            f"{issue['message']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
