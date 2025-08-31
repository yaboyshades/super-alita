#!/usr/bin/env python3
"""
Scan repository for placeholder markers and TODO-style notes in first-party code.

Excludes: .venv/, node_modules/, .git/, egg-info/, generated build outputs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = [
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bTBD\b",
    r"\bWIP\b",
    r"(?i)placeholder",
    r"(?i)in a real implementation",
    r"(?i)in real implementation",
    r"(?i)in production,",
]

IGNORE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    "*.egg-info",
}

INCLUDE_DIRS = {"src", "scripts", "cortex"}


def should_skip(path: Path) -> bool:
    parts = set(p.name for p in path.parents) | {path.name}
    for d in IGNORE_DIRS:
        if d in parts:
            return True
    return False


def main() -> int:
    problems: list[tuple[Path, int, str]] = []
    for top in INCLUDE_DIRS:
        base = ROOT / top
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_dir() or should_skip(p):
                continue
            if p.suffix not in {".py", ".md", ".json", ".ts", ".js"}:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            lines = text.splitlines()
            for i, line in enumerate(lines, 1):
                for pat in PATTERNS:
                    if re.search(pat, line):
                        problems.append((p.relative_to(ROOT), i, line.strip()))
                        break
    if problems:
        print("Found placeholder/TODO markers:")
        for p, i, line in problems:
            print(f" - {p}:{i}: {line}")
        return 1
    print("No placeholder/TODO markers found in first-party code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

