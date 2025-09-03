#!/usr/bin/env python3
"""Strict guard against introducing mock / placeholder / scaffold files.

Updated to reduce false positives via token based matching and allow‑listing
selected legitimate files that previously triggered the guard.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Broad patterns – matched on token boundaries only.
BLOCKED_NAME_PATTERNS: tuple[str, ...] = (
    "mock",
    "mocks",
    "dummy",
    "placeholder",
    "scaffold",
    "skeleton",
    "tmp",
    "temp",
    "example",
)

BLOCKED_CONTENT_HINTS: tuple[str, ...] = (
    "THIS IS A PLACEHOLDER",
    "DO NOT USE",
    "MOCK IMPLEMENTATION",
    "DUMMY IMPLEMENTATION",
    "TO BE IMPLEMENTED",
    "INSERT CODE HERE",
)

# Explicit allow-list of legitimate files whose names previously matched tokens
# but are acceptable (e.g. generated lockfiles, the guard script itself, etc.).
ALLOWLIST_PATHS: set[str] = {
    "tools/git_hooks/no_mock_guard.py",
    "extensions/copilot-prompt-optimizer/package-lock.json",
}


def _tokenize(path: str) -> set[str]:
    base = path.lower().rsplit("/", 1)[-1]
    parts = re.split(r"[^a-z0-9]+", base)
    return {p for p in parts if p}


def _staged_changes() -> list[tuple[str, str]]:
    out = subprocess.check_output([
        "git",
        "diff",
        "--cached",
        "--name-status",
        "-z",
    ]).decode("utf-8", "replace")
    parts = [p for p in out.split("\x00") if p]
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(parts):
        status = parts[i]
        if i + 1 < len(parts):
            path = parts[i + 1]
            pairs.append((status, path))
        i += 2
    return pairs


def _is_blocked_name(path: str) -> bool:
    tokens = _tokenize(path)
    return any(pattern in tokens for pattern in BLOCKED_NAME_PATTERNS)


def _has_blocked_content(path: str) -> bool:
    try:
        data = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    up = data.upper()
    return any(hint in up for hint in BLOCKED_CONTENT_HINTS)


def main() -> int:
    violations: list[str] = []
    for status, path in _staged_changes():
        if not status or status[0] != "A":
            continue
        if path in ALLOWLIST_PATHS:
            continue
        if _is_blocked_name(path):
            violations.append(f"Added file matches blocked pattern: {path}")
            continue
        if _has_blocked_content(path):
            violations.append(
                f"Added file contains placeholder/mock content: {path}"
            )
    if violations:
        sys.stderr.write(
            "\n[no-mock-files] Blocked commit due to policy violations:\n"
        )
        for v in violations:
            sys.stderr.write(f" - {v}\n")
        sys.stderr.write(
            "\nPolicy: Avoid introducing mock/dummy/placeholder/scaffold "
            "files. Integrate with existing components unless absolutely "
            "required.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
