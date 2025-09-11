#!/usr/bin/env python3
"""
Diff Preview

Shows a unified diff between the current working tree and HEAD for given files.

Usage:
  python tools/diff_preview.py --files src/plugins/compose_plugin.py src/abilities/api_client.py

This is useful to demonstrate before/after changes such as:
  - Sandbox wrapping eval/exec
  - PluginInterface inheritance
  - Event bus initialize() injection
"""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


def git_show(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "show", f"HEAD:{path.as_posix()}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode != 0:
            return None
        return out.stdout
    except Exception:
        return None


def working_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def preview(files: Iterable[str]) -> int:
    rc = 0
    for f in files:
        p = Path(f)
        old = git_show(p)
        new = working_text(p)
        if old is None:
            print(f"\n--- {p} (no baseline in HEAD)\n")
            print(new)
            continue
        diff = difflib.unified_diff(
            old.splitlines(), new.splitlines(), fromfile=f"HEAD:{p}", tofile=str(p), lineterm=""
        )
        print("\n".join([*diff]) or f"\n(no changes) {p}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff preview against HEAD")
    ap.add_argument("--files", nargs="+", required=True, help="Files to diff")
    args = ap.parse_args()
    return preview(args.files)


if __name__ == "__main__":
    sys.exit(main())

