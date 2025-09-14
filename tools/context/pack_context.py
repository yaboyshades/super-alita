#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

BUDGET_CHARS = int(os.environ.get("CONTEXT_CHAR_BUDGET", "12000"))
INCLUDE = [
    "tools/context/spine.md",
    "features/*/state.md",
    "features/*/spec.md",
    "features/*/plan.md",
    "features/*/tasks.md",
    "research/reports/latest.md",
]

HEADERS = [
    r"^# CMA Spine",
    r"^# Feature Specification",
    r"^# Implementation Plan",
    r"^# Tasks",
    r"^### Compliance Summary",
]


def take_head(text: str, max_chars: int = 2000) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    n = 0
    for ln in lines:
        kept.append(ln)
        n += len(ln) + 1
        if n >= max_chars:
            break
    return "\n".join(kept)


def pull_sections(p: Path) -> str:
    t = p.read_text(encoding="utf-8", errors="ignore")
    # Grab title + first N chars + any lines matching our beacon headers
    parts = [take_head(t, 1200)]
    for h in HEADERS:
        m = re.search(h, t, re.M)
        if m is not None:
            # include header ± ~800 chars around it
            start = max(0, m.start() - 200)
            parts.append(t[start: start + 1000])
    return f"\n\n<!-- file:{p} -->\n" + "\n\n".join(parts).strip()


def main() -> None:
    candidates: list[tuple[int, Path]] = []
    for pat in INCLUDE:
        for path in glob.glob(pat, recursive=True):
            p = Path(path)
            if p.is_file():
                # score: prefer state > spec > plan > tasks > report,
                # and more recent files
                name = p.name
                base = 0
                if name == "state.md":
                    base = 100
                elif name == "spec.md":
                    base = 90
                elif name == "plan.md":
                    base = 80
                elif name == "tasks.md":
                    base = 70
                elif name.endswith(".md"):
                    base = 60
                score = base + int(p.stat().st_mtime // 3600)
                candidates.append((score, p))
    candidates.sort(reverse=True)

    out: list[str] = ["# Packed Context\n"]
    size = 0
    for _, p in candidates:
        chunk = pull_sections(p)
        if size + len(chunk) > BUDGET_CHARS:
            continue
        out.append(chunk)
        size += len(chunk)
    bundle = "\n\n---\n".join(out)
    Path("tools/context/bundle.md").write_text(bundle, encoding="utf-8")
    print({"bundle": "tools/context/bundle.md", "chars": len(bundle)})


if __name__ == "__main__":
    main()
