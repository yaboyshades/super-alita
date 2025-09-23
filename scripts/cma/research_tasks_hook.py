#!/usr/bin/env python
"""Generate research verification tasks to merge into tasks.md."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate research artifacts before /tasks generation")
    parser.add_argument("feature_dir", type=Path, help="Path to feature specs directory")
    args = parser.parse_args()

    research_file = args.feature_dir / "research.md"
    if not research_file.exists():
        raise SystemExit("research.md missing; run research_plan_hook first")

    tasks = [
        "Draft verification checklist for top citations in research.md",
        "Schedule follow-up research if any query produced <5 sources",
    ]
    for task in tasks:
        print(f"- {task}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
