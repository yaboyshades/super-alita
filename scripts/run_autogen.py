#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from pipelines.autogen_pipeline import autogen_any  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Generic ability autogen driver.")
    ap.add_argument(
        "--desc", required=True, help="Task description (natural language)."
    )
    ap.add_argument("--repo", default=".", help="Repo path.")
    ap.add_argument(
        "--iterations", type=int, default=5, help="Max DeepCode iterations."
    )
    args = ap.parse_args()

    res = autogen_any(
        description=args.desc, repo_path=args.repo, iterations=args.iterations
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
