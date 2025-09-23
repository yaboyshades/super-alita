#!/usr/bin/env python
"""CLI entrypoint for the CMA v5.3 research pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_agent.pipeline import ResearchPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the constitutional research agent")
    parser.add_argument("query", type=str, help="Search query to execute")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file to write results (markdown when suffix=.md else JSON)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = ResearchPipeline()
    result = pipeline.run(args.query)
    payload = result.to_dict()

    if args.output:
        if args.output.suffix.lower() == ".md":
            args.output.write_text(result.to_markdown(), encoding="utf-8")
        else:
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
