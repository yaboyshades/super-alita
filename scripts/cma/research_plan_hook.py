#!/usr/bin/env python
"""Plan hook that runs research queries defined in the specification."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from research_agent.pipeline import ResearchPipeline

QUERY_PATTERN = re.compile(r"^-\s*Q\d+\s*:\s*\"(?P<query>.+?)\"", re.IGNORECASE)


def parse_queries(lines: Iterable[str]) -> list[str]:
    queries: list[str] = []
    for raw in lines:
        line = raw.strip()
        match = QUERY_PATTERN.match(line)
        if match:
            queries.append(match.group("query"))
    return queries


def build_markdown(results: list[dict]) -> str:
    lines = ["# Consolidated Research", ""]
    for idx, result in enumerate(results, start=1):
        query = result["query"]
        items = result["items"]
        count = result["stats"]["count"]
        lines.extend(
            [
                f"## Query {idx}",
                "",
                f"**Prompt:** {query}",
                "",
                f"Results found: {count}",
                "",
                "| # | Title | URL |", 
                "|---|-------|-----|",
            ]
        )
        if not items:
            lines.append("| – | No authoritative sources found | – |")
        else:
            for item_idx, item in enumerate(items, start=1):
                title = item["title"].replace("|", "\\|")
                url = item["url"]
                lines.append(f"| {item_idx} | {title} | {url} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run research pipeline for plan phase")
    parser.add_argument("spec", type=Path, help="Path to feature specification")
    parser.add_argument("--output", type=Path, help="Where to write research.md", required=False)
    parser.add_argument("--minimum", type=int, default=3, help="Minimum successful results required")
    args = parser.parse_args()

    if not args.spec.exists():
        raise SystemExit(f"spec file not found: {args.spec}")

    queries = parse_queries(args.spec.read_text(encoding="utf-8").splitlines())
    if not queries:
        raise SystemExit("no queries detected in specification")

    pipeline = ResearchPipeline()
    results = []
    for query in queries:
        record = pipeline.run(query).to_dict()
        if record["stats"]["count"] < args.minimum:
            raise SystemExit(f"insufficient sources for query '{query}'")
        results.append(record)

    feature_dir = args.spec.parent
    output = args.output or (feature_dir / "research.md")
    output.write_text(build_markdown(results), encoding="utf-8")

    json_payload = json.dumps(results[-1], indent=2)
    sys.stdout.write(json_payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
