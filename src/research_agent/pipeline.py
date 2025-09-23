"""Pipeline orchestrating Searx searches and schema validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from .metrics import Timer, log_event
from .searx_client import SearxClient

SCHEMA_PATH = Path(__file__).resolve().parent / "contracts" / "research_query.schema.json"


def _load_schema() -> Draft7Validator:
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    return Draft7Validator(schema)


@dataclass
class ResearchResult:
    query: str
    items: list[dict[str, str]]

    @property
    def stats(self) -> dict[str, int]:
        return {"count": len(self.items)}

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "items": self.items, "stats": self.stats}

    def to_markdown(self) -> str:
        lines = ["# Research Findings", "", f"**Query:** {self.query}", "", "| # | Title | URL |", "|---|-------|-----|"]
        for idx, item in enumerate(self.items, start=1):
            title = item.get("title", "").replace("|", r"\|")
            url = item.get("url", "")
            lines.append(f"| {idx} | {title} | {url} |")
        if not self.items:
            lines.append("| – | No results | – |")
        lines.append("")
        return "\n".join(lines)


class ResearchPipeline:
    """Coordinates querying, validation, and formatting."""

    def __init__(self, client: SearxClient | None = None) -> None:
        self.client = client or SearxClient()
        self.validator = _load_schema()

    def run(self, query: str) -> ResearchResult:
        with Timer("research.pipeline") as timer:
            items = self.client.search(query)
            result = ResearchResult(query=query, items=items)
            payload = result.to_dict()
            self.validator.validate(payload)
            log_event(
                "research.pipeline.completed",
                query=query,
                elapsed=timer.elapsed,
                count=result.stats["count"],
            )
            return result


__all__ = ["ResearchPipeline", "ResearchResult", "SCHEMA_PATH"]
