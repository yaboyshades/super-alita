from __future__ import annotations

import re
from re import Pattern


class NeedDetector:
    """Detect capability kinds implied by a natural-language task description."""

    KINDS: dict[str, list[Pattern]] = {
        "web_scraper": [
            re.compile(r"\b(scrap(?:e|er|ing)|crawl(?:er|ing)?)\b", re.I),
            re.compile(r"\bextract (?:price|product|listing|sku|catalog)\b", re.I),
            re.compile(
                r"\be-?commerce|price\s*(?:tracker|fetch|pull|harvest)\b", re.I
            ),
        ],
        "etl_task": [
            re.compile(r"\bETL\b|\bextract-?transform-?load\b", re.I),
            re.compile(
                r"\bdata (?:pipeline|ingestion|normaliz(?:e|ation))\b", re.I
            ),
        ],
        "api_client": [
            re.compile(
                r"\bapi client\b|\bcall(?:ing)? (?:an )?api\b|\bfetch via "
                r"(?:http|rest)\b",
                re.I,
            ),
        ],
        "report_generator": [
            re.compile(r"\breport\b|\bsummary\b|\bexport (?:csv|xlsx|pdf)\b", re.I),
        ],
    }

    def detect(self, description: str) -> list[str]:
        text = (description or "").strip()
        if not text:
            return []
        hits: list[str] = []
        for kind, pats in self.KINDS.items():
            if any(p.search(text) for p in pats):
                hits.append(kind)
        return hits