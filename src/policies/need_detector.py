from __future__ import annotations

import re
from re import Pattern


class NeedDetector:
    """Detect capability kinds implied by a natural-language task description."""

    KINDS: dict[str, list[Pattern]] = {
        "web_scraper": [
            re.compile(r"\b(scrap(?:e|er|ing)|crawl(?:er|ing)?)\b", re.I),
            re.compile(r"\bextract (?:price|product|listing|sku|catalog)\b", re.I),
            re.compile(r"\be-?commerce|price\s*(?:tracker|fetch|pull|harvest)\b", re.I),
        ],
        "etl_task": [
            re.compile(r"\bETL\b|\bextract-?transform-?load\b", re.I),
            re.compile(r"\bdata (?:pipeline|ingestion|normaliz(?:e|ation))\b", re.I),
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
        "paper2code": [
            re.compile(r"\b(paper|research|algorithm|implement|reproduce)\b", re.I),
            re.compile(r"\b(transformer|attention|neural|network|model)\b", re.I),
            re.compile(r"\b(machine learning|deep learning|AI|ML)\b", re.I),
            re.compile(r"\b(pytorch|tensorflow|numpy|sklearn)\b", re.I),
        ],
        "ml_algorithm": [
            re.compile(
                r"\b(classification|regression|clustering|optimization)\b", re.I
            ),
            re.compile(r"\b(gradient|backprop|training|learning)\b", re.I),
            re.compile(r"\b(embedding|encoder|decoder|layer)\b", re.I),
        ],
        "web_search": [
            re.compile(r"\b(search|find|lookup|query|discover)\b", re.I),
            re.compile(r"\b(google|bing|search engine|web search)\b", re.I),
            re.compile(r"\b(academic search|paper search|arxiv)\b", re.I),
            re.compile(r"\b(news search|reddit search|image search)\b", re.I),
        ],
        "research_assistant": [
            re.compile(r"\b(research|study|investigate|analyze)\b", re.I),
            re.compile(r"\b(academic|scholar|citation|bibliography)\b", re.I),
            re.compile(r"\b(literature review|systematic review)\b", re.I),
            re.compile(r"\b(knowledge|information|facts|data)\b", re.I),
        ],
        "information_retrieval": [
            re.compile(r"\b(retrieve|gather|collect|aggregate)\b", re.I),
            re.compile(r"\b(information|data|content|sources)\b", re.I),
            re.compile(r"\b(multi-source|cross-reference|verify)\b", re.I),
            re.compile(r"\b(knowledge base|database|index)\b", re.I),
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
