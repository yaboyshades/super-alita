#!/usr/bin/env python3

"""
Tribal Knowledge Extractor (TKE) v1.0

Converts resolved Socratic challenge outcomes into Architectural Decision Registry (ADR) entries.

Usage:
  python tools/tribal_knowledge_extractor.py --socratic-report report.json --commit abc123 --spec-path docs/spec.md --format yaml > adr.yaml
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def _now_date() -> str:
    return _dt.datetime.now(_dt.UTC).date().isoformat()


class TribalKnowledgeExtractor:
    def __init__(self, project: str = "unknown", output_format: str = "yaml") -> None:
        self.project = project
        self.output_format = output_format

    def extract_from_inputs(
        self,
        *,
        commit: str | None,
        spec_path: str | None,
        socratic_report: str | None,
    ) -> dict[str, Any]:
        """Build a minimal ADR registry from available inputs.

        Heuristics: use the first finding as the seed decision; map categories to
        constitutional articles; fall back to placeholders when needed.
        """
        report: dict[str, Any] = {}
        if socratic_report:
            try:
                report = json.loads(socratic_report)
            except Exception:
                # try YAML
                if yaml is not None:  # pragma: no cover
                    report = yaml.safe_load(socratic_report) or {}
        meta = report.get("analysis_metadata", {}) if isinstance(report, dict) else {}
        findings = report.get("findings", []) if isinstance(report, dict) else []
        first = findings[0] if findings else {}

        category = str(first.get("category", "ambiguity")).lower()
        article_map = {
            "ambiguity": ["V"],  # Clarity and Unambiguity
            "edge_case": ["II"],  # Test-First / edge tests
            "acceptance_criteria": ["II"],
        }
        articles = article_map.get(category, ["VI"])  # default to codification

        context_lines = []
        context_lines.append(f"Spec: {spec_path or meta.get('spec_file','unknown')}")
        if first.get("question"):
            context_lines.append(f"Socratic challenge: {first.get('question')}")
        if first.get("issue"):
            context_lines.append(f"Issue: {first.get('issue')}")
        context = "\n".join(context_lines).strip()

        decision = str(
            first.get("suggested_resolution") or "Document resolution details here."
        )
        rationale = "Resolution derived from Socratic challenge outcomes and project constraints."
        consequences = [
            "Improves specification clarity",
            "Enables measurable verification",
        ]

        adr = {
            "id": "ADR-001",
            "date": _now_date(),
            "title": "Decision extracted from Socratic challenge",
            "status": "accepted",
            "context": context or "Original context unavailable",
            "decision": decision,
            "alternatives": [],
            "rationale": rationale,
            "consequences": consequences,
            "constitutional_articles": articles,
            "source_commit": commit or "unknown",
        }

        registry = {
            "architectural_decision_registry": {
                "version": "1.0.0",
                "last_updated": _dt.datetime.now(_dt.UTC).isoformat(),
                "project": self.project,
            },
            "decisions": [adr],
        }
        return registry

    def append_to_registry(self, registry: dict[str, Any], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_format == "json":
            output_path.write_text(
                json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            if yaml is None:  # pragma: no cover
                raise RuntimeError("PyYAML required for YAML output")
            output_path.write_text(
                yaml.safe_dump(registry, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tribal Knowledge Extractor (TKE) v1.0"
    )
    parser.add_argument("--commit", help="Git commit hash (optional)")
    parser.add_argument("--spec-path", help="Path to spec file (optional)")
    parser.add_argument("--socratic-report", help="Path to Socratic report (YAML/JSON)")
    parser.add_argument(
        "--project", default="super-alita", help="Project name for ADR header"
    )
    parser.add_argument(
        "--output", help="Write ADR registry to this path (default: stdout)"
    )
    parser.add_argument("--format", choices=["yaml", "json"], default="yaml")
    args = parser.parse_args(argv)

    try:
        ste_payload = None
        if args.socratic_report:
            ste_payload = Path(args.socratic_report).read_text(encoding="utf-8")
        extractor = TribalKnowledgeExtractor(
            project=args.project, output_format=args.format
        )
        registry = extractor.extract_from_inputs(
            commit=args.commit, spec_path=args.spec_path, socratic_report=ste_payload
        )
        if args.output:
            extractor.append_to_registry(registry, Path(args.output))
        else:
            if args.format == "json":
                print(json.dumps(registry, ensure_ascii=False, indent=2))
            else:
                if yaml is None:  # pragma: no cover
                    raise RuntimeError("PyYAML required for YAML output")
                print(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True))
        return 0
    except Exception as e:  # pragma: no cover
        import sys

        sys.stderr.write(f"TKE error: {e}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
