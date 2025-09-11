#!/usr/bin/env python3
"""
Socratic Testing Engine (STE) v1.0

Proactively challenges draft specifications to discover ambiguities,
edge cases, and hidden assumptions before implementation.

Usage:
  python tools/socratic_testing_engine.py --spec docs/feature_spec.md --output report.yaml
  python tools/socratic_testing_engine.py --spec - < docs/feature_spec.md

Dependencies: PyYAML (yaml). Optional: markdown. If 'markdown' is missing, a
lightweight parser fallback is used that scans '#', '##' headings.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception as _e:  # pragma: no cover
    yaml = None  # type: ignore

try:
    import markdown  # type: ignore
except Exception:
    markdown = None  # type: ignore


ENGINE_VERSION = "1.0.0"


@dataclass
class Finding:
    category: str
    severity: str
    location: str
    issue: str
    question: str
    suggested_resolution: str


class SocraticTestingEngine:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def _parse_spec(self, text: str) -> dict[str, Any]:
        """Parse markdown text into a basic structure of sections and lines.

        Prefer python-markdown when available; otherwise, perform a simple
        heading-based split on lines starting with '#', '##', etc.
        """
        lines = text.splitlines()
        sections: dict[str, dict[str, Any]] = {}
        current = "ROOT"
        buf: list[str] = []
        for idx, line in enumerate(lines, start=1):
            if re.match(r"^#{1,6}\s+", line):
                # save previous
                if buf:
                    sections.setdefault(current, {"content": ""})
                    sections[current]["content"] = "\n".join(buf).strip()
                # new section
                hdr = re.sub(r"^#{1,6}\s+", "", line).strip()
                current = hdr
                buf = []
            else:
                buf.append(line)
        if buf:
            sections.setdefault(current, {"content": ""})
            sections[current]["content"] = "\n".join(buf).strip()
        # Include original lines for location mapping
        sections["__lines__"] = {"content": lines}
        return sections

    def _scan_ambiguities(self, lines: list[str]) -> list[tuple[int, str]]:
        patterns = [
            r"\b(many|some|often|usually|typically|fast|quickly)\b",
            r"handles errors gracefully",
            r"as needed|etc\.|and so on",
        ]
        amb: list[tuple[int, str]] = []
        for i, line in enumerate(lines, start=1):
            for pat in patterns:
                if re.search(pat, line, re.IGNORECASE):
                    amb.append((i, line.strip()))
                    break
        return amb

    def _scan_acceptance(self, text: str) -> dict[str, Any]:
        # Look for Given/When/Then patterns
        gwt = re.findall(r"Given .*? when .*? then .*?", text, flags=re.IGNORECASE | re.DOTALL)
        return {"gwt_count": len(gwt)}

    def _scan_edge_cases(self, text: str) -> list[str]:
        hints: list[str] = []
        # Explicitly missing bounds phrasing
        if re.search(r"\bno\s+maximum\s+(size|length|limit)\b", text, re.IGNORECASE) or re.search(
            r"\b(without\s+(limit|bounds?)|unbounded)\b", text, re.IGNORECASE
        ):
            hints.append("Inputs described as unbounded or without maximum limits")
        # Generic error mention without enumerated types
        if re.search(r"error", text, re.IGNORECASE) and not re.search(
            r"specific error types|error codes", text, re.IGNORECASE
        ):
            hints.append("Errors referenced without enumerating specific types/codes")
        return hints

    def challenge_spec(self, spec_content: str, *, spec_file: str | None = None) -> dict[str, Any]:
        parsed = self._parse_spec(spec_content)
        lines: list[str] = parsed["__lines__"]["content"]

        findings: list[Finding] = []
        # Ambiguity detection
        for lineno, snippet in self._scan_ambiguities(lines):
            findings.append(
                Finding(
                    category="ambiguity",
                    severity="high",
                    location=f"line {lineno}",
                    issue=f"Ambiguous phrasing: '{snippet[:80]}'",
                    question="What specific, measurable behavior is expected?",
                    suggested_resolution="Replace vague terms with quantifiable criteria.",
                )
            )

        # Acceptance criteria checks
        acc = self._scan_acceptance(spec_content)
        if acc["gwt_count"] == 0:
            findings.append(
                Finding(
                    category="acceptance_criteria",
                    severity="medium",
                    location="User Stories / Acceptance Criteria",
                    issue="No Given/When/Then scenarios detected",
                    question="Provide concrete Given/When/Then examples for key stories?",
                    suggested_resolution="Add at least one G/W/T scenario per primary story.",
                )
            )

        # Edge case hints
        edge_hints = self._scan_edge_cases(spec_content)
        for hint in edge_hints:
            findings.append(
                Finding(
                    category="edge_case",
                    severity="medium",
                    location="Functional Requirements / Inputs",
                    issue=hint,
                    question="What happens at boundaries and invalid inputs?",
                    suggested_resolution="Specify boundary conditions and invalid input handling.",
                )
            )

        # Constitutional compliance quick check
        compliance: dict[str, dict[str, Any]] = {
            "article_i_library_first": {"status": "unknown", "notes": ""},
            "article_ii_test_first": {"status": "unknown", "notes": ""},
            "article_iii_simplicity_gate": {"status": "unknown", "notes": ""},
        }
        # Heuristics
        compliance["article_ii_test_first"]["status"] = "violation" if acc["gwt_count"] == 0 else "compliant"
        compliance["article_ii_test_first"]["notes"] = (
            "No G/W/T scenarios found" if acc["gwt_count"] == 0 else "Has G/W/T scenarios"
        )

        # Score (naive): fewer findings → higher readiness
        base = 1.0
        penalty = min(0.8, 0.05 * len(findings))
        readiness = round(base - penalty, 2)

        report = {
            "analysis_metadata": {
                "spec_file": spec_file or "inline",
                "analysis_timestamp": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat(),
                "engine_version": ENGINE_VERSION,
                "spec_readiness_score": readiness,
            },
            "findings": [f.__dict__ for f in findings],
            "constitutional_compliance": compliance,
            "readiness_assessment": {
                "ready_for_implementation": readiness >= 0.9 and not findings,
                "blocking_issues": sum(1 for f in findings if f.severity == "high"),
                "recommended_actions": "Address high-severity ambiguities and add concrete acceptance scenarios",
            },
        }
        return report

    def generate_report(self, findings: dict[str, Any]) -> str:
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML not available; cannot generate YAML report")
        return yaml.safe_dump(findings, sort_keys=False, allow_unicode=True)

    def generate_report_json(self, findings: dict[str, Any]) -> str:
        import json as _json

        return _json.dumps(findings, ensure_ascii=False, indent=2)


def _read_spec(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Socratic Testing Engine (STE) v1.0")
    parser.add_argument("--spec", required=True, help="Path to markdown specification file or '-' for stdin")
    parser.add_argument("--output", help="Path to write report (default: stdout)")
    parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format (yaml|json). Default: yaml",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Shorthand for --format json (overrides --format)",
    )
    args = parser.parse_args(argv)

    try:
        spec_text = _read_spec(args.spec)
        engine = SocraticTestingEngine()
        report = engine.challenge_spec(spec_text, spec_file=None if args.spec == '-' else args.spec)
        fmt = "json" if args.json else args.format
        payload = (
            engine.generate_report_json(report)
            if fmt == "json"
            else engine.generate_report(report)
        )
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload)
        return 0
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"STE error: {e}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
