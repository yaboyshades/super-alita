from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _sample_socratic_report() -> dict[str, Any]:
    return {
        "analysis_metadata": {
            "spec_file": "docs/example_feature.md",
            "engine_version": "1.0.0",
            "spec_readiness_score": 0.9,
        },
        "findings": [
            {
                "category": "ambiguity",
                "severity": "high",
                "location": "line 12",
                "issue": "Ambiguous phrasing: 'handles errors gracefully'",
                "question": "What specific, measurable behavior is expected?",
                "suggested_resolution": "Specify exact error handling for timeouts and network failures.",
            }
        ],
    }


def test_extractor_produces_registry(tmp_path: Path):
    from tools.tribal_knowledge_extractor import TribalKnowledgeExtractor

    report = _sample_socratic_report()
    ste_json = json.dumps(report)
    extractor = TribalKnowledgeExtractor(project="super-alita")
    registry = extractor.extract_from_inputs(
        commit="abc123",
        spec_path=report["analysis_metadata"]["spec_file"],
        socratic_report=ste_json,
    )

    assert isinstance(registry, dict)
    assert "architectural_decision_registry" in registry
    assert "decisions" in registry and isinstance(registry["decisions"], list)
    assert registry["decisions"], "Expected at least one ADR decision"
    adr = registry["decisions"][0]
    # Required keys
    for key in [
        "id",
        "date",
        "title",
        "context",
        "decision",
        "rationale",
        "consequences",
        "constitutional_articles",
        "source_commit",
    ]:
        assert key in adr


def test_cli_json_output(tmp_path: Path):
    import subprocess
    import sys

    report = _sample_socratic_report()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    cmd = [
        sys.executable,
        "tools/tribal_knowledge_extractor.py",
        "--socratic-report",
        str(report_path),
        "--commit",
        "abc123",
        "--spec-path",
        report["analysis_metadata"]["spec_file"],
        "--format",
        "json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert "architectural_decision_registry" in data
    assert data["decisions"], "No decisions produced by CLI"


def test_append_to_registry_writes_file(tmp_path: Path):
    from tools.tribal_knowledge_extractor import TribalKnowledgeExtractor

    report = _sample_socratic_report()
    extractor = TribalKnowledgeExtractor(project="super-alita")
    registry = extractor.extract_from_inputs(
        commit="abc123",
        spec_path=report["analysis_metadata"]["spec_file"],
        socratic_report=json.dumps(report),
    )
    out = tmp_path / "adr.yaml"
    extractor.append_to_registry(registry, out)
    content = out.read_text(encoding="utf-8")
    assert "architectural_decision_registry" in content
    assert "decisions:" in content
