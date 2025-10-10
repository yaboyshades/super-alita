from __future__ import annotations

from pathlib import Path

import pytest


def _sample_spec() -> str:
    return (
        "# Feature Specification\n\n"
        "## Objective\n"
        "The system should handle errors gracefully and perform quickly.\n\n"
        "## User Stories\n"
        "As a user I want X so that Y.\n\n"
        "### Acceptance Criteria\n"
        "- A narrative without Given/When/Then structure.\n\n"
        "## Functional Requirements\n"
        "Input validation is required.\n"
    )


def test_engine_basic_analysis(tmp_path: Path):
    from tools.socratic_testing_engine import SocraticTestingEngine

    spec = _sample_spec()
    engine = SocraticTestingEngine()
    report = engine.challenge_spec(spec, spec_file="inline")

    assert isinstance(report, dict)
    assert "analysis_metadata" in report
    assert "findings" in report and len(report["findings"]) >= 1
    # Expect ambiguity finding from "gracefully" / "quickly"
    categories = {f["category"] for f in report["findings"]}
    assert "ambiguity" in categories
    # GWT scenarios missing
    assert any(
        f["category"] == "acceptance_criteria" for f in report["findings"]
    )


def test_engine_report_yaml(tmp_path: Path):
    from tools.socratic_testing_engine import SocraticTestingEngine

    spec = _sample_spec()
    engine = SocraticTestingEngine()
    report = engine.challenge_spec(spec)
    yaml_text = engine.generate_report(report)
    # basic checks
    assert "analysis_metadata:" in yaml_text
    assert "findings:" in yaml_text


@pytest.mark.parametrize(
    "content, expect_edge",
    [
        ("Input is validated.", False),
        ("We handle input but no maximum size is specified.", True),
    ],
)
def test_edge_hints(content: str, expect_edge: bool):
    from tools.socratic_testing_engine import SocraticTestingEngine

    engine = SocraticTestingEngine()
    report = engine.challenge_spec("# Spec\n\n" + content)
    has_edge = any(f["category"] == "edge_case" for f in report["findings"])
    assert has_edge is expect_edge
