import json
import subprocess

import pytest


def test_cli_specify_output_format() -> None:
    """Contract test for CLI specify command output format."""
    # Run the CLI specify command
    workspace_root = "d:/Coding_Projects/super-alita-clean"
    cmd = [
        "python",
        "-m",
        "src.sdd.sdd_cli",
        "specify",
        "Test Constitutional Feature",
        "--format",
        "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=workspace_root)

    # The command should succeed (exit code 0)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Parse the output as JSON - extract JSON part after any text
    try:
        stdout_lines = result.stdout.strip().split("\n")
        json_start = None
        for i, line in enumerate(stdout_lines):
            if line.strip().startswith("{"):
                json_start = i
                break

        if json_start is not None:
            json_content = "\n".join(stdout_lines[json_start:])
            output = json.loads(json_content)
        else:
            pytest.fail(f"No JSON found in output: {result.stdout}")
    except json.JSONDecodeError as e:
        pytest.fail(f"Output is not valid JSON: {result.stdout}, error: {e}")

    # Check required fields in the output
    assert isinstance(output, dict), "Output should be a dictionary"

    # Check for expected keys based on ArtifactResult model
    required_keys = [
        "feature_id",
        "artifact_path",
        "artifact_content",
        "guidance",
        "constitutional_compliance",
        "overall_compliance_score",
        "compliance_threshold_met",
        "next_steps",
    ]
    for key in required_keys:
        assert key in output, f"Missing required key: {key}"

    # Check types
    assert isinstance(output["feature_id"], str)
    assert isinstance(output["artifact_path"], str)
    assert isinstance(output["artifact_content"], str)
    assert isinstance(output["guidance"], dict | type(None))
    assert isinstance(output["constitutional_compliance"], dict)
    assert isinstance(output["overall_compliance_score"], int | float)
    assert isinstance(output["compliance_threshold_met"], bool)
    assert isinstance(output["next_steps"], list)

    # Check that feature_id is generated
    assert len(output["feature_id"]) > 0, "Feature ID should be generated"

    # Check that artifact content is not empty
    assert len(output["artifact_content"]) > 0, "Artifact content should not be empty"

    # Check constitutional compliance structure
    compliance = output["constitutional_compliance"]
    assert isinstance(compliance, dict), "Compliance should be dict"
    assert len(compliance) > 0, "Should have constitutional compliance data"
