"""Contract tests for `sdd_cli tasks` JSON output format.

These tests document the expected contract for the `/tasks` CLI command.
The command is expected to accept a plan path and emit a JSON payload
matching the `TasksResponse` schema defined in `src/sdd/models.py`.
"""

import json
import subprocess

import pytest


def test_cli_tasks_output_format() -> None:
    """Contract test for CLI tasks command output format."""
    # First create a feature spec and plan
    workspace_root = "d:/Coding_Projects/super-alita-clean"

    # Create a feature spec first
    specify_cmd = [
        "python",
        "-m",
        "src.sdd.sdd_cli",
        "specify",
        "Test Tasks Feature",
        "--format",
        "json",
    ]
    specify_result = subprocess.run(
        specify_cmd, capture_output=True, text=True, cwd=workspace_root
    )

    # Extract feature ID from specify output
    stdout_lines = specify_result.stdout.strip().split("\n")
    json_start = None
    for i, line in enumerate(stdout_lines):
        if line.strip().startswith("{"):
            json_start = i
            break

    if json_start is not None:
        json_content = "\n".join(stdout_lines[json_start:])
        specify_output = json.loads(json_content)
        feature_id = specify_output.get("feature_id")
    else:
        pytest.fail("Could not extract feature ID from specify command")

    # Create a plan from the spec
    plan_cmd = [
        "python",
        "-m",
        "src.sdd.sdd_cli",
        "plan",
        feature_id,
        "--format",
        "json",
    ]
    subprocess.run(
        plan_cmd, capture_output=True, text=True, cwd=workspace_root
    )

    # Now run the tasks command using the feature ID
    cmd = [
        "python",
        "-m",
        "src.sdd.sdd_cli",
        "tasks",
        feature_id,
        "--format",
        "json",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=workspace_root
    )

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

    # Check that feature_id matches the one from specify
    assert output["feature_id"] == feature_id, "Feature ID should match"

    # Check that tasks breakdown is not empty
    tasks_length = len(output["artifact_content"])
    assert tasks_length > 0, "Tasks breakdown should not be empty"

    # Check constitutional compliance structure
    compliance = output["constitutional_compliance"]
    assert isinstance(compliance, dict), "Compliance should be dict"
    assert len(compliance) > 0, "Should have constitutional compliance data"
