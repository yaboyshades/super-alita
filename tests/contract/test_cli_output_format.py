"""
Contract test for CLI output format specification.

These tests MUST FAIL initially as CLI implementation does not exist yet.
Part of TDD Phase 3.2 - T007.
"""

import json
import subprocess
from pathlib import Path


class TestCLIOutputFormatContract:
    """Contract tests for calculus gate CLI output format."""

    def test_cli_help_output_format(self):
        """Test that CLI help follows expected format."""
        # This test MUST FAIL - no CLI exists yet
        result = subprocess.run(
            ["python", "-m", "src.calculus_gate.cli", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        help_text = result.stdout

        # Expected sections in help output
        assert "usage:" in help_text.lower()
        assert "calculus gate" in help_text.lower()
        assert "--function" in help_text
        assert "--threshold" in help_text
        assert "--output" in help_text

    def test_cli_json_output_format(self):
        """Test that CLI JSON output follows certificate schema."""
        # This test MUST FAIL - no CLI exists yet
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.calculus_gate.cli",
                "--function",
                "test_function",
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Should be valid JSON
        output_data = json.loads(result.stdout)

        # Should match certificate schema structure
        assert "function_name" in output_data
        assert "certificate_grade" in output_data
        assert "overall_compliance" in output_data

    def test_cli_human_readable_output_format(self):
        """Test that CLI human-readable output follows expected format."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.calculus_gate.cli",
                "--function",
                "test_function",
                "--output",
                "human",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Expected sections in human output
        assert "Calculus Gate Analysis" in output
        assert "Function:" in output
        assert "Grade:" in output
        assert "Gates:" in output
        assert "Slope Gate" in output
        assert "Curvature Gate" in output
        assert "Lipschitz Gate" in output

    def test_cli_table_output_format(self):
        """Test that CLI table output has proper structure."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.calculus_gate.cli",
                "--function",
                "test_function",
                "--output",
                "table",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Should have table headers and borders
        assert "Gate" in output
        assert "Threshold" in output
        assert "Measured" in output
        assert "Status" in output
        assert "┌" in output or "+" in output  # Table borders

    def test_cli_error_output_format(self):
        """Test that CLI error output follows expected format."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.calculus_gate.cli",
                "--function",
                "nonexistent_function",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        error_output = result.stderr

        # Should have structured error format
        assert "Error:" in error_output
        assert "nonexistent_function" in error_output

    def test_cli_verbose_output_includes_details(self):
        """Test that verbose mode includes additional details."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.calculus_gate.cli",
                "--function",
                "test_function",
                "--verbose",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Verbose output should include extra details
        assert "Sample Count:" in output
        assert "Input Sizes:" in output
        assert "Runtime Range:" in output
        assert "Confidence Intervals:" in output

    def test_cli_quiet_output_minimal(self):
        """Test that quiet mode produces minimal output."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.calculus_gate.cli",
                "--function",
                "test_function",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Quiet output should be minimal - just grade and pass/fail
        lines = output.strip().split("\n")
        assert len(lines) <= 3  # Grade, pass/fail, maybe one more line
        assert any("Grade:" in line for line in lines)

    def test_cli_file_output_format(self):
        """Test that CLI can write to file with proper format."""
        output_file = Path("test_calculus_output.json")

        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "src.calculus_gate.cli",
                    "--function",
                    "test_function",
                    "--output-file",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert output_file.exists()

            # File should contain valid JSON certificate
            with open(output_file) as f:
                data = json.load(f)

            assert "function_name" in data
            assert "certificate_grade" in data

        finally:
            if output_file.exists():
                output_file.unlink()

    def test_cli_threshold_configuration_output(self):
        """Test that CLI accepts and displays threshold configuration."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.calculus_gate.cli",
                "--function",
                "test_function",
                "--slope-limit",
                "1.5",
                "--curvature-limit",
                "0.8",
                "--lipschitz-limit",
                "8.0",
                "--verbose",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Should show configured thresholds
        assert "1.5" in output  # slope limit
        assert "0.8" in output  # curvature limit
        assert "8.0" in output  # lipschitz limit

    def test_cli_sampling_configuration_output(self):
        """Test that CLI accepts sampling configuration parameters."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.calculus_gate.cli",
                "--function",
                "test_function",
                "--min-size",
                "10",
                "--max-size",
                "1000",
                "--samples",
                "8",
                "--verbose",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Should show sampling parameters in verbose output
        assert "Sample Count: 8" in output or "samples" in output.lower()

    def test_cli_exit_codes(self):
        """Test that CLI uses proper exit codes."""
        # Success case
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.calculus_gate.cli",
                "--function",
                "passing_function",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Failure case (gates failed)
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.calculus_gate.cli",
                "--function",
                "failing_function",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

        # Error case (invalid arguments)
        result = subprocess.run(
            ["python", "-m", "src.calculus_gate.cli", "--invalid-argument"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2

    def test_cli_artifact_path_output(self):
        """Test that CLI shows path to generated artifacts."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "src.calculus_gate.cli",
                "--function",
                "test_function",
                "--verbose",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        # Should show artifact path
        assert "Certificate saved:" in output or "artifacts/" in output
