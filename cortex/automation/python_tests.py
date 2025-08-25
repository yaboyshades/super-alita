import contextlib
import json
import os
import subprocess
from typing import Any

from cortex.common.logging import get_logger

logger = get_logger(__name__)


class PythonTestAutomation:
    def run_pytest_with_coverage(self, test_path: str = "tests/") -> dict[str, Any]:
        """Run pytest with coverage reporting"""
        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    test_path,
                    "--cov=.",
                    "--cov-report=json",
                    "-v",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            coverage = None
            if os.path.exists("coverage.json"):
                with open("coverage.json", encoding="utf-8") as f:
                    coverage_data = json.load(f)
                    coverage = coverage_data.get("totals", {}).get("percent_covered")
                with contextlib.suppress(OSError):
                    os.remove("coverage.json")
            return {
                "success": result.returncode == 0,
                "output": result.stdout + result.stderr,
                "coverage": coverage,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Test execution timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def format_and_lint(self) -> dict[str, Any]:
        """Run Black formatting and Ruff linting (with fixes)"""
        results = {}
        fmt = subprocess.run(["black", "."], capture_output=True, text=True)
        results["formatting"] = {
            "success": fmt.returncode == 0,
            "output": fmt.stdout + fmt.stderr,
        }
        lint = subprocess.run(
            ["ruff", "check", "--fix", "."], capture_output=True, text=True
        )
        results["linting"] = {
            "success": lint.returncode == 0,
            "output": lint.stdout + lint.stderr,
        }
        return results
