import contextlib
import json
import os
import subprocess
from typing import Any

from cortex.common.logging import get_logger

logger = get_logger("cortex.tools.testing")


def run_tests(test_path: str = "", coverage: bool = True) -> dict[str, Any]:
    """Run pytest tests and return results"""
    cmd = ["pytest", "-v", "--tb=short"]
    if coverage:
        cmd.extend(["--cov", ".", "--cov-report", "json"])
    if test_path:
        cmd.append(test_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        coverage_data = None
        if coverage and (result.returncode in (0, 5)):
            try:
                with open("coverage.json", encoding="utf-8") as f:
                    coverage_json = json.load(f)
                    coverage_data = coverage_json.get("totals", {}).get(
                        "percent_covered"
                    )
            except FileNotFoundError:
                pass
            with contextlib.suppress(OSError):
                os.remove("coverage.json")
        return {
            "success": result.returncode == 0,
            "passed": result.returncode == 0,
            "output": result.stdout + result.stderr,
            "coverage": coverage_data,
        }
    except subprocess.TimeoutExpired:
        logger.error("test_execution_timeout")
        return {
            "success": False,
            "passed": False,
            "output": "Test execution timed out after 2 minutes",
            "coverage": None,
        }
    except Exception as e:
        logger.error("test_execution_error", extra={"error": str(e)})
        return {
            "success": False,
            "passed": False,
            "output": f"Test execution error: {str(e)}",
            "coverage": None,
        }


def run_linters(code: str, file_path: str) -> dict[str, Any]:
    """Run linters on code and return results (Ruff + Mypy on demand)"""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name
    try:
        ruff_res = subprocess.run(
            ["ruff", "check", temp_file], capture_output=True, text=True, timeout=30
        )
        mypy_res = None
        if file_path.endswith(".py"):
            mypy_res = subprocess.run(
                ["mypy", temp_file], capture_output=True, text=True, timeout=30
            )

        return {
            "ruff": {
                "success": ruff_res.returncode == 0,
                "output": ruff_res.stdout + ruff_res.stderr,
            },
            "mypy": {
                "success": (mypy_res.returncode == 0) if mypy_res else True,
                "output": (mypy_res.stdout + mypy_res.stderr) if mypy_res else "",
            },
        }
    except subprocess.TimeoutExpired:
        logger.warning("linter_timeout")
        return {
            "ruff": {"success": False, "output": "Timeout"},
            "mypy": {"success": False, "output": "Timeout"},
        }
    except Exception as e:
        logger.error("linter_error", extra={"error": str(e)})
        return {
            "ruff": {"success": False, "output": str(e)},
            "mypy": {"success": False, "output": str(e)},
        }
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)
