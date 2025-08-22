import contextlib
import subprocess
import tempfile
from typing import Any

from cortex.common.logging import get_logger

logger = get_logger("cortex.tools.formatters")


def format_code_with_black(code: str) -> str:
    """Format Python code using black"""
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name
    try:
        result = subprocess.run(
            ["black", "--quiet", "--line-length", "88", temp_file],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("black_formatting_failed", extra={"stderr": result.stderr})
            return code
        with open(temp_file, encoding="utf-8") as f:
            return f.read()
    except subprocess.TimeoutExpired:
        logger.warning("black_formatting_timeout")
        return code
    except Exception as e:
        logger.error("black_formatting_error", extra={"error": str(e)})
        return code
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


def format_code_with_ruff(code: str, file_path: str) -> dict[str, Any]:
    """(Optional) Example: format/check with ruff"""
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name
    try:
        check = subprocess.run(
            ["ruff", "check", "--select", "I", temp_file],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if check.returncode != 0:
            fix = subprocess.run(
                ["ruff", "format", temp_file],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if fix.returncode == 0:
                with open(temp_file, encoding="utf-8") as f:
                    return {"success": True, "formatted": True, "content": f.read()}
        return {"success": True, "formatted": False, "content": code}
    except subprocess.TimeoutExpired:
        logger.warning("ruff_formatting_timeout")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        logger.error("ruff_formatting_error", extra={"error": str(e)})
        return {"success": False, "error": str(e)}
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)
