import os
import subprocess
from typing import Any


class ExtensionTestAutomation:
    def _ext_dir(self) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(here, "..", "..", "cortex-extension"))

    def run_extension_tests(self) -> dict[str, Any]:
        """Run VS Code extension tests (expects npm test configured)"""
        try:
            result = subprocess.run(
                ["npm", "test"],
                cwd=self._ext_dir(),
                capture_output=True,
                text=True,
                timeout=900,
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout + result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Extension tests timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def build_extension(self) -> dict[str, Any]:
        """Build the VS Code extension (npm run compile)"""
        try:
            result = subprocess.run(
                ["npm", "run", "compile"],
                cwd=self._ext_dir(),
                capture_output=True,
                text=True,
                timeout=600,
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout + result.stderr,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
