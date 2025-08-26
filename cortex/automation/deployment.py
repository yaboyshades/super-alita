import os
import subprocess
from typing import Any


class DeploymentAutomation:
    def deploy_to_test(self) -> dict[str, Any]:
        """Shell out to your test deployment script"""
        try:
            res = subprocess.run(
                ["./deploy_test.sh"], capture_output=True, text=True, timeout=900
            )
            return {"success": res.returncode == 0, "output": res.stdout + res.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def package_extension(self) -> dict[str, Any]:
        """Package VS Code extension for publishing (expects npm run package)"""
        ext_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "cortex-extension")
        )
        try:
            res = subprocess.run(
                ["npm", "run", "package"],
                cwd=ext_dir,
                capture_output=True,
                text=True,
                timeout=600,
            )
            return {"success": res.returncode == 0, "output": res.stdout + res.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}
