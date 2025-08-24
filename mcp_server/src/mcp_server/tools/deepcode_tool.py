"""MCP tool for DeepCode repository-level operations (safe, dry-run by default)."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import os
import subprocess


def _within_workspace(repo_path: Path, workspace_root: Path) -> bool:
    try:
        repo_path = repo_path.resolve()
        workspace_root = workspace_root.resolve()
        return os.path.commonpath([repo_path, workspace_root]) == str(workspace_root)
    except Exception:
        return False


async def execute(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args:
      action: "generate" | "analyze"
      repo_path: str
      prompt: str (for generate)
      context_window: int (default 16000)
      dry_run: bool (default True)
    """
    action = params.get("action", "analyze").lower()
    repo_path = Path(params.get("repo_path", "."))
    dry_run = bool(params.get("dry_run", True))
    context_window = int(params.get("context_window", 16000))
    prompt = params.get("prompt", "")

    workspace_root = Path.cwd()
    if not _within_workspace(repo_path, workspace_root):
        return {"success": False, "error": f"path {repo_path} is outside workspace"}

    deepcode_dir = Path(os.getenv("DEEPCODE_PATH", "external/DeepCode")).resolve()
    run_script = deepcode_dir / "run_generation.py"

    if action == "generate":
        cmd = [
            "python",
            str(run_script),
            "--repo",
            str(repo_path.resolve()),
            "--prompt",
            prompt,
            "--context-window",
            str(context_window),
        ]
        if dry_run:
            return {"success": True, "dry_run": True, "command": " ".join(cmd)}
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr if result.returncode else None,
        }

    if action == "analyze":
        cmd = ["python", str(run_script), "--repo", str(repo_path.resolve()), "--analyze"]
        if dry_run:
            return {"success": True, "dry_run": True, "command": " ".join(cmd)}
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr if result.returncode else None,
        }

    return {"success": False, "error": f"unknown action: {action}"}
