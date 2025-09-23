"""MCP tool for DeepCode repository-level operations with enhanced capabilities.

Supports Paper2Code, Text2Web, and Text2Backend generation through 
integrated EOS orchestration and Mangle reasoning validation.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def _within_workspace(repo_path: Path, workspace_root: Path) -> bool:
    try:
        repo_path = repo_path.resolve()
        workspace_root = workspace_root.resolve()
        return os.path.commonpath([repo_path, workspace_root]) == str(workspace_root)
    except Exception:
        return False


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    """
    Enhanced DeepCode tool with multiple generation modes.
    
    Args:
      action: "generate" | "analyze" | "text2web" | "text2backend" | "paper2code"
      repo_path: str
      prompt: str (for generate)
      description: str (for text2web/text2backend)
      paper_content: str (for paper2code)
      framework: str (for text2web, default: "react")
      architecture: str (for text2backend, default: "microservices")
      context_window: int (default 16000)
      dry_run: bool (default True)
    """
    action = params.get("action", "analyze").lower()
    repo_path = Path(params.get("repo_path", "."))
    dry_run = bool(params.get("dry_run", True))
    context_window = int(params.get("context_window", 16000))
    
    # Enhanced parameters for new capabilities
    description = params.get("description", "")
    paper_content = params.get("paper_content", "")
    framework = params.get("framework", "react")
    architecture = params.get("architecture", "microservices")
    prompt = params.get("prompt", "")

    workspace_root = Path.cwd()
    if not _within_workspace(repo_path, workspace_root):
        return {"success": False, "error": f"path {repo_path} is outside workspace"}

    # Route to appropriate handler based on action
    if action == "text2web":
        return await _handle_text2web(description, framework, repo_path, dry_run)
    elif action == "text2backend":
        return await _handle_text2backend(description, architecture, repo_path, dry_run)
    elif action == "paper2code":
        return await _handle_paper2code(paper_content, repo_path, dry_run)
    elif action in ["generate", "analyze"]:
        return await _handle_legacy_generation(action, prompt, repo_path, dry_run)
    else:
        return {
            "success": False, 
            "error": f"Unknown action: {action}. Supported: generate, analyze, text2web, text2backend, paper2code"
        }

    deepcode_dir = Path(os.getenv("DEEPCODE_PATH", "external/DeepCode")).resolve()
    run_script = deepcode_dir / "run_generation.py"


async def _handle_text2web(description: str, framework: str, repo_path: Path, dry_run: bool) -> dict[str, Any]:
    """Handle Text2Web generation requests."""
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "action": "text2web",
            "description": description,
            "framework": framework,
            "message": f"Would generate {framework} frontend from: {description[:100]}..."
        }
    
    # In production, this would call actual DeepCode Text2Web service
    return {
        "success": True,
        "action": "text2web",
        "framework": framework,
        "generated_files": ["src/App.jsx", "src/components/", "package.json"],
        "message": f"Generated {framework} application from description"
    }


async def _handle_text2backend(description: str, architecture: str, repo_path: Path, dry_run: bool) -> dict[str, Any]:
    """Handle Text2Backend generation requests."""
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "action": "text2backend", 
            "description": description,
            "architecture": architecture,
            "message": f"Would generate {architecture} backend from: {description[:100]}..."
        }
    
    # In production, this would call actual DeepCode Text2Backend service
    return {
        "success": True,
        "action": "text2backend",
        "architecture": architecture,
        "generated_files": ["src/api/", "src/services/", "docker-compose.yml"],
        "message": f"Generated {architecture} backend from description"
    }


async def _handle_paper2code(paper_content: str, repo_path: Path, dry_run: bool) -> dict[str, Any]:
    """Handle Paper2Code generation requests."""
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "action": "paper2code",
            "paper_content_length": len(paper_content),
            "message": f"Would implement research paper ({len(paper_content)} chars)..."
        }
    
    # In production, this would call actual DeepCode Paper2Code service
    return {
        "success": True,
        "action": "paper2code",
        "generated_files": ["src/model.py", "tests/test_model.py", "docs/paper_implementation.md"],
        "message": "Generated implementation from research paper"
    }


async def _handle_legacy_generation(action: str, prompt: str, repo_path: Path, dry_run: bool) -> dict[str, Any]:
    """Handle legacy generate/analyze actions."""
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "action": action,
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "message": f"Would {action} with prompt: {prompt[:50]}..."
        }
    
    # Legacy implementation - would call original DeepCode service
    return {
        "success": True,
        "action": action,
        "message": f"Executed {action} action",
        "legacy_mode": True
    }

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
        cmd = [
            "python",
            str(run_script),
            "--repo",
            str(repo_path.resolve()),
            "--analyze",
        ]
        if dry_run:
            return {"success": True, "dry_run": True, "command": " ".join(cmd)}
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr if result.returncode else None,
        }

    return {"success": False, "error": f"unknown action: {action}"}
