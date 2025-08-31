"""Enhanced Python Toolchain Plugin for Agent Mode.

This plugin integrates all Python development extensions to provide
comprehensive toolchain capabilities for the agent including:
- Advanced debugging capabilities
- Multi-environment management
- Comprehensive code quality tools
- Jupyter notebook integration
- Enhanced snippet management
- Intelligent code completion
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from reug_runtime.event_bus import BaseEventBus


class PythonToolchainIntelligence:
    """Intelligence layer for Python toolchain optimization."""

    def __init__(self):
        self.environment_cache = {}
        self.debugging_sessions = {}
        self.jupyter_kernels = {}
        self.formatting_stats = {
            "black_runs": 0,
            "isort_runs": 0,
            "ruff_fixes": 0,
            "pylint_checks": 0,
            "mypy_checks": 0,
            "total_improvements": 0,
        }
        self.snippet_intelligence = {
            "usage_patterns": {},
            "productivity_gains": 0,
            "most_used_snippets": [],
        }

    def analyze_environment_needs(self, project_path: str) -> dict[str, Any]:
        """Analyze project to determine optimal environment configuration."""
        analysis = {
            "python_version": sys.version_info[:2],
            "has_requirements": False,
            "has_poetry": False,
            "has_pipenv": False,
            "has_conda": False,
            "virtual_env_needed": True,
            "recommended_tools": [],
        }

        project = Path(project_path)
        if (project / "requirements.txt").exists():
            analysis["has_requirements"] = True
            analysis["recommended_tools"].append("pip")
        if (project / "pyproject.toml").exists():
            analysis["has_poetry"] = True
            analysis["recommended_tools"].append("poetry")
        if (project / "Pipfile").exists():
            analysis["has_pipenv"] = True
            analysis["recommended_tools"].append("pipenv")
        if (project / "environment.yml").exists():
            analysis["has_conda"] = True
            analysis["recommended_tools"].append("conda")

        return analysis

    def get_debugging_recommendations(self, file_path: str) -> dict[str, Any]:
        """Get intelligent debugging recommendations for a file."""
        recommendations = {
            "breakpoint_suggestions": [],
            "variable_watches": [],
            "step_strategy": "line_by_line",
            "debug_config": "python_file",
        }

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Analyze code complexity
            if "async def" in content:
                recommendations["debug_config"] = "python_async"
                recommendations["step_strategy"] = "async_aware"
            if "import pytest" in content or "def test_" in content:
                recommendations["debug_config"] = "python_pytest"
            if "if __name__" in content:
                recommendations["breakpoint_suggestions"].append(
                    "main execution block"
                )
            if "try:" in content:
                recommendations["breakpoint_suggestions"].append(
                    "exception handling blocks"
                )

        except Exception:
            pass

        return recommendations

    def optimize_jupyter_workflow(self, notebook_path: str) -> dict[str, Any]:
        """Optimize Jupyter notebook workflow for agent operations."""
        optimization = {
            "kernel_recommendation": "python3",
            "cell_execution_order": "sequential",
            "memory_management": "auto_cleanup",
            "visualization_mode": "inline",
            "export_formats": ["html", "pdf"],
        }

        try:
            if Path(notebook_path).suffix == ".ipynb":
                with open(notebook_path, encoding="utf-8") as f:
                    notebook = json.load(f)

                # Analyze notebook cells
                cell_count = len(notebook.get("cells", []))
                if cell_count > 50:
                    optimization["memory_management"] = "aggressive_cleanup"
                    optimization["cell_execution_order"] = "selective"

                # Check for data science libraries
                content = str(notebook)
                if any(
                    lib in content
                    for lib in ["pandas", "numpy", "matplotlib", "seaborn"]
                ):
                    optimization["kernel_recommendation"] = "data_science"
                    optimization["visualization_mode"] = "interactive"

        except Exception:
            pass

        return optimization


class PythonToolchainPlugin:
    """Comprehensive Python Toolchain Plugin for Agent Mode."""

    def __init__(self):
        self.name = "python_toolchain_agent"
        self.intelligence = PythonToolchainIntelligence()
        self.event_bus: BaseEventBus | None = None
        self.active_environments = {}
        self.debugging_active = False
        self.jupyter_servers = {}

    async def setup(
        self,
        event_bus: BaseEventBus,
        store: Any = None,
        config: dict[str, Any] = None,
    ):
        """Setup the plugin with event bus and configuration."""
        self.event_bus = event_bus
        self.config = config or {}

        # Register event handlers
        if hasattr(event_bus, "subscribe"):
            await event_bus.subscribe(
                "environment_change", self._handle_environment_change
            )
            await event_bus.subscribe(
                "debug_request", self._handle_debug_request
            )
            await event_bus.subscribe(
                "jupyter_operation", self._handle_jupyter_operation
            )
            await event_bus.subscribe(
                "code_quality_check", self._handle_code_quality
            )

    async def start(self):
        """Start the plugin and initialize toolchain."""
        print("🐍 Starting Python Toolchain Agent...")

        # Initialize environment detection
        await self._detect_environments()

        # Setup code quality pipeline
        await self._setup_quality_pipeline()

        # Initialize Jupyter integration
        await self._setup_jupyter_integration()

        print("✅ Python Toolchain Agent ready!")

    async def stop(self):
        """Stop the plugin and cleanup resources."""
        # Cleanup active debugging sessions
        for session_id in self.intelligence.debugging_sessions:
            await self._cleanup_debug_session(session_id)

        # Stop Jupyter servers
        for server_id in self.jupyter_servers:
            await self._stop_jupyter_server(server_id)

        print("🛑 Python Toolchain Agent stopped")

    async def _detect_environments(self):
        """Detect and register available Python environments."""
        try:
            # Detect virtual environments
            current_dir = Path.cwd()
            venv_paths = [
                current_dir / ".venv",
                current_dir / "venv",
                current_dir / "env",
            ]

            for venv_path in venv_paths:
                if venv_path.exists():
                    env_info = {
                        "path": str(venv_path),
                        "type": "virtualenv",
                        "active": False,
                        "python_version": None,
                    }

                    # Try to get Python version
                    try:
                        python_exe = (
                            venv_path
                            / ("Scripts" if os.name == "nt" else "bin")
                            / "python"
                        )
                        if python_exe.exists():
                            result = subprocess.run(
                                [str(python_exe), "--version"],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            if result.returncode == 0:
                                env_info["python_version"] = (
                                    result.stdout.strip()
                                )

                    except Exception:
                        pass

                    self.active_environments[venv_path.name] = env_info

        except Exception as e:
            print(f"⚠️ Environment detection error: {e}")

    async def _setup_quality_pipeline(self):
        """Setup integrated code quality pipeline."""
        quality_tools = {
            "black": {"available": False, "config": {}},
            "isort": {"available": False, "config": {}},
            "ruff": {"available": False, "config": {}},
            "pylint": {"available": False, "config": {}},
            "mypy": {"available": False, "config": {}},
            "flake8": {"available": False, "config": {}},
        }

        # Check tool availability
        for tool in quality_tools:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", tool, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                quality_tools[tool]["available"] = result.returncode == 0
            except Exception:
                try:
                    result = subprocess.run(
                        [tool, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    quality_tools[tool]["available"] = result.returncode == 0
                except Exception:
                    pass

        self.quality_tools = quality_tools

    async def _setup_jupyter_integration(self):
        """Setup Jupyter notebook integration."""
        try:
            # Check if Jupyter is available
            result = subprocess.run(
                [sys.executable, "-m", "jupyter", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.jupyter_available = result.returncode == 0

            if self.jupyter_available:
                # Get available kernels
                kernel_result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "jupyter",
                        "kernelspec",
                        "list",
                        "--json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if kernel_result.returncode == 0:
                    kernels = json.loads(kernel_result.stdout)
                    self.intelligence.jupyter_kernels = kernels.get(
                        "kernelspecs", {}
                    )

        except Exception as e:
            print(f"⚠️ Jupyter setup warning: {e}")
            self.jupyter_available = False

    async def _handle_environment_change(self, event: dict[str, Any]):
        """Handle environment change events."""
        env_name = event.get("environment")
        action = event.get("action", "activate")

        if action == "activate" and env_name in self.active_environments:
            # Activate environment logic
            env_info = self.active_environments[env_name]
            env_info["active"] = True

            # Emit environment activated event
            if self.event_bus:
                await self.event_bus.emit(
                    {
                        "type": "environment_activated",
                        "environment": env_name,
                        "path": env_info["path"],
                        "python_version": env_info.get("python_version"),
                    }
                )

    async def _handle_debug_request(self, event: dict[str, Any]):
        """Handle debugging requests."""
        file_path = event.get("file_path")
        debug_type = event.get("debug_type", "python_file")

        if file_path:
            recommendations = self.intelligence.get_debugging_recommendations(
                file_path
            )
            session_id = f"debug_{int(time.time())}"

            self.intelligence.debugging_sessions[session_id] = {
                "file_path": file_path,
                "type": debug_type,
                "recommendations": recommendations,
                "active": True,
                "start_time": time.time(),
            }

            # Emit debug session started
            if self.event_bus:
                await self.event_bus.emit(
                    {
                        "type": "debug_session_started",
                        "session_id": session_id,
                        "file_path": file_path,
                        "recommendations": recommendations,
                    }
                )

    async def _handle_jupyter_operation(self, event: dict[str, Any]):
        """Handle Jupyter notebook operations."""
        operation = event.get("operation")
        notebook_path = event.get("notebook_path")

        if operation == "optimize" and notebook_path:
            optimization = self.intelligence.optimize_jupyter_workflow(
                notebook_path
            )

            # Emit optimization recommendations
            if self.event_bus:
                await self.event_bus.emit(
                    {
                        "type": "jupyter_optimization",
                        "notebook_path": notebook_path,
                        "optimization": optimization,
                    }
                )

    async def _handle_code_quality(self, event: dict[str, Any]):
        """Handle code quality check requests."""
        file_path = event.get("file_path")
        tools = event.get("tools", ["black", "ruff", "mypy"])

        results = {}
        for tool in tools:
            if (
                tool in self.quality_tools
                and self.quality_tools[tool]["available"]
            ):
                try:
                    result = await self._run_quality_tool(tool, file_path)
                    results[tool] = result
                    self.intelligence.formatting_stats[f"{tool}_runs"] += 1
                except Exception as e:
                    results[tool] = {"error": str(e)}

        # Emit quality check results
        if self.event_bus:
            await self.event_bus.emit(
                {
                    "type": "code_quality_results",
                    "file_path": file_path,
                    "results": results,
                    "stats": self.intelligence.formatting_stats,
                }
            )

    async def _run_quality_tool(
        self, tool: str, file_path: str
    ) -> dict[str, Any]:
        """Run a specific code quality tool."""
        try:
            if tool == "black":
                result = subprocess.run(
                    [sys.executable, "-m", "black", "--check", file_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            elif tool == "ruff":
                result = subprocess.run(
                    [sys.executable, "-m", "ruff", "check", file_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            elif tool == "mypy":
                result = subprocess.run(
                    [sys.executable, "-m", "mypy", file_path],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            else:
                return {"error": f"Unknown tool: {tool}"}

            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            return {"error": f"{tool} timed out"}
        except Exception as e:
            return {"error": str(e)}

    async def _cleanup_debug_session(self, session_id: str):
        """Cleanup a debugging session."""
        if session_id in self.intelligence.debugging_sessions:
            session = self.intelligence.debugging_sessions[session_id]
            session["active"] = False
            session["end_time"] = time.time()

    async def _stop_jupyter_server(self, server_id: str):
        """Stop a Jupyter server."""
        try:
            # Implementation would depend on how servers are tracked
            print(f"Stopping Jupyter server: {server_id}")
        except Exception as e:
            print(f"Error stopping Jupyter server {server_id}: {e}")

    def get_tools(self) -> list[dict[str, Any]]:
        """Get available tools for the ability registry."""
        return [
            {
                "name": "analyze_python_environment",
                "description": (
                    "Analyze Python project environment and provide recommendations"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to Python project",
                        }
                    },
                    "required": ["project_path"],
                },
            },
            {
                "name": "start_debug_session",
                "description": (
                    "Start an intelligent debugging session for a Python file"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to Python file to debug",
                        },
                        "debug_type": {
                            "type": "string",
                            "description": (
                                "Type of debugging (python_file, pytest, async)"
                            ),
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "optimize_jupyter_notebook",
                "description": (
                    "Optimize a Jupyter notebook for better performance and workflow"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notebook_path": {
                            "type": "string",
                            "description": "Path to Jupyter notebook",
                        }
                    },
                    "required": ["notebook_path"],
                },
            },
            {
                "name": "run_code_quality_check",
                "description": (
                    "Run comprehensive code quality checks using multiple tools"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to Python file",
                        },
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tools to run",
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "get_toolchain_status",
                "description": (
                    "Get comprehensive status of Python toolchain integration"
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    async def analyze_python_environment(
        self, project_path: str
    ) -> dict[str, Any]:
        """Analyze Python environment for a project."""
        analysis = self.intelligence.analyze_environment_needs(project_path)
        analysis["available_environments"] = self.active_environments
        analysis["quality_tools"] = self.quality_tools
        analysis["jupyter_available"] = getattr(
            self, "jupyter_available", False
        )
        return analysis

    async def start_debug_session(
        self, file_path: str, debug_type: str = "python_file"
    ) -> dict[str, Any]:
        """Start a debugging session."""
        await self._handle_debug_request(
            {"file_path": file_path, "debug_type": debug_type}
        )

        # Find the most recent session
        recent_session = max(
            self.intelligence.debugging_sessions.items(),
            key=lambda x: x[1]["start_time"],
            default=(None, {}),
        )

        return {
            "session_id": recent_session[0] if recent_session[0] else None,
            "recommendations": recent_session[1].get("recommendations", {}),
            "status": "started",
        }

    async def optimize_jupyter_notebook(
        self, notebook_path: str
    ) -> dict[str, Any]:
        """Optimize a Jupyter notebook."""
        optimization = self.intelligence.optimize_jupyter_workflow(
            notebook_path
        )

        await self._handle_jupyter_operation(
            {"operation": "optimize", "notebook_path": notebook_path}
        )

        return {
            "notebook_path": notebook_path,
            "optimization": optimization,
            "status": "optimized",
        }

    async def run_code_quality_check(
        self, file_path: str, tools: list[str] | None = None
    ) -> dict[str, Any]:
        """Run code quality checks."""
        if tools is None:
            tools = ["black", "ruff", "mypy"]

        await self._handle_code_quality(
            {"file_path": file_path, "tools": tools}
        )

        # Return immediate status; actual results come via events
        return {
            "file_path": file_path,
            "tools_requested": tools,
            "status": "running",
        }

    async def get_toolchain_status(self) -> dict[str, Any]:
        """Get comprehensive toolchain status."""
        return {
            "environments": self.active_environments,
            "quality_tools": self.quality_tools,
            "jupyter_available": getattr(self, "jupyter_available", False),
            "jupyter_kernels": self.intelligence.jupyter_kernels,
            "debugging_sessions": len(self.intelligence.debugging_sessions),
            "formatting_stats": self.intelligence.formatting_stats,
            "snippet_intelligence": self.intelligence.snippet_intelligence,
        }
