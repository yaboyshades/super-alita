#!/usr/bin/env python3
"""
DeepCode Integration Ability - Provides workspace analysis and code understanding capabilities
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.plugin_interface import PluginInterface
from src.deepcode.integration import (
    analyze_current_file,
    analyze_workspace_sample,
    get_deepcode_integration,
    is_supported_file,
)

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class DeepCodeIntegrationAbility(PluginInterface):
    """Provides deepcode integration capabilities for workspace analysis and code understanding"""

    def __init__(self) -> None:
        super().__init__()
        self.integration = get_deepcode_integration()
        self.enabled = (
            os.getenv("DEEPCODE_INTEGRATION_ENABLED", "true").lower() == "true"
        )

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        logger.info("DeepCodeIntegrationAbility setup complete")

    @property
    def name(self) -> str:
        return "deepcode_integration_ability"

    async def start(self) -> None:
        await super().start()
        if not self.enabled:
            logger.info("DeepCodeIntegrationAbility disabled; not starting.")
            return

        # Register as a tool provider
        await self.subscribe("tool_execution_request", self._handle_tool_request)
        logger.info("DeepCodeIntegrationAbility started")

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Return list of available deepcode integration tools"""
        return [
            {
                "name": "analyze_workspace_context",
                "description": "Analyze workspace to understand project structure and context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_path": {
                            "type": "string",
                            "description": "Path to the workspace directory",
                        },
                        "max_files": {
                            "type": "integer",
                            "description": "Maximum number of files to analyze",
                            "default": 20,
                        },
                        "include_quality_metrics": {
                            "type": "boolean",
                            "description": "Whether to include quality metrics in analysis",
                            "default": True,
                        },
                    },
                    "required": ["workspace_path"],
                },
            },
            {
                "name": "analyze_file_context",
                "description": "Analyze a specific file with full context understanding",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to analyze",
                        },
                        "include_suggestions": {
                            "type": "boolean",
                            "description": "Whether to include improvement suggestions",
                            "default": True,
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "check_file_support",
                "description": "Check if a file type is supported by deepcode analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to check",
                        }
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "get_supported_extensions",
                "description": "Get list of file extensions supported by deepcode",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "understand_code_structure",
                "description": "Analyze code structure and provide architectural insights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_path": {
                            "type": "string",
                            "description": "Path to file or directory to analyze",
                        },
                        "focus_area": {
                            "type": "string",
                            "enum": [
                                "architecture",
                                "dependencies",
                                "complexity",
                                "patterns",
                            ],
                            "description": "Specific area to focus analysis on",
                            "default": "architecture",
                        },
                    },
                    "required": ["target_path"],
                },
            },
        ]

    async def _handle_tool_request(self, event: dict[str, Any]) -> None:
        """Handle tool execution requests for deepcode integration"""
        tool_name = event.get("tool_name")
        args = event.get("args", {})
        request_id = event.get("request_id", str(uuid.uuid4()))

        if tool_name not in [tool["name"] for tool in self.get_available_tools()]:
            return  # Not our tool

        try:
            result = await self._execute_tool(tool_name, args)
            await self.emit_event(
                "tool_execution_result",
                request_id=request_id,
                tool_name=tool_name,
                result=result,
                status="success",
                timestamp=_utcnow(),
            )
        except Exception as e:
            logger.exception(f"Tool execution failed for {tool_name}: {e}")
            await self.emit_event(
                "tool_execution_result",
                request_id=request_id,
                tool_name=tool_name,
                error=str(e),
                status="error",
                timestamp=_utcnow(),
            )

    async def _execute_tool(
        self, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the specified deepcode integration tool"""

        if tool_name == "analyze_workspace_context":
            return await self._analyze_workspace_context(args)
        elif tool_name == "analyze_file_context":
            return await self._analyze_file_context(args)
        elif tool_name == "check_file_support":
            return await self._check_file_support(args)
        elif tool_name == "get_supported_extensions":
            return await self._get_supported_extensions(args)
        elif tool_name == "understand_code_structure":
            return await self._understand_code_structure(args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def _analyze_workspace_context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Analyze workspace to understand project structure and context"""
        workspace_path = args["workspace_path"]
        max_files = args.get("max_files", 20)
        include_quality_metrics = args.get("include_quality_metrics", True)

        # Validate workspace exists
        if not Path(workspace_path).exists():
            raise FileNotFoundError(f"Workspace not found: {workspace_path}")

        # Use the integration to analyze workspace
        analysis_result = await analyze_workspace_sample(workspace_path)

        if "error" in analysis_result:
            return {
                "workspace_path": workspace_path,
                "status": "error",
                "error": analysis_result["error"],
            }

        # Extract insights from workspace analysis
        workspace_info = {
            "workspace_path": workspace_path,
            "status": "success",
            "total_files_found": analysis_result.get("files_analyzed", 0),
            "supported_files": analysis_result.get("supported_files", []),
            "project_type": self._infer_project_type(workspace_path),
            "main_languages": self._identify_main_languages(workspace_path),
        }

        if include_quality_metrics and "quality_summary" in analysis_result:
            workspace_info["quality_metrics"] = analysis_result["quality_summary"]

        # Limit file details to max_files
        if "file_details" in analysis_result:
            workspace_info["file_details"] = analysis_result["file_details"][:max_files]

        return workspace_info

    async def _analyze_file_context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Analyze a specific file with full context understanding"""
        file_path = args["file_path"]
        include_suggestions = args.get("include_suggestions", True)

        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if file is supported
        if not is_supported_file(file_path):
            return {
                "file_path": file_path,
                "supported": False,
                "message": "File type not supported by deepcode analysis",
            }

        # Analyze the file
        analysis_result = await analyze_current_file(file_path)

        if "error" in analysis_result:
            return {
                "file_path": file_path,
                "status": "error",
                "error": analysis_result["error"],
            }

        response = {
            "file_path": file_path,
            "supported": True,
            "status": "success",
            "issues_count": analysis_result.get("issues_count", 0),
            "quality_score": analysis_result.get("quality_score", 0),
            "execution_time": analysis_result.get("execution_time", 0),
            "metrics": analysis_result.get("metrics", {}),
        }

        # Include detailed issues if there are any
        if "issues" in analysis_result and analysis_result["issues"]:
            response["issues"] = analysis_result["issues"]

        # Include suggestions if requested
        if include_suggestions and analysis_result.get("issues_count", 0) > 0:
            response["improvement_suggestions"] = self._generate_file_suggestions(
                analysis_result
            )

        return response

    async def _check_file_support(self, args: dict[str, Any]) -> dict[str, Any]:
        """Check if a file type is supported by deepcode analysis"""
        file_path = args["file_path"]

        path = Path(file_path)
        supported = is_supported_file(file_path)

        return {
            "file_path": file_path,
            "file_extension": path.suffix,
            "supported": supported,
            "can_analyze": supported and path.exists(),
        }

    async def _get_supported_extensions(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get list of file extensions supported by deepcode"""
        # Get supported extensions from the integration
        supported_extensions = self.integration.get_supported_extensions()

        return {
            "supported_extensions": list(supported_extensions),
            "count": len(supported_extensions),
            "categories": {
                "python": [".py", ".pyw"],
                "config": [".json", ".yaml", ".yml", ".toml"],
                "documentation": [".md", ".rst", ".txt"],
            },
        }

    async def _understand_code_structure(self, args: dict[str, Any]) -> dict[str, Any]:
        """Analyze code structure and provide architectural insights"""
        target_path = args["target_path"]
        focus_area = args.get("focus_area", "architecture")

        path = Path(target_path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {target_path}")

        if path.is_file():
            # Single file structure analysis
            if not is_supported_file(str(path)):
                return {
                    "target_path": target_path,
                    "supported": False,
                    "message": "File type not supported for structure analysis",
                }

            analysis_result = await analyze_current_file(str(path))
            structure_info = self._extract_structure_info(analysis_result, focus_area)
        else:
            # Directory structure analysis
            analysis_result = await analyze_workspace_sample(str(path))
            structure_info = self._extract_workspace_structure(
                analysis_result, focus_area
            )

        return {
            "target_path": target_path,
            "focus_area": focus_area,
            "structure_analysis": structure_info,
        }

    def _infer_project_type(self, workspace_path: str) -> str:
        """Infer the type of project based on files present"""
        path = Path(workspace_path)

        # Check for specific project indicators
        if (path / "requirements.txt").exists() or (path / "pyproject.toml").exists():
            return "python"
        elif (path / "package.json").exists():
            return "javascript/nodejs"
        elif (path / "Cargo.toml").exists():
            return "rust"
        elif (path / "go.mod").exists():
            return "go"
        elif (path / "pom.xml").exists() or (path / "build.gradle").exists():
            return "java"
        elif (path / "README.md").exists():
            return "documentation"
        else:
            return "unknown"

    def _identify_main_languages(self, workspace_path: str) -> list[str]:
        """Identify the main programming languages used in the workspace"""
        path = Path(workspace_path)
        language_files = {}

        # Count files by extension
        for file_path in path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext:
                    language_files[ext] = language_files.get(ext, 0) + 1

        # Map extensions to languages
        ext_to_lang = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".rs": "Rust",
            ".go": "Go",
            ".rb": "Ruby",
            ".php": "PHP",
        }

        # Get top languages by file count
        languages = []
        for ext, count in sorted(
            language_files.items(), key=lambda x: x[1], reverse=True
        ):
            if ext in ext_to_lang and count > 0:
                languages.append(ext_to_lang[ext])
            if len(languages) >= 3:  # Limit to top 3
                break

        return languages

    def _generate_file_suggestions(self, analysis_result: dict[str, Any]) -> list[str]:
        """Generate improvement suggestions based on file analysis"""
        suggestions = []
        issues = analysis_result.get("issues", [])

        # Group issues by category
        security_issues = [
            i for i in issues if "security" in i.get("category", "").lower()
        ]
        complexity_issues = [
            i for i in issues if "complexity" in i.get("category", "").lower()
        ]
        performance_issues = [
            i for i in issues if "performance" in i.get("category", "").lower()
        ]

        if security_issues:
            suggestions.append("Review security issues and sanitize inputs")
        if complexity_issues:
            suggestions.append(
                "Consider refactoring complex functions for better maintainability"
            )
        if performance_issues:
            suggestions.append("Optimize performance-related patterns")

        # Add generic suggestions based on quality score
        quality_score = analysis_result.get("quality_score", 100)
        if quality_score < 70:
            suggestions.append("Overall code quality could be improved")

        return suggestions

    def _extract_structure_info(
        self, analysis_result: dict[str, Any], focus_area: str
    ) -> dict[str, Any]:
        """Extract structure information from file analysis"""
        metrics = analysis_result.get("metrics", {})

        base_info = {
            "lines_of_code": metrics.get("lines_of_code", 0),
            "functions": metrics.get("functions", 0),
            "classes": metrics.get("classes", 0),
        }

        if focus_area == "complexity":
            # Focus on complexity metrics
            issues = analysis_result.get("issues", [])
            complexity_issues = [
                i for i in issues if "complexity" in i.get("category", "").lower()
            ]
            base_info["complexity_issues"] = len(complexity_issues)
            base_info["complexity_details"] = [
                i.get("message", "") for i in complexity_issues[:5]
            ]

        elif focus_area == "dependencies":
            # Focus on imports and dependencies
            base_info["imports"] = metrics.get("imports", 0)

        elif focus_area == "patterns":
            # Focus on detected patterns
            issues = analysis_result.get("issues", [])
            patterns = {}
            for issue in issues:
                category = issue.get("category", "")
                patterns[category] = patterns.get(category, 0) + 1
            base_info["pattern_breakdown"] = patterns

        return base_info

    def _extract_workspace_structure(
        self, analysis_result: dict[str, Any], focus_area: str
    ) -> dict[str, Any]:
        """Extract structure information from workspace analysis"""
        return {
            "files_analyzed": analysis_result.get("files_analyzed", 0),
            "total_issues": analysis_result.get("total_issues", 0),
            "focus_area": focus_area,
            "workspace_summary": "Structure analysis completed",
        }


def create_ability() -> DeepCodeIntegrationAbility:
    """Factory function to create the ability"""
    return DeepCodeIntegrationAbility()
