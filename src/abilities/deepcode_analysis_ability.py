#!/usr/bin/env python3
"""
DeepCode Analysis Ability - Exposes advanced code analysis capabilities to the agent
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.plugin_interface import PluginInterface
from src.deepcode.analyzer_simple import (
    AnalysisLevel,
    SeverityLevel,
    create_deepcode_engine,
)

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class DeepCodeAnalysisAbility(PluginInterface):
    """Provides deep code analysis capabilities to the agent"""

    def __init__(self) -> None:
        super().__init__()
        self.engine = create_deepcode_engine()
        self.enabled = os.getenv("DEEPCODE_ANALYSIS_ENABLED", "true").lower() == "true"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        logger.info("DeepCodeAnalysisAbility setup complete")

    @property
    def name(self) -> str:
        return "deepcode_analysis_ability"

    async def start(self) -> None:
        await super().start()
        if not self.enabled:
            logger.info("DeepCodeAnalysisAbility disabled; not starting.")
            return

        # Register as a tool provider
        await self.subscribe("tool_execution_request", self._handle_tool_request)
        logger.info("DeepCodeAnalysisAbility started")

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Return list of available deepcode analysis tools"""
        return [
            {
                "name": "analyze_code_file",
                "description": "Perform deep analysis on a single code file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the code file to analyze"
                        },
                        "analysis_level": {
                            "type": "string",
                            "enum": ["syntax", "semantic", "security", "performance", "architecture", "deep"],
                            "description": "Level of analysis to perform",
                            "default": "semantic"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "analyze_code_directory",
                "description": "Analyze all supported code files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "Path to the directory to analyze"
                        },
                        "analysis_level": {
                            "type": "string",
                            "enum": ["syntax", "semantic", "security", "performance", "architecture", "deep"],
                            "description": "Level of analysis to perform",
                            "default": "semantic"
                        },
                        "generate_report": {
                            "type": "boolean",
                            "description": "Whether to generate a comprehensive report",
                            "default": True
                        }
                    },
                    "required": ["directory_path"]
                }
            },
            {
                "name": "get_code_quality_score",
                "description": "Calculate quality score for code analysis results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_path": {
                            "type": "string",
                            "description": "Path to file or directory to analyze for quality score"
                        },
                        "include_metrics": {
                            "type": "boolean",
                            "description": "Whether to include detailed metrics in response",
                            "default": False
                        }
                    },
                    "required": ["target_path"]
                }
            },
            {
                "name": "detect_code_patterns",
                "description": "Detect specific patterns and anti-patterns in code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the code file to analyze"
                        },
                        "pattern_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of patterns to detect (security, performance, architecture)",
                            "default": ["security", "performance"]
                        }
                    },
                    "required": ["file_path"]
                }
            }
        ]

    async def _handle_tool_request(self, event: dict[str, Any]) -> None:
        """Handle tool execution requests for deepcode analysis"""
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
                timestamp=_utcnow()
            )
        except Exception as e:
            logger.exception(f"Tool execution failed for {tool_name}: {e}")
            await self.emit_event(
                "tool_execution_result",
                request_id=request_id,
                tool_name=tool_name,
                error=str(e),
                status="error",
                timestamp=_utcnow()
            )

    async def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute the specified deepcode analysis tool"""

        if tool_name == "analyze_code_file":
            return await self._analyze_file(args)
        elif tool_name == "analyze_code_directory":
            return await self._analyze_directory(args)
        elif tool_name == "get_code_quality_score":
            return await self._get_quality_score(args)
        elif tool_name == "detect_code_patterns":
            return await self._detect_patterns(args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def _analyze_file(self, args: dict[str, Any]) -> dict[str, Any]:
        """Analyze a single code file"""
        file_path = args["file_path"]
        analysis_level_str = args.get("analysis_level", "semantic")

        # Convert string to enum
        try:
            analysis_level = AnalysisLevel(analysis_level_str.lower())
        except ValueError:
            analysis_level = AnalysisLevel.SEMANTIC

        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Perform analysis
        result = await self.engine.analyze_file(file_path, analysis_level)

        return {
            "file_path": file_path,
            "analysis_level": analysis_level.value,
            "issues_found": len(result.issues),
            "execution_time": result.execution_time,
            "issues": [issue.to_dict() for issue in result.issues[:20]],  # Limit for performance
            "metrics": result.metrics,
            "suggestions": result.suggestions,
            "severity_breakdown": {
                severity.value: len(result.get_issues_by_severity(severity))
                for severity in SeverityLevel
            }
        }

    async def _analyze_directory(self, args: dict[str, Any]) -> dict[str, Any]:
        """Analyze all supported files in a directory"""
        directory_path = args["directory_path"]
        analysis_level_str = args.get("analysis_level", "semantic")
        generate_report = args.get("generate_report", True)

        # Convert string to enum
        try:
            analysis_level = AnalysisLevel(analysis_level_str.lower())
        except ValueError:
            analysis_level = AnalysisLevel.SEMANTIC

        # Validate directory exists
        if not Path(directory_path).exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Perform analysis
        results = await self.engine.analyze_directory(directory_path, analysis_level)

        if generate_report:
            report = self.engine.generate_report(results)
            return {
                "directory_path": directory_path,
                "analysis_level": analysis_level.value,
                "files_analyzed": len(results),
                "report": report
            }
        else:
            # Return simplified results
            total_issues = sum(len(result.issues) for result in results.values())
            return {
                "directory_path": directory_path,
                "analysis_level": analysis_level.value,
                "files_analyzed": len(results),
                "total_issues": total_issues,
                "files_with_issues": len([r for r in results.values() if r.issues])
            }

    async def _get_quality_score(self, args: dict[str, Any]) -> dict[str, Any]:
        """Calculate quality score for a file or directory"""
        target_path = args["target_path"]
        include_metrics = args.get("include_metrics", False)

        path = Path(target_path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {target_path}")

        if path.is_file():
            # Single file analysis
            result = await self.engine.analyze_file(str(path), AnalysisLevel.SEMANTIC)
            results = {str(path): result}
        else:
            # Directory analysis
            results = await self.engine.analyze_directory(str(path), AnalysisLevel.SEMANTIC)

        # Generate report to get quality score
        report = self.engine.generate_report(results)

        response = {
            "target_path": target_path,
            "quality_score": report["summary"]["quality_score"],
            "total_issues": report["summary"]["total_issues"],
            "files_analyzed": report["summary"]["files_analyzed"]
        }

        if include_metrics:
            response["metrics"] = report["metrics"]
            response["severity_breakdown"] = report["summary"]["severity_breakdown"]
            response["category_breakdown"] = report["summary"]["category_breakdown"]

        return response

    async def _detect_patterns(self, args: dict[str, Any]) -> dict[str, Any]:
        """Detect specific patterns in code"""
        file_path = args["file_path"]
        pattern_types = args.get("pattern_types", ["security", "performance"])

        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Perform deep analysis to catch all patterns
        result = await self.engine.analyze_file(file_path, AnalysisLevel.DEEP)

        # Filter issues by requested pattern types
        filtered_issues = []
        for issue in result.issues:
            for pattern_type in pattern_types:
                if pattern_type.lower() in issue.category.lower():
                    filtered_issues.append(issue)
                    break

        return {
            "file_path": file_path,
            "pattern_types": pattern_types,
            "patterns_found": len(filtered_issues),
            "patterns": [
                {
                    "category": issue.category,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "line_number": issue.line_number,
                    "suggestions": issue.suggestions
                }
                for issue in filtered_issues
            ]
        }


def create_ability() -> DeepCodeAnalysisAbility:
    """Factory function to create the ability"""
    return DeepCodeAnalysisAbility()
