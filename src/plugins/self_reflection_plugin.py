#!/usr/bin/env python3
"""
SelfReflectionPlugin for Super Alita AI Agent

This plugin provides concrete introspection capabilities to enable the agent
to analyze its own abilities, identify capability gaps, and provide system
status information. This breaks the cognitive deadlock of recursive self-analysis
by providing concrete, actionable tools for introspection.

Key Features:
- List all available tools and capabilities
- Analyze capability gaps based on requested functionality
- Provide system health and status information
- Break recursive thought loops with concrete actions

Architecture Integration:
- Works with the Code Skeleton Architecture
- Compatible with AST-based parsing in LLMPlannerPlugin
- Follows the plugin interface pattern
- Emits proper ToolResultEvent responses

Version: 1.0.0
Author: Super Alita Agent
"""

import logging
from datetime import UTC, datetime
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import ToolCallEvent, ToolResultEvent
from src.core.neural_atom import NeuralStore
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class SelfReflectionPlugin(PluginInterface):
    """
    Provides concrete introspection capabilities to break cognitive deadlocks.

    This plugin transforms abstract requests like "analyze your capabilities"
    into concrete, executable operations that return actionable data. It serves
    as the foundational tool for self-awareness and capability assessment.

    Core Operations:
    1. list_capabilities: Enumerate all available tools and plugins
    2. analyze_gaps: Identify missing capabilities for specific requests
    3. system_status: Provide overall system health information
    4. capability_match: Assess if current tools can handle a given task

    Cognitive Deadlock Resolution:
    Instead of recursive loops trying to define introspection, this plugin
    provides concrete implementations that return structured data about
    the agent's state and capabilities.
    """

    def __init__(self):
        super().__init__()
        self._orchestrator_instance: Any | None = None
        self._stats = {
            "introspection_requests": 0,
            "capability_analyses": 0,
            "gap_analyses": 0,
            "system_checks": 0,
            "last_activity": None,
        }

    @property
    def name(self) -> str:
        return "self_reflection"

    @property
    def version(self) -> str:
        return "1.0.0"

    async def setup(
        self, event_bus: EventBus, store: NeuralStore, config: dict[str, Any]
    ):
        """Setup the plugin with access to the main orchestrator."""
        await super().setup(event_bus, store, config)

        # Critical: Get reference to orchestrator to access plugin registry
        plugin_config = config.get(self.name, {})
        self._orchestrator_instance = plugin_config.get("orchestrator_instance")

        if not self._orchestrator_instance:
            logger.warning(
                "SelfReflectionPlugin: No orchestrator instance provided. "
                "Self-reflection capabilities will be limited."
            )

        logger.info("SelfReflectionPlugin setup complete")

    async def start(self) -> None:
        """Start the plugin and subscribe to tool call events."""
        await super().start()

        # Subscribe to tool calls for this plugin
        await self.subscribe("tool_call", self._handle_tool_call)

        logger.info("SelfReflectionPlugin started successfully")

    async def shutdown(self) -> None:
        """Clean shutdown of the plugin."""
        logger.info(f"SelfReflectionPlugin shutting down. Final stats: {self._stats}")
        self._is_running = False

    async def _handle_tool_call(self, event: ToolCallEvent) -> None:
        """
        Handle tool calls for self-reflection operations.

        This is the core method that processes introspection requests and
        transforms them into concrete analysis results.
        """
        # Only handle calls intended for this plugin
        if event.tool_name != self.name:
            return

        logger.info(f"Processing self-reflection request: {event.parameters}")
        self._stats["introspection_requests"] += 1

        success = False
        result = {}

        try:
            operation = event.parameters.get("operation", "list_capabilities")

            if operation == "list_capabilities":
                result = await self._list_capabilities()
                success = True
                self._stats["capability_analyses"] += 1

            elif operation == "analyze_gaps":
                result = await self._analyze_gaps(event.parameters)
                success = True
                self._stats["gap_analyses"] += 1

            elif operation == "system_status":
                result = await self._get_system_status()
                success = True
                self._stats["system_checks"] += 1

            elif operation == "capability_match":
                result = await self._assess_capability_match(event.parameters)
                success = True

            else:
                result = {
                    "error": f"Unknown self-reflection operation: {operation}",
                    "available_operations": [
                        "list_capabilities",
                        "analyze_gaps",
                        "system_status",
                        "capability_match",
                    ],
                    "summary": f"Operation '{operation}' is not supported",
                }
                success = False

        except Exception as e:
            logger.error(f"Error during self-reflection: {e}", exc_info=True)
            result = {
                "error": f"Self-reflection failed: {e!s}",
                "summary": "An error occurred while analyzing capabilities",
            }
            success = False

        # Update activity tracking
        self._stats["last_activity"] = datetime.now(UTC).isoformat()

        # Publish the result event
        await self.event_bus.publish(
            ToolResultEvent(
                source_plugin=self.name,
                tool_call_id=event.tool_call_id,
                session_id=event.session_id,
                conversation_id=event.conversation_id,
                success=success,
                result=result,
            )
        )

    async def _list_capabilities(self) -> dict[str, Any]:
        """
        List all available tools and their descriptions.

        This provides concrete data about the agent's current capabilities,
        breaking the recursive loop of trying to analyze capabilities without
        knowing what they are.
        """
        if not self._orchestrator_instance:
            return {
                "summary": "Unable to access orchestrator for capability analysis",
                "error": "No orchestrator instance available",
                "capabilities": [],
            }

        try:
            available_tools = []

            # Get all active plugins from orchestrator
            if hasattr(self._orchestrator_instance, "plugins"):
                plugins = self._orchestrator_instance.plugins

                for name, plugin in plugins.items():
                    # Filter out internal/private plugins
                    if not name.startswith("_") and plugin is not None:
                        tool_info = {
                            "name": name,
                            "type": type(plugin).__name__,
                            "status": "active",
                            "version": getattr(plugin, "version", "unknown"),
                        }

                        # Add description from docstring if available
                        if hasattr(plugin, "__doc__") and plugin.__doc__:
                            # Get first line of docstring as description
                            doc_lines = plugin.__doc__.strip().split("\n")
                            tool_info["description"] = doc_lines[0].strip()

                        # Add specific capabilities if the plugin has them
                        if hasattr(plugin, "get_capabilities"):
                            try:
                                tool_info[
                                    "capabilities"
                                ] = await plugin.get_capabilities()
                            except Exception as e:
                                logger.debug(
                                    f"Could not get capabilities from {name}: {e}"
                                )

                        available_tools.append(tool_info)

            # Get additional tool information from LLM planner if available
            llm_planner = plugins.get("llm_planner")
            if llm_planner and hasattr(llm_planner, "_tool_definitions"):
                for tool_name, tool_def in llm_planner._tool_definitions.items():
                    # Enhance existing tool info or add new entries
                    existing_tool = next(
                        (t for t in available_tools if t["name"] == tool_name), None
                    )
                    if existing_tool:
                        existing_tool["llm_description"] = tool_def.get(
                            "description", ""
                        )
                        existing_tool["examples"] = tool_def.get("examples", [])
                    else:
                        # Add tools that might be defined in LLM planner but not as plugins
                        available_tools.append(
                            {
                                "name": tool_name,
                                "type": "LLM-Managed Tool",
                                "status": "available",
                                "description": tool_def.get("description", ""),
                                "examples": tool_def.get("examples", []),
                            }
                        )

            return {
                "summary": f"Found {len(available_tools)} available tools and capabilities",
                "capabilities": available_tools,
                "total_count": len(available_tools),
                "analysis_timestamp": self._get_timestamp(),
                "orchestrator_status": "available",
            }

        except Exception as e:
            logger.error(f"Error listing capabilities: {e}", exc_info=True)
            return {
                "summary": "Error analyzing capabilities",
                "error": str(e),
                "capabilities": [],
            }

    async def _analyze_gaps(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze potential capability gaps based on requested functionality.

        This transforms vague requests like "can you do X?" into concrete
        gap analysis with actionable recommendations.
        """
        requested_capability = parameters.get("requested_capability", "")
        parameters.get("detailed", False)

        # Get current capabilities first
        current_caps = await self._list_capabilities()

        if "error" in current_caps:
            return current_caps

        # Extract tool names and descriptions for analysis
        current_tools = {
            tool["name"]: tool.get("description", "").lower()
            for tool in current_caps["capabilities"]
        }

        # Perform gap analysis using heuristics
        gap_analysis = {
            "requested_capability": requested_capability,
            "current_tools": list(current_tools.keys()),
            "gap_identified": False,
            "suggested_tool_type": None,
            "reasoning": "",
            "confidence": 0.0,
            "recommendations": [],
        }

        # Analyze the request against current capabilities
        request_lower = requested_capability.lower()

        # File processing analysis
        if any(
            keyword in request_lower
            for keyword in ["file", "document", "pdf", "csv", "excel"]
        ):
            file_tools = [
                name
                for name, desc in current_tools.items()
                if any(word in desc for word in ["file", "document", "pdf"])
            ]
            if not file_tools:
                gap_analysis.update(
                    {
                        "gap_identified": True,
                        "suggested_tool_type": "file_processor",
                        "reasoning": "No file processing capabilities detected",
                        "confidence": 0.8,
                        "recommendations": [
                            "Create a file processing tool",
                            "Add document parsing capabilities",
                        ],
                    }
                )

        # Mathematical analysis
        elif any(
            keyword in request_lower
            for keyword in ["calculate", "math", "compute", "formula"]
        ):
            calc_tools = [
                name
                for name, desc in current_tools.items()
                if any(word in desc for word in ["calc", "math", "compute"])
            ]
            if not calc_tools:
                gap_analysis.update(
                    {
                        "gap_identified": True,
                        "suggested_tool_type": "calculator",
                        "reasoning": "No mathematical calculation capabilities detected",
                        "confidence": 0.9,
                        "recommendations": [
                            "Create a calculator tool",
                            "Add mathematical computation capabilities",
                        ],
                    }
                )

        # Image processing analysis
        elif any(
            keyword in request_lower
            for keyword in ["image", "visual", "picture", "photo"]
        ):
            image_tools = [
                name
                for name, desc in current_tools.items()
                if any(word in desc for word in ["image", "visual", "picture"])
            ]
            if not image_tools:
                gap_analysis.update(
                    {
                        "gap_identified": True,
                        "suggested_tool_type": "image_processor",
                        "reasoning": "No image processing capabilities detected",
                        "confidence": 0.85,
                        "recommendations": [
                            "Create an image processing tool",
                            "Add computer vision capabilities",
                        ],
                    }
                )

        # Code analysis
        elif any(
            keyword in request_lower
            for keyword in ["code", "program", "script", "debug"]
        ):
            code_tools = [
                name
                for name, desc in current_tools.items()
                if any(word in desc for word in ["code", "program", "git"])
            ]
            if not code_tools:
                gap_analysis.update(
                    {
                        "gap_identified": True,
                        "suggested_tool_type": "code_analyzer",
                        "reasoning": "No code analysis capabilities detected",
                        "confidence": 0.7,
                        "recommendations": [
                            "Create a code analysis tool",
                            "Add programming language support",
                        ],
                    }
                )

        # If no specific gap identified, check if any existing tools might handle it
        if not gap_analysis["gap_identified"]:
            potential_matches = []
            for tool_name, desc in current_tools.items():
                # Simple keyword matching
                common_words = set(request_lower.split()) & set(desc.split())
                if common_words:
                    potential_matches.append(
                        {
                            "tool": tool_name,
                            "match_keywords": list(common_words),
                            "confidence": len(common_words)
                            / max(len(request_lower.split()), 1),
                        }
                    )

            if potential_matches:
                gap_analysis.update(
                    {
                        "gap_identified": False,
                        "reasoning": "Existing tools may handle this request",
                        "confidence": max(
                            match["confidence"] for match in potential_matches
                        ),
                        "potential_matches": potential_matches,
                    }
                )
            else:
                gap_analysis.update(
                    {
                        "gap_identified": True,
                        "suggested_tool_type": "custom_tool",
                        "reasoning": "No existing tools appear to match the requested capability",
                        "confidence": 0.6,
                        "recommendations": [
                            "Create a custom tool for this specific request"
                        ],
                    }
                )

        return {
            "summary": f"Gap analysis complete. Gap identified: {gap_analysis['gap_identified']}",
            "analysis": gap_analysis,
            "analysis_timestamp": self._get_timestamp(),
            "total_tools_analyzed": len(current_tools),
        }

    async def _assess_capability_match(
        self, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Assess if current tools can handle a specific task.

        This provides a direct answer to "can you do X?" questions.
        """
        task_description = parameters.get("task_description", "")

        if not task_description:
            return {
                "summary": "No task description provided for capability assessment",
                "error": "task_description parameter required",
            }

        # Get current capabilities
        current_caps = await self._list_capabilities()
        if "error" in current_caps:
            return current_caps

        # Analyze task against available tools
        tools = current_caps["capabilities"]
        matches = []

        task_lower = task_description.lower()
        task_words = set(task_lower.split())

        for tool in tools:
            tool_desc = tool.get("description", "").lower()
            tool_examples = tool.get("examples", [])

            # Calculate match score
            desc_words = set(tool_desc.split())
            common_words = task_words & desc_words

            # Check examples for better matching
            example_matches = 0
            if tool_examples:
                for example in tool_examples:
                    example_words = set(example.lower().split())
                    if task_words & example_words:
                        example_matches += 1

            if common_words or example_matches:
                match_score = (len(common_words) * 0.7 + example_matches * 0.3) / max(
                    len(task_words), 1
                )
                matches.append(
                    {
                        "tool_name": tool["name"],
                        "match_score": round(match_score, 3),
                        "matching_keywords": list(common_words),
                        "example_matches": example_matches,
                        "description": tool.get("description", ""),
                    }
                )

        # Sort by match score
        matches.sort(key=lambda x: x["match_score"], reverse=True)

        # Determine if task can be handled
        best_match = matches[0] if matches else None
        can_handle = best_match and best_match["match_score"] > 0.3

        return {
            "summary": f"Task can be handled: {can_handle}",
            "task_description": task_description,
            "can_handle": can_handle,
            "best_match": best_match,
            "all_matches": matches[:5],  # Top 5 matches
            "confidence": best_match["match_score"] if best_match else 0.0,
            "analysis_timestamp": self._get_timestamp(),
        }

    async def _get_system_status(self) -> dict[str, Any]:
        """
        Get overall system health and status information.

        This provides concrete data about the agent's operational state.
        """
        status = {
            "orchestrator_available": self._orchestrator_instance is not None,
            "event_bus_active": self.event_bus is not None,
            "store_available": self.store is not None,
            "plugin_status": "active",
            "plugin_stats": self._stats.copy(),
            "timestamp": self._get_timestamp(),
        }

        # Get detailed plugin information
        if self._orchestrator_instance and hasattr(
            self._orchestrator_instance, "plugins"
        ):
            plugins = self._orchestrator_instance.plugins
            status.update(
                {
                    "total_plugins": len(plugins),
                    "active_plugins": len(
                        [p for p in plugins.values() if p is not None]
                    ),
                    "plugin_names": list(plugins.keys()),
                }
            )

            # Check health of critical plugins
            critical_plugins = ["llm_planner", "memory_manager", "conversation"]
            status["critical_plugins_status"] = {}

            for plugin_name in critical_plugins:
                plugin = plugins.get(plugin_name)
                if plugin:
                    # Check if plugin has health check method
                    if hasattr(plugin, "health_check"):
                        try:
                            health = await plugin.health_check()
                            status["critical_plugins_status"][plugin_name] = health
                        except Exception as e:
                            status["critical_plugins_status"][plugin_name] = {
                                "status": "error",
                                "error": str(e),
                            }
                    else:
                        status["critical_plugins_status"][plugin_name] = {
                            "status": "active",
                            "health_check": "not_available",
                        }
                else:
                    status["critical_plugins_status"][plugin_name] = {
                        "status": "not_found"
                    }

        return {
            "summary": "System status check complete",
            "status": status,
            "overall_health": self._assess_overall_health(status),
        }

    def _assess_overall_health(self, status: dict[str, Any]) -> str:
        """Assess overall system health based on status data."""
        issues = []

        if not status["orchestrator_available"]:
            issues.append("orchestrator_unavailable")
        if not status["event_bus_active"]:
            issues.append("event_bus_inactive")
        if not status["store_available"]:
            issues.append("store_unavailable")

        # Check critical plugins
        critical_status = status.get("critical_plugins_status", {})
        for plugin_name, plugin_status in critical_status.items():
            if plugin_status.get("status") not in ["active", "healthy"]:
                issues.append(f"critical_plugin_{plugin_name}_unhealthy")

        if not issues:
            return "healthy"
        if len(issues) <= 2:
            return "degraded"
        return "unhealthy"

    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis records."""
        return datetime.now(UTC).isoformat()

    async def health_check(self) -> dict[str, Any]:
        """Health check for the self-reflection plugin."""
        health = {
            "status": "healthy" if self.is_running else "stopped",
            "version": self.version,
            "issues": [],
            "stats": self._stats.copy(),
        }

        # Check orchestrator availability
        if not self._orchestrator_instance:
            health["status"] = "degraded"
            health["issues"].append("orchestrator_instance_not_available")

        # Check if we can perform basic operations
        try:
            # Test basic capability listing
            caps = await self._list_capabilities()
            if "error" in caps:
                health["issues"].append("capability_listing_failed")
            else:
                health["capability_count"] = caps.get("total_count", 0)
        except Exception as e:
            health["status"] = "unhealthy"
            health["issues"].append(f"capability_test_failed: {e}")

        return health
