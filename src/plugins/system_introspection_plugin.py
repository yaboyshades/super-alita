#!/usr/bin/env python3
"""
System Introspection Plugin for Super Alita

This plugin provides comprehensive system status, capability assessment,
and introspective diagnostics to address the cognitive limitations identified
in user analysis. It can assess the current state of all subsystems,
identify what's working and what needs fixing, and provide actionable
diagnostic information.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

from src.core.events import SystemStatusResponseEvent
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class SystemIntrospectionPlugin(PluginInterface):
    """
    Plugin that provides deep system introspection and self-assessment capabilities.

    Key features:
    - Comprehensive subsystem health checks
    - Capability mapping and limitation identification
    - Configuration analysis and recommendations
    - Real-time system diagnostics
    - Actionable troubleshooting guidance
    """

    def __init__(self):
        super().__init__()
        self.subsystem_status = {}
        self.last_health_check = None
        self.diagnostic_cache = {}

    @property
    def name(self) -> str:
        return "system_introspection"

    async def setup(self, event_bus, store, config: Dict[str, Any]) -> None:
        """Initialize the system introspection plugin."""
        await super().setup(event_bus, store, config)

        self.enable_detailed_diagnostics = config.get(
            "enable_detailed_diagnostics", True
        )
        self.health_check_interval = config.get("health_check_interval_seconds", 60)

        logger.info("System Introspection plugin setup complete")

    async def start(self) -> None:
        """Start the system introspection plugin."""
        await super().start()

        # Subscribe to system status requests
        await self.subscribe("system_status_request", self._handle_status_request)
        await self.subscribe("conversation_message", self._handle_introspection_queries)

        # Start periodic health monitoring
        if self.enable_detailed_diagnostics:
            self.add_task(self._periodic_health_check())

        logger.info(
            "System Introspection plugin started - ready to provide system diagnostics"
        )

    async def shutdown(self) -> None:
        """Shutdown the system introspection plugin."""
        logger.info("Shutting down System Introspection plugin")

    async def _periodic_health_check(self) -> None:
        """Perform periodic health checks of all subsystems."""
        while self.is_running:
            try:
                await self._comprehensive_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _handle_status_request(self, event) -> None:
        """Handle explicit system status requests."""
        try:
            data = event.model_dump() if hasattr(event, "model_dump") else event
            session_id = data.get("session_id")

            # Perform comprehensive diagnosis
            status_report = await self._comprehensive_system_diagnosis()

            # Emit detailed status response
            response_event = SystemStatusResponseEvent(
                source_plugin=self.name,
                session_id=session_id,
                status_report=status_report,
                timestamp=datetime.now().isoformat(),
            )

            await self.event_bus.publish(response_event)
            logger.info("Published comprehensive system status response")

        except Exception as e:
            logger.error(f"Error handling status request: {e}")

    async def _handle_introspection_queries(self, event) -> None:
        """Handle conversation messages that ask for system introspection."""
        try:
            data = event.model_dump() if hasattr(event, "model_dump") else event
            user_message = data.get("user_message", "").lower()
            data.get("session_id")

            # Check if this is an introspection query
            introspection_keywords = [
                "assess",
                "limitations",
                "what needs",
                "fix",
                "status",
                "diagnose",
                "capabilities",
                "working",
                "broken",
                "help",
                "system status",
                "what can you do",
                "what works",
            ]

            if any(keyword in user_message for keyword in introspection_keywords):
                # Provide detailed system analysis
                diagnosis = await self._comprehensive_system_diagnosis()
                response_text = self._format_user_friendly_diagnosis(
                    diagnosis, user_message
                )

                # Send response via agent_reply channel
                await self.event_bus._redis.publish(
                    "agent_reply", json.dumps({"text": response_text})
                )

                logger.info("Provided introspection response to user query")

        except Exception as e:
            logger.error(f"Error handling introspection query: {e}")

    async def _comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all subsystems."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "subsystems": {},
            "issues": [],
            "recommendations": [],
        }

        try:
            # Check EventBus
            eventbus_status = await self._check_eventbus_health()
            health_status["subsystems"]["event_bus"] = eventbus_status

            # Check NeuralStore
            store_status = await self._check_neural_store_health()
            health_status["subsystems"]["neural_store"] = store_status

            # Check Redis connection
            redis_status = await self._check_redis_health()
            health_status["subsystems"]["redis"] = redis_status

            # Check plugin status
            plugin_status = await self._check_plugin_health()
            health_status["subsystems"]["plugins"] = plugin_status

            # Check memory system
            memory_status = await self._check_memory_system_health()
            health_status["subsystems"]["memory_system"] = memory_status

            # Check LLM integration
            llm_status = await self._check_llm_health()
            health_status["subsystems"]["llm_integration"] = llm_status

            # Determine overall status
            all_statuses = [
                subsystem.get("status", "unknown")
                for subsystem in health_status["subsystems"].values()
            ]

            if all(status == "healthy" for status in all_statuses):
                health_status["overall_status"] = "healthy"
            elif any(status == "critical" for status in all_statuses):
                health_status["overall_status"] = "critical"
            else:
                health_status["overall_status"] = "degraded"

            # Collect issues and recommendations
            for subsystem, details in health_status["subsystems"].items():
                if details.get("issues"):
                    health_status["issues"].extend(
                        [f"{subsystem}: {issue}" for issue in details["issues"]]
                    )
                if details.get("recommendations"):
                    health_status["recommendations"].extend(
                        [f"{subsystem}: {rec}" for rec in details["recommendations"]]
                    )

            self.last_health_check = health_status
            return health_status

        except Exception as e:
            logger.error(f"Error in comprehensive health check: {e}")
            health_status["overall_status"] = "error"
            health_status["issues"].append(f"Health check failed: {e}")
            return health_status

    async def _check_eventbus_health(self) -> Dict[str, Any]:
        """Check EventBus health and connectivity."""
        status = {
            "status": "unknown",
            "details": {},
            "issues": [],
            "recommendations": [],
        }

        try:
            if self.event_bus and hasattr(self.event_bus, "is_running"):
                if self.event_bus.is_running:
                    status["status"] = "healthy"
                    status["details"]["running"] = True
                else:
                    status["status"] = "critical"
                    status["issues"].append("EventBus not running")
                    status["recommendations"].append("Restart EventBus service")
            else:
                status["status"] = "critical"
                status["issues"].append("EventBus not available")
                status["recommendations"].append("Initialize EventBus")

        except Exception as e:
            status["status"] = "error"
            status["issues"].append(f"EventBus check failed: {e}")

        return status

    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        status = {
            "status": "unknown",
            "details": {},
            "issues": [],
            "recommendations": [],
        }

        try:
            if self.event_bus and hasattr(self.event_bus, "_redis"):
                # Test Redis ping
                try:
                    await self.event_bus._redis.ping()
                    status["status"] = "healthy"
                    status["details"]["ping_successful"] = True
                except Exception as e:
                    status["status"] = "critical"
                    status["issues"].append(f"Redis ping failed: {e}")
                    status["recommendations"].append(
                        "Check Redis/Memurai service is running on localhost:6379"
                    )
            else:
                status["status"] = "critical"
                status["issues"].append("Redis connection not available")
                status["recommendations"].append(
                    "Ensure Redis is configured in EventBus"
                )

        except Exception as e:
            status["status"] = "error"
            status["issues"].append(f"Redis check failed: {e}")

        return status

    async def _check_neural_store_health(self) -> Dict[str, Any]:
        """Check NeuralStore health and statistics."""
        status = {
            "status": "unknown",
            "details": {},
            "issues": [],
            "recommendations": [],
        }

        try:
            if self.store:
                # Get basic statistics
                stats = (
                    self.store.get_stats() if hasattr(self.store, "get_stats") else {}
                )
                status["details"]["statistics"] = stats

                # Check if store is functional
                if hasattr(self.store, "attention"):
                    status["status"] = "healthy"
                    status["details"]["attention_available"] = True
                else:
                    status["status"] = "degraded"
                    status["issues"].append(
                        "NeuralStore attention mechanism not available"
                    )
                    status["recommendations"].append(
                        "Verify NeuralStore implementation includes attention method"
                    )
            else:
                status["status"] = "critical"
                status["issues"].append("NeuralStore not available")
                status["recommendations"].append("Initialize NeuralStore")

        except Exception as e:
            status["status"] = "error"
            status["issues"].append(f"NeuralStore check failed: {e}")

        return status

    async def _check_memory_system_health(self) -> Dict[str, Any]:
        """Check semantic memory system health."""
        status = {
            "status": "unknown",
            "details": {},
            "issues": [],
            "recommendations": [],
        }

        try:
            # Check if memory plugin is available through event bus or direct access
            # This is a simplified check - in a real system we'd query the semantic memory plugin

            if hasattr(self.store, "embed_query") and hasattr(self.store, "attention"):
                status["status"] = "healthy"
                status["details"]["embedding_available"] = True
                status["details"]["attention_available"] = True
            else:
                status["status"] = "degraded"
                status["issues"].append("Memory embedding or attention not available")
                status["recommendations"].append(
                    "Enable semantic_memory plugin in config"
                )
                status["recommendations"].append(
                    "Ensure Gemini API key is set for embeddings"
                )

        except Exception as e:
            status["status"] = "error"
            status["issues"].append(f"Memory system check failed: {e}")

        return status

    async def _check_llm_health(self) -> Dict[str, Any]:
        """Check LLM integration health."""
        status = {
            "status": "unknown",
            "details": {},
            "issues": [],
            "recommendations": [],
        }

        try:
            # Check if Gemini API key is available
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if gemini_key:
                status["details"]["api_key_present"] = True

                # Try to import and configure Gemini
                try:
                    try:
                        import google.generativeai as genai
                    except ImportError:
                        genai = None

                    if genai:
                        genai.configure(api_key=gemini_key)
                        status["status"] = "healthy"
                        status["details"]["gemini_configured"] = True
                    else:
                        status["status"] = "degraded"
                        status["issues"].append(
                            "google.generativeai package not installed"
                        )
                        status["recommendations"].append(
                            "Install google-generativeai package"
                        )
                except Exception as e:
                    status["status"] = "degraded"
                    status["issues"].append(f"Gemini configuration failed: {e}")
                    status["recommendations"].append("Check Gemini API key validity")
            else:
                status["status"] = "critical"
                status["issues"].append("GEMINI_API_KEY environment variable not set")
                status["recommendations"].append(
                    "Set GEMINI_API_KEY environment variable"
                )
                status["recommendations"].append(
                    "Use: $Env:GEMINI_API_KEY='your-key-here'"
                )

        except Exception as e:
            status["status"] = "error"
            status["issues"].append(f"LLM check failed: {e}")

        return status

    async def _check_plugin_health(self) -> Dict[str, Any]:
        """Check plugin system health."""
        status = {
            "status": "healthy",
            "details": {"total_plugins": 0, "running_plugins": 0},
            "issues": [],
            "recommendations": [],
        }

        try:
            # This would need integration with the main agent to get actual plugin status
            # For now, provide basic health indicator
            status["details"]["introspection_plugin_running"] = self.is_running

        except Exception as e:
            status["status"] = "error"
            status["issues"].append(f"Plugin check failed: {e}")

        return status

    async def _comprehensive_system_diagnosis(self) -> Dict[str, Any]:
        """Perform comprehensive system diagnosis."""
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_health": "unknown",
                "critical_issues": [],
                "working_features": [],
                "degraded_features": [],
                "recommendations": [],
            },
            "detailed_status": {},
            "environment": {},
            "configuration": {},
        }

        try:
            # Get health check results
            health_results = await self._comprehensive_health_check()
            diagnosis["detailed_status"] = health_results

            # Analyze environment
            diagnosis["environment"] = {
                "os": "Windows",
                "python_version": "3.x",
                "gemini_api_key_set": bool(os.environ.get("GEMINI_API_KEY")),
                "working_directory": os.getcwd(),
                "redis_expected": "localhost:6379",
            }

            # Analyze configuration
            diagnosis["configuration"] = await self._analyze_configuration()

            # Generate summary
            diagnosis["summary"] = self._generate_diagnostic_summary(health_results)

            return diagnosis

        except Exception as e:
            logger.error(f"Error in comprehensive diagnosis: {e}")
            diagnosis["summary"]["overall_health"] = "error"
            diagnosis["summary"]["critical_issues"].append(f"Diagnosis failed: {e}")
            return diagnosis

    async def _analyze_configuration(self) -> Dict[str, Any]:
        """Analyze system configuration for issues."""
        config_analysis = {
            "agent_yaml_status": "unknown",
            "plugin_configuration": {},
            "issues": [],
            "recommendations": [],
        }

        try:
            # Check if agent.yaml exists and is readable
            config_path = "src/config/agent.yaml"
            if os.path.exists(config_path):
                config_analysis["agent_yaml_status"] = "present"

                # Read and analyze config
                try:
                    import yaml

                    with open(config_path) as f:
                        config_data = yaml.safe_load(f)

                    # Check plugin configurations
                    plugins_config = config_data.get("plugins", {})
                    for plugin_name, plugin_config in plugins_config.items():
                        enabled = plugin_config.get("enabled", False)
                        config_analysis["plugin_configuration"][plugin_name] = {
                            "enabled": enabled,
                            "has_config": len(plugin_config) > 1,
                        }

                        # Check for common issues
                        if plugin_name == "semantic_memory" and enabled:
                            if not plugin_config.get("gemini_api_key"):
                                config_analysis["issues"].append(
                                    f"{plugin_name}: Missing API key configuration"
                                )

                        if plugin_name == "conversation" and enabled:
                            if not plugin_config.get("gemini_api_key"):
                                config_analysis["issues"].append(
                                    f"{plugin_name}: Missing API key configuration"
                                )

                except Exception as e:
                    config_analysis["issues"].append(f"Failed to parse agent.yaml: {e}")
            else:
                config_analysis["agent_yaml_status"] = "missing"
                config_analysis["issues"].append(
                    "agent.yaml configuration file not found"
                )

        except Exception as e:
            config_analysis["issues"].append(f"Configuration analysis failed: {e}")

        return config_analysis

    def _generate_diagnostic_summary(
        self, health_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a human-readable diagnostic summary."""
        summary = {
            "overall_health": health_results.get("overall_status", "unknown"),
            "critical_issues": [],
            "working_features": [],
            "degraded_features": [],
            "recommendations": [],
        }

        # Analyze subsystem statuses
        for subsystem, status_info in health_results.get("subsystems", {}).items():
            status = status_info.get("status", "unknown")

            if status == "healthy":
                summary["working_features"].append(
                    f"✓ {subsystem.replace('_', ' ').title()}"
                )
            elif status == "critical":
                summary["critical_issues"].append(
                    f"✗ {subsystem.replace('_', ' ').title()}"
                )
            elif status == "degraded":
                summary["degraded_features"].append(
                    f"⚠ {subsystem.replace('_', ' ').title()}"
                )

        # Collect all recommendations
        summary["recommendations"] = health_results.get("recommendations", [])

        return summary

    def _format_user_friendly_diagnosis(
        self, diagnosis: Dict[str, Any], user_query: str
    ) -> str:
        """Format diagnosis results for user consumption."""
        summary = diagnosis.get("summary", {})
        overall_health = summary.get("overall_health", "unknown")

        # Create user-friendly response
        response_parts = []

        # Header
        response_parts.append("## 🔍 Super Alita System Diagnosis")
        response_parts.append(f"**Overall Status**: {overall_health.upper()}")
        response_parts.append("")

        # Working features
        working_features = summary.get("working_features", [])
        if working_features:
            response_parts.append("### ✅ **What's Working**")
            for feature in working_features:
                response_parts.append(f"  {feature}")
            response_parts.append("")

        # Critical issues
        critical_issues = summary.get("critical_issues", [])
        if critical_issues:
            response_parts.append("### ❌ **Critical Issues**")
            for issue in critical_issues:
                response_parts.append(f"  {issue}")
            response_parts.append("")

        # Degraded features
        degraded_features = summary.get("degraded_features", [])
        if degraded_features:
            response_parts.append("### ⚠️ **Partially Working**")
            for feature in degraded_features:
                response_parts.append(f"  {feature}")
            response_parts.append("")

        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            response_parts.append("### 🔧 **How to Fix**")
            for i, rec in enumerate(recommendations[:5], 1):  # Limit to top 5
                response_parts.append(f"  {i}. {rec}")
            response_parts.append("")

        # Environment info
        env_info = diagnosis.get("environment", {})
        if env_info:
            response_parts.append("### 🌐 **Environment**")
            response_parts.append(
                f"  • API Key Set: {'Yes' if env_info.get('gemini_api_key_set') else 'No'}"
            )
            response_parts.append(
                f"  • Redis Expected: {env_info.get('redis_expected', 'Unknown')}"
            )
            response_parts.append("")

        # Actionable next steps
        response_parts.append("### 🎯 **Next Steps**")

        if "limitations" in user_query or "assess" in user_query:
            response_parts.append("Based on your request for limitation assessment:")
            response_parts.append("")

            if not env_info.get("gemini_api_key_set"):
                response_parts.append("**Priority 1**: Set your Gemini API key")
                response_parts.append("  ```")
                response_parts.append("  $Env:GEMINI_API_KEY='your-api-key-here'")
                response_parts.append("  ```")
                response_parts.append("")

            if "redis" in [issue.lower() for issue in critical_issues]:
                response_parts.append("**Priority 2**: Start Redis/Memurai service")
                response_parts.append("  Check that Redis is running on localhost:6379")
                response_parts.append("")

            if not working_features:
                response_parts.append("**Priority 3**: Initialize core systems")
                response_parts.append(
                    "  Restart the agent with: `python launch_super_alita.py`"
                )
                response_parts.append("")

        response_parts.append("Type 'system status' anytime for an updated diagnosis.")

        return "\n".join(response_parts)
