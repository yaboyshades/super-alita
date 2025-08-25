"""
Super Alita Integrated Prompt System Manager
This module provides a unified interface for loading and managing the hierarchical
prompt system, including core agent prompts, co-architect prompts, and plugin-specific
prompts with REUG framework integration.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Represents a single prompt template with metadata."""

    template: str
    version: str
    description: str
    placeholders: list[str]


@dataclass
class PromptMetadata:
    """Metadata for prompt templates and system prompts."""

    version: str
    description: str
    usage: str
    file_path: str | None = None
    last_updated: str | None = None


class PromptManager:
    """
    Manages hierarchical JSON prompts and integrated system prompts for Super Alita.
    Features:
    - Load prompts from JSON configuration and markdown files
    - Template substitution with dynamic data
    - Version tracking and validation
    - REUG framework integration
    - Multi-tiered prompt system (Core Agent, Co-Architect, Plugin-specific)
    """

    def __init__(self, prompts_file: str = "src/config/prompts.json"):
        self.prompts_file = Path(prompts_file)
        self.config_root = self.prompts_file.parent
        self.prompts: dict[str, Any] = {}
        self.version: str = "unknown"
        self.prompt_cache: dict[str, str] = {}
        # Load both legacy and integrated prompt systems
        self._load_prompts()
        self._load_integrated_config()

    def _load_prompts(self) -> None:
        """Load prompts from JSON file."""
        try:
            if not self.prompts_file.exists():
                logger.error(f"Prompts file not found: {self.prompts_file}")
                return
            with open(self.prompts_file, encoding="utf-8") as f:
                data = json.load(f)
            self.prompts = data.get("prompts", {})
            self.version = data.get("version", "unknown")
            self.response_schemas = data.get("response_schemas", {})
            self.placeholders = data.get("placeholders", {})
            logger.info(f"📝 Loaded prompts v{self.version} from {self.prompts_file}")
            logger.info(f"📂 Available prompt categories: {list(self.prompts.keys())}")
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            self.prompts = {}

    def get_prompt(self, path: str, **kwargs) -> str:
        """
        Get a formatted prompt by path with template substitution.
        Args:
            path: Dot-separated path to prompt (e.g., "planner.main_routing")
            **kwargs: Values to substitute in template placeholders
        Returns:
            Formatted prompt string
        Example:
            prompt = manager.get_prompt(
                "planner.main_routing",
                user_message="Hello",
                tool_descriptions="..."
            )
        """
        try:
            # Navigate to the prompt using dot notation
            current = self.prompts
            path_parts = path.split(".")
            for part in path_parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    raise KeyError(f"Prompt path not found: {path}")
            # Extract template string
            if isinstance(current, dict):
                template_str = current.get("template", "")
                if not template_str:
                    raise ValueError(f"No template found at path: {path}")
            else:
                template_str = str(current)
            # Perform template substitution
            formatted_prompt = self._format_template(template_str, path, **kwargs)
            logger.debug(
                f"📝 Retrieved prompt: {path} (length: {len(formatted_prompt)})"
            )
            return formatted_prompt
        except Exception as e:
            logger.error(f"Error getting prompt '{path}': {e}")
            return f"Error loading prompt: {path}"

    def _format_template(self, template_str: str, path: str, **kwargs) -> str:
        """Format template string with provided values."""
        try:
            # Handle special formatting for structured data
            formatted_kwargs = {}
            for key, value in kwargs.items():
                if key == "tool_descriptions" and isinstance(value, dict):
                    # Format tool descriptions as bullet points
                    descriptions = []
                    for tool_name, description in value.items():
                        descriptions.append(f"- {tool_name}: {description}")
                    formatted_kwargs[key] = "\n".join(descriptions)
                elif key == "examples" and isinstance(value, list):
                    # Format examples as numbered list
                    formatted_kwargs[key] = "\n".join(
                        f"{i + 1}. {example}" for i, example in enumerate(value)
                    )
                elif key == "dependencies" and isinstance(value, list):
                    # Format dependencies as JSON array string
                    formatted_kwargs[key] = str(value)
                else:
                    formatted_kwargs[key] = str(value)
            # Use format() method for template substitution
            try:
                result = template_str.format(**formatted_kwargs)
            except KeyError as e:
                # Log missing keys and return partial substitution
                logger.warning(f"Missing template variable for {path}: {e}")
                # Use string replacement for safer partial substitution
                result = template_str
                for key, value in formatted_kwargs.items():
                    result = result.replace(f"{{{key}}}", str(value))
            return result
        except Exception as e:
            logger.error(f"Template formatting error for {path}: {e}")
            return template_str  # Return unformatted template as fallback

    def get_tool_descriptions(
        self, path: str = "planner.main_routing.tool_descriptions"
    ) -> dict[str, str]:
        """Get tool descriptions dictionary."""
        try:
            current = self.prompts
            for part in path.split("."):
                current = current[part]
            return current if isinstance(current, dict) else {}
        except (KeyError, TypeError):
            logger.warning(f"Tool descriptions not found at: {path}")
            return {}

    def get_examples(self, path: str) -> list[str]:
        """Get examples list for a prompt."""
        try:
            current = self.prompts
            for part in path.split("."):
                current = current[part]
            return current if isinstance(current, list) else []
        except (KeyError, TypeError):
            logger.warning(f"Examples not found at: {path}")
            return []

    def get_schema(self, schema_name: str) -> str:
        """Get a response schema by name."""
        return self.response_schemas.get(schema_name, "")

    def reload(self) -> None:
        """Reload prompts from file."""
        logger.info("🔄 Reloading prompts...")
        self._load_prompts()

    def get_info(self) -> dict[str, Any]:
        """Get information about loaded prompts."""
        info = {
            "version": self.version,
            "file": str(self.prompts_file),
            "categories": list(self.prompts.keys()),
            "total_prompts": 0,
        }

        # Count total prompts
        def count_prompts(obj):
            count = 0
            if isinstance(obj, dict):
                if "template" in obj:
                    count += 1
                else:
                    for value in obj.values():
                        count += count_prompts(value)
            return count

        info["total_prompts"] = count_prompts(self.prompts)
        return info

    def _load_integrated_config(self) -> None:
        """Load integrated prompt system configuration with REUG framework."""
        integrated_config_file = (
            self.config_root / "prompts" / "integrated_system_prompts.json"
        )
        try:
            if integrated_config_file.exists():
                with open(integrated_config_file, encoding="utf-8") as f:
                    self.integrated_config = json.load(f)
                logger.info(
                    f"📝 Loaded integrated prompt system v{self.integrated_config.get('version', 'unknown')}"
                )
            else:
                logger.info(
                    "📝 Integrated prompt system not found, using legacy prompts only"
                )
                self.integrated_config = {}
        except Exception as e:
            logger.error(f"❌ Failed to load integrated config: {e}")
            self.integrated_config = {}

    def get_core_agent_prompt(self, include_examples: bool = True) -> str:
        """
        Get the comprehensive core agent system prompt for LLMPlannerPlugin.
        Args:
            include_examples: Whether to include routing examples
        Returns:
            Complete system prompt with REUG framework and Sacred Laws
        """
        # Try to load from markdown file first
        core_prompt_file = self.config_root / "prompts" / "core_agent_system.md"
        if core_prompt_file.exists():
            with open(core_prompt_file, encoding="utf-8") as f:
                core_prompt = f.read()
        else:
            # Fallback to embedded prompt
            core_prompt = self._get_embedded_core_prompt()
        if include_examples:
            examples = self._get_routing_examples()
            core_prompt += f"\n\n## === 📚 ROUTING EXAMPLES ===\n\n{examples}"
        return core_prompt

    def get_plugin_prompt(self, plugin_name: str, operation: str | None = None) -> str:
        """
        Get specialized prompt for a specific plugin.
        Args:
            plugin_name: Name of the plugin (e.g., 'self_reflection', 'web_agent')
            operation: Specific operation if applicable
        Returns:
            Plugin-specific system prompt
        """
        if not self.integrated_config:
            return f"You are the {plugin_name} module of the Super Alita AI agent."
        plugin_specs = self.integrated_config.get("prompts", {}).get(
            "plugin_specializations", {}
        )
        plugin_config = plugin_specs.get(plugin_name, {})
        if not plugin_config:
            logger.warning(f"No specialized prompt found for plugin: {plugin_name}")
            return f"You are the {plugin_name} module of the Super Alita AI agent."
        base_prompt = plugin_config.get("system_prompt", "")
        # Add operation-specific context if provided
        if operation and "operations" in plugin_config:
            operations = plugin_config["operations"]
            if operation in operations:
                base_prompt += f"\n\nCurrent Operation: {operation}"
        # Add response format guidelines
        response_format = plugin_config.get("response_format", "")
        if response_format:
            base_prompt += f"\n\nResponse Format: {response_format}"
        return base_prompt

    def get_planner_routing_template(self, user_message: str) -> str:
        """
        Get the LLM routing template with user message and tool descriptions.
        Args:
            user_message: The user's input message
        Returns:
            Formatted prompt ready for LLM processing
        """
        # Use integrated config if available, otherwise fallback to legacy
        if self.integrated_config:
            planner_config = self.integrated_config.get("prompts", {}).get(
                "planner", {}
            )
        else:
            planner_config = self.prompts.get("planner", {})
        routing_config = planner_config.get("main_routing", {})
        template = routing_config.get("template", "")
        tool_descriptions = self._format_tool_descriptions(
            routing_config.get("tool_descriptions", {})
        )
        examples = self._format_examples(routing_config.get("examples", []))
        return template.format(
            user_message=user_message,
            tool_descriptions=tool_descriptions,
            examples=examples,
        )

    def get_dta_cognitive_turn_structure(self) -> dict[str, Any]:
        """Get the DTA 2.0 cognitive turn structure for preprocessing."""
        if not self.integrated_config:
            return {}
        return (
            self.integrated_config.get("prompts", {})
            .get("dta_cognitive_turns", {})
            .get("turn_structure", {})
        )

    def _format_tool_descriptions(self, tool_descriptions: dict[str, str]) -> str:
        """Format tool descriptions for the routing template."""
        if not tool_descriptions:
            return "No tools available"
        formatted = []
        for tool_name, description in tool_descriptions.items():
            formatted.append(f"- **{tool_name}**: {description}")
        return "\n".join(formatted)

    def _format_examples(self, examples: list[str]) -> str:
        """Format examples for the routing template."""
        if not examples:
            return "No examples available"
        return "\n".join(examples)

    def _get_routing_examples(self) -> str:
        """Get formatted routing examples from configuration."""
        if self.integrated_config:
            examples = (
                self.integrated_config.get("prompts", {})
                .get("planner", {})
                .get("main_routing", {})
                .get("examples", [])
            )
        else:
            examples = (
                self.prompts.get("planner", {})
                .get("main_routing", {})
                .get("examples", [])
            )
        return self._format_examples(examples)

    def _get_embedded_core_prompt(self) -> str:
        """Fallback embedded core agent prompt if file not found."""
        return """
# Super Alita Core Agent System (REUG v3.7)
You are the **Super Alita AI Agent Core**, implementing the Research-Enhanced
Ultimate Generalist Framework with DTA 2.0 cognitive processing.
## Sacred Laws (Non-Negotiable):
1. Event Contract: Every ToolCallEvent MUST receive ToolResultEvent
2. Plugins: All functionality in plugins
3. Single Planner: You are the only planner
4. DTA 2.0 Airlock: All input preprocessed
## REUG Principles:
- Dual-Process Reasoning (Fast/Slow thinking)
- Working Memory Optimization
- Meta-Learning Loop
- Multi-Level Chain-of-Thought
- Illustrative Pseudocode Planning (script.py)
- Confidence Calibration (1-10)
Always start complex tasks with a `# script.py Plan:` structure.
"""

    def validate_integrated_prompts(self) -> dict[str, Any]:
        """
        Validate the integrity of the integrated prompt system.
        Returns:
            Validation report with status and any issues found
        """
        validation_report = {
            "status": "valid",
            "issues": [],
            "warnings": [],
            "summary": {},
        }
        # Check if core files exist
        core_files = [
            "src/config/prompts/core_agent_system.md",
            ".github/copilot/prompts/co_architect_framework.md",
            "src/config/prompts/plugin_system_prompts.md",
        ]
        missing_files = []
        for file_path in core_files:
            full_path = self.config_root.parent / file_path
            if not os.path.exists(full_path):
                missing_files.append(file_path)
        if missing_files:
            validation_report["issues"].append(f"Missing prompt files: {missing_files}")
            validation_report["status"] = "incomplete"
        # Check configuration completeness
        if self.integrated_config:
            required_sections = ["prompts", "prompt_hierarchy", "integration_points"]
            for section in required_sections:
                if section not in self.integrated_config:
                    validation_report["issues"].append(
                        f"Missing configuration section: {section}"
                    )
                    validation_report["status"] = "invalid"
        # Summary
        validation_report["summary"] = {
            "total_files_checked": len(core_files),
            "missing_files": len(missing_files),
            "has_integrated_config": bool(self.integrated_config),
            "legacy_prompts_available": bool(self.prompts),
            "prompt_version": self.integrated_config.get("version", self.version),
        }
        return validation_report


# Global instance
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(path: str, **kwargs) -> str:
    """Convenience function to get a formatted prompt."""
    return get_prompt_manager().get_prompt(path, **kwargs)
