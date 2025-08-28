from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from src.core.events import ToolCallEvent
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)

# Enhanced mapping from OaK options to concrete tool calls
# This bridges the abstract OaK tactical layer to operational tool execution
OPTION_TO_ACTION_MAPPING = {
    # Web and search related options
    "option-web-search": {
        "tool_name": "web_agent",
        "parameters": {"query": "{subgoal_description}"}
    },
    "option-search-info": {
        "tool_name": "fetch_github_raw", 
        "parameters": {
            "owner": "github",
            "repo": "docs", 
            "path": "README.md",
            "truncate": 1000
        }
    },
    
    # File operations
    "option-write-file": {
        "tool_name": "file_manager",
        "parameters": {
            "action": "write", 
            "path": "/tmp/output.txt", 
            "content": "{subgoal_description}"
        }
    },
    "option-read-file": {
        "tool_name": "fs_read",
        "parameters": {"path": "/tmp/input.txt"}
    },
    
    # Analysis and brainstorming
    "option-analyze": {
        "tool_name": "analyze_code_file",
        "parameters": {
            "file_path": "{subgoal_description}",
            "analysis_level": "semantic"
        }
    },
    "option-brainstorm": {
        "tool_name": "brainstorm_mcp_stub",
        "parameters": {"task": "{subgoal_description}"}
    },
    
    # Code and development
    "option-code-scan": {
        "tool_name": "secure_scan_code",
        "parameters": {"code": "{subgoal_description}"}
    },
    "option-prototype": {
        "tool_name": "full_cycle_prototype", 
        "parameters": {"task": "{subgoal_description}"}
    },
    
    # Default echo option for testing
    "option-echo": {
        "tool_name": "echo",
        "parameters": {"payload": "{subgoal_description}"}
    },
    
    # Enhanced analysis options
    "option-deep-analysis": {
        "tool_name": "analyze_workspace_context",
        "parameters": {
            "workspace_path": ".",
            "max_files": 10,
            "include_quality_metrics": True
        }
    },
    "option-understand-code": {
        "tool_name": "understand_code_structure",
        "parameters": {
            "target_path": ".",
            "focus_area": "architecture"
        }
    },
    
    # Backward compatibility with existing mapping
    "test-option": {
        "tool_name": "echo",
        "parameters": {"payload": "{goal}"}
    }
}


class OptionExecutorPlugin(PluginInterface):
    """
    Executes an OaK option by translating it into a concrete tool call.
    This plugin acts as the bridge between the Tactical (OaK) and Operational (Tool Execution) layers.
    
    Enhanced to support the full three-layer OaK reasoning architecture with improved
    parameter substitution and option mapping.
    """

    @property
    def name(self) -> str:
        return "option_executor"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        self.option_mapping = self.get_config(
            "option_mapping", OPTION_TO_ACTION_MAPPING
        )
        self.log("info", f"Option executor initialized with {len(self.option_mapping)} option mappings")

    async def start(self) -> None:
        await super().start()
        await self.subscribe("oak.plan_proposed", self._handle_plan_proposed)
        self.log("info", "Option executor started and listening for oak.plan_proposed events")

    async def _handle_plan_proposed(self, event: Any) -> None:
        """Handles a plan proposed by the OaK planning engine."""
        try:
            option_id = self._extract_option_id(event)
            if not option_id:
                self.log("warning", "No option_id found in oak.plan_proposed event")
                return
                
            subgoal_info = self._extract_subgoal_info(event)
            session_id = self._extract_session_id(event)
            
            # Look up the option in our mapping
            if option_id not in self.option_mapping:
                self.log("warning", f"Option ID '{option_id}' not found in the action mapping. Using fallback.")
                # Fallback to a known option for testing compatibility
                option_id = self._get_fallback_option()

            action_template = self.option_mapping[option_id]
            tool_name = action_template["tool_name"]

            # Enhanced parameter substitution
            populated_params = self._populate_parameters(action_template["parameters"], subgoal_info, event)

            self.log("info", f"Executing option '{option_id}' by calling tool '{tool_name}' with params: {populated_params}")

            # Emit tool call event
            await self._emit_tool_call(tool_name, populated_params, session_id, event)

        except Exception as e:
            self.log("error", f"Failed to execute option: {e}", exc_info=True)

    def _extract_option_id(self, event: Any) -> str | None:
        """Extract option ID from the plan event with multiple fallback strategies."""
        # Handle dict events first (most common)
        if isinstance(event, dict):
            # Direct option_id field
            if 'option_id' in event:
                return event['option_id']
            # From selected_option field
            elif 'selected_option' in event:
                selected_option = event['selected_option']
                if isinstance(selected_option, dict):
                    return selected_option.get('id')
                else:
                    return str(selected_option)
            # From plan array
            elif 'plan' in event and event['plan']:
                plan = event['plan']
                if isinstance(plan, list) and len(plan) > 0:
                    selected_option = plan[0]
                    if isinstance(selected_option, dict):
                        return selected_option.get("option_id")
                    else:
                        return str(selected_option)
            # Fallback to looking for any field containing 'option'
            for key, value in event.items():
                if 'option' in key.lower() and isinstance(value, str):
                    return value
        
        # Handle object events (hasattr checks)
        elif hasattr(event, 'option_id'):
            return event.option_id
        elif hasattr(event, 'selected_option'):
            if hasattr(event.selected_option, 'id'):
                return event.selected_option.id
            elif isinstance(event.selected_option, dict):
                return event.selected_option.get('id')
        elif hasattr(event, 'plan') and event.plan:
            # Legacy: extract from plan array
            plan = event.plan
            if isinstance(plan, list) and len(plan) > 0:
                selected_option = plan[0]
                if isinstance(selected_option, dict):
                    return selected_option.get("option_id")
                    
        return None

    def _extract_subgoal_info(self, event: Any) -> dict[str, Any]:
        """Extract subgoal information from the plan event."""
        subgoal_info = {}
        
        # Handle dict events first
        if isinstance(event, dict):
            if 'subgoal' in event:
                subgoal_info = event['subgoal']
            elif 'goal' in event:
                subgoal_info['description'] = event['goal']
        # Handle object events
        elif hasattr(event, 'subgoal'):
            if hasattr(event.subgoal, 'description'):
                subgoal_info['description'] = event.subgoal.description
                subgoal_info['subgoal_id'] = getattr(event.subgoal, 'subgoal_id', '')
                subgoal_info['parent_goal_id'] = getattr(event.subgoal, 'parent_goal_id', '')
            elif isinstance(event.subgoal, dict):
                subgoal_info = event.subgoal
        elif hasattr(event, 'goal'):
            subgoal_info['description'] = event.goal
            
        return subgoal_info

    def _extract_session_id(self, event: Any) -> str:
        """Extract session ID from the plan event."""
        if isinstance(event, dict):
            return event.get('session_id', 'default')
        elif hasattr(event, 'session_id'):
            return event.session_id
        return 'default'

    def _get_fallback_option(self) -> str:
        """Get a fallback option when the requested option is not found."""
        # Prefer existing test option for backward compatibility
        if "test-option" in self.option_mapping:
            return "test-option"
        elif "option-echo" in self.option_mapping:
            return "option-echo"
        elif self.option_mapping:
            return list(self.option_mapping.keys())[0]
        else:
            # Create a minimal fallback if no options exist
            self.option_mapping["fallback-echo"] = {
                "tool_name": "echo",
                "parameters": {"payload": "{subgoal_description}"}
            }
            return "fallback-echo"

    def _populate_parameters(self, param_template: dict[str, Any], subgoal_info: dict[str, Any], event: Any) -> dict[str, Any]:
        """Enhanced parameter population with multiple substitution sources."""
        populated_params = {}
        
        # Build substitution context
        substitution_context = {
            "subgoal_description": subgoal_info.get("description", ""),
            "subgoal_id": subgoal_info.get("subgoal_id", ""),
            "parent_goal_id": subgoal_info.get("parent_goal_id", ""),
            "goal": getattr(event, 'goal', subgoal_info.get("description", "")),  # Legacy compatibility
        }
        
        for param, value_template in param_template.items():
            if isinstance(value_template, str) and "{" in value_template:
                # Perform substitution
                populated_value = self._substitute_template(value_template, substitution_context)
                populated_params[param] = populated_value
            else:
                populated_params[param] = value_template
                
        return populated_params

    def _substitute_template(self, template: str, context: dict[str, Any]) -> str:
        """Substitute template placeholders with context values."""
        result = template
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result

    async def _emit_tool_call(self, tool_name: str, parameters: dict[str, Any], session_id: str, event: Any) -> None:
        """Emit a tool call event using the appropriate event format."""
        try:
            # Try to use ToolCallEvent if available
            tool_call_event = ToolCallEvent(
                source_plugin=self.name,
                tool_name=tool_name,
                parameters=parameters,
                session_id=session_id,
                conversation_id=getattr(event, "conversation_id", session_id),
                tool_call_id=f"tc_{uuid.uuid4()}",
            )
            await self.event_bus.publish(tool_call_event)
        except Exception as e:
            # Fallback to generic event emission
            self.log("debug", f"ToolCallEvent failed, using generic emit: {e}")
            await self.emit_event(
                "tool_call_request",
                action=tool_name,  # Use tool_name as action
                parameters=parameters,
                session_id=session_id,
                source_plugin=self.name,
            )

    def add_option_mapping(self, option_id: str, tool_spec: dict[str, Any]) -> None:
        """Dynamically add a new option-to-action mapping."""
        self.option_mapping[option_id] = tool_spec
        self.log("info", f"Added dynamic option mapping: {option_id} -> {tool_spec['tool_name']}")

    def get_available_options(self) -> list[str]:
        """Get list of all available option IDs."""
        return list(self.option_mapping.keys())

    async def health_check(self) -> dict[str, Any]:
        """Return health status for the option executor."""
        base_health = await super().health_check()
        base_health.update({
            "available_options": len(self.option_mapping),
            "option_mappings": list(self.option_mapping.keys()),
        })
        return base_health
