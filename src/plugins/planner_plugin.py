#!/usr/bin/env python3
"""
Legacy Planner Plugin for Super Alita
Converts goals into executable tool-call sequences using LADDER-AOG planning
Version 1.6.0 - LEGACY MODE: Defers to LLMPlannerPlugin for conversation handling

⚠️  DEPRECATION NOTICE:
    This plugin is now in legacy mode. The new LLMPlannerPlugin handles most
    conversation routing. This plugin now focuses on complex multi-step planning
    when explicitly requested via goal_received events.

    Migration Timeline:
    - v1.6.0: Compatibility mode with LLMPlannerPlugin (current)
    - v2.0.0: Full deprecation (planned)
"""

import json
import logging
import time
from typing import Any

from src.core.events import ToolCallEvent
from src.core.plan_executor import PlanExecutor
from src.core.plugin_interface import PluginInterface
from src.core.prompt_manager import get_prompt_manager

logger = logging.getLogger(__name__)


class PlannerPlugin(PluginInterface):
    """
    Legacy Plugin that converts user goals into structured plans and tool execution sequences.

    ⚠️  LEGACY MODE: Now defers to LLMPlannerPlugin for most conversation handling.

    Current Features:
    - Listens for goal_received events for complex multi-step planning
    - Creates LADDER-AOG style plans with tool calls
    - Emits tool_call events for execution by WebAgentAtom and other tools
    - Manages multi-step task coordination

    Migration Notes:
    - Simple conversation routing now handled by LLMPlannerPlugin
    - This plugin focuses on complex planning scenarios
    - Maintained for backward compatibility
    """

    def __init__(self):
        super().__init__()
        self.active_plans = {}
        self.plan_counter = 0
        self.executor = None  # Will be initialized in setup
        self._handled_tool_calls: set[str] = set()  # Added deduplication cache
        self.gemini_client = None  # Will be initialized in setup
        self.prompt_manager = get_prompt_manager()  # Initialize prompt manager

    @property
    def name(self) -> str:
        return "planner"

    async def setup(self, event_bus, store, config: dict[str, Any]) -> None:
        """Initialize the planner plugin."""
        await super().setup(event_bus, store, config)

        # Initialize closed-loop executor
        self.executor = PlanExecutor(event_bus, store)

        # Initialize Gemini client for intent detection
        try:
            from src.core.gemini_pilot import GeminiPilotClient

            self.gemini_client = GeminiPilotClient()
            logger.info(
                "Gemini client initialized for natural language intent detection"
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize Gemini client: {e} - falling back to pattern matching"
            )

        logger.info(
            "PlannerPlugin setup complete with closed-loop executor and natural language routing"
        )

    async def start(self) -> None:
        """Start the planner plugin."""
        await super().start()

        # Subscribe to goal events from conversation plugin
        await self.subscribe("goal_received", self._create_plan)
        await self.subscribe("tool_result", self._handle_tool_result)
        # NEW: Subscribe to user messages for natural language intent detection
        await self.subscribe("user_message", self._handle_user_message)
        # NEW: Subscribe to atom_ready events to notify users of new tools
        await self.subscribe("atom_ready", self._handle_atom_ready)

        logger.info(
            "PlannerPlugin started - ready to create plans from goals and natural language messages"
        )

    async def shutdown(self) -> None:
        """Shutdown the planner plugin."""
        # Cancel any active plans
        for plan_id in list(self.active_plans.keys()):
            plan = self.active_plans[plan_id]
            if plan["status"] == "executing":
                plan["status"] = "cancelled"

        logger.info(
            f"PlannerPlugin shutdown - cancelled {len(self.active_plans)} active plans"
        )
        self.active_plans.clear()

    async def _create_plan(self, event):
        """Create and execute plan using closed-loop executor."""
        try:
            goal = event.goal
            tools_needed = getattr(event, "tools_needed", ["web_agent"])
            session_id = event.session_id

            logger.info(f"🎯 Planning for goal: {goal}")
            logger.info(f"🔧 Tools needed: {tools_needed}")

            # Use closed-loop executor for stateful execution
            plan_id = f"plan_{session_id}_{self.plan_counter}"
            result = await self.executor.execute_plan(
                plan_id, session_id, goal, tools_needed
            )

            # Send final result to user
            await self.emit_event("agent_reply", text=result, session_id=session_id)

            logger.info(f"✅ Plan execution completed for session {session_id}")

        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            await self.emit_event(
                "agent_reply",
                text=f"❌ Plan execution failed: {e!s}",
                session_id=event.session_id,
            )

    async def _handle_user_message(self, event):
        """
        Handle user messages with LLM-first routing and minimal keyword shortcuts.
        Version 1.5.0 - Simplified routing, rely on LLM for better intent understanding.
        """
        try:
            # Extract message text from various event formats
            if hasattr(event, "text"):
                user_message = event.text
            elif hasattr(event, "user_message"):
                user_message = event.user_message
            elif isinstance(event, dict):
                user_message = event.get("text", event.get("user_message", ""))
            else:
                logger.warning(f"Could not extract message from event: {event}")
                return

            if not user_message or not user_message.strip():
                return

            user_message = user_message.strip()
            session_id = getattr(event, "session_id", "default")
            conversation_id = getattr(event, "conversation_id", session_id)

            # --- Only explicit override retained ---
            if user_message.startswith("/search "):
                query = user_message[8:].strip()
                logger.info(f"🔍 Explicit search override: {query}")
                await self._emit_tool_call(
                    "web_agent", {"query": query}, session_id, conversation_id
                )
                return

            # --- LLM-Based Routing for Everything Else ---
            if self.gemini_client:
                try:
                    # Get static tool descriptions from prompt manager
                    static_tools = self.prompt_manager.get_tool_descriptions(
                        "planner.main_routing.tool_descriptions"
                    )

                    # Get dynamic tools from the store
                    dynamic_tools = await self._get_dynamic_tools()

                    # Combine static and dynamic tools
                    all_tools = {**static_tools, **dynamic_tools}
                    tool_descriptions = "\n".join(
                        [f"{name}: {desc}" for name, desc in all_tools.items()]
                    )

                    examples = self.prompt_manager.get_examples(
                        "planner.main_routing.examples"
                    )

                    # Format the prompt using the template system
                    prompt = self.prompt_manager.get_prompt(
                        "planner.main_routing",
                        user_message=user_message,
                        tool_descriptions=tool_descriptions,
                        examples=examples,
                    )

                    logger.debug(
                        f"🎯 Using templated prompt for: {user_message[:50]}..."
                    )

                    decision = await self.gemini_client.generate_content(prompt)
                    decision_text = decision.strip()
                    logger.info(f"LLM Decision for '{user_message}': {decision_text}")

                    if decision_text.startswith("TOOL "):
                        try:
                            # Split carefully to handle potential spaces in JSON
                            parts = decision_text[5:].split(" ", 1)
                            if len(parts) != 2:
                                raise ValueError("Malformed TOOL response")
                            tool_name, params_str = parts[0], parts[1]
                            params = json.loads(params_str)
                            logger.info(
                                f"🤖 LLM-selected tool: {tool_name} with params: {params}"
                            )
                            await self._emit_tool_call(
                                tool_name, params, session_id, conversation_id
                            )
                            return
                        except (ValueError, json.JSONDecodeError) as e:
                            logger.warning(
                                f"Failed to parse LLM tool decision: {decision_text}. Error: {e}"
                            )
                            fallback_msg = "I had trouble figuring out how to use my tools for that request. Could you please rephrase it?"
                            await self._emit_chat_response(
                                fallback_msg, session_id, conversation_id
                            )
                            return

                    elif decision_text.startswith("GAP "):
                        # Tool gap detected - trigger autonomous tool creation
                        gap_description = decision_text[4:].strip()
                        logger.info(f"🔧 Tool gap detected: {gap_description}")
                        await self._handle_tool_gap(
                            gap_description, user_message, session_id, conversation_id
                        )
                        return

                    elif decision_text.startswith("NONE "):
                        # Graceful fallback with LLM-generated response
                        reply = decision_text[5:].strip()
                        logger.info(
                            f"🤖 LLM decided no tool needed, providing response: {reply[:50]}..."
                        )
                        await self._emit_chat_response(
                            reply, session_id, conversation_id
                        )
                        return

                    else:
                        # Handle unexpected LLM output
                        logger.warning(
                            f"Unexpected LLM response format: {decision_text}"
                        )
                        reply = "I'm not sure how to help with that right now. Could you try asking differently?"
                        await self._emit_chat_response(
                            reply, session_id, conversation_id
                        )
                        return

                except Exception as e:
                    logger.error(
                        f"PlannerPlugin failed to generate decision for '{user_message}': {e}",
                        exc_info=True,
                    )
                    error_msg = "I encountered an error while processing your request. Please try again later."
                    await self._emit_chat_response(
                        error_msg, session_id, conversation_id
                    )
                    return

            # Fallback if LLM is not available
            logger.warning("Gemini client not available - using basic fallback")
            await self._emit_chat_response(
                "I'm currently unable to process your request. Please try again later or use '/search <query>' for web searches.",
                session_id,
                conversation_id,
            )

        except Exception as e:
            logger.error(f"Error in user message handling: {e}", exc_info=True)

    async def _handle_tool_gap(
        self,
        gap_description: str,
        original_message: str,
        session_id: str,
        conversation_id: str,
    ):
        """
        Handle detected tool gaps by emitting atom gap request and providing user feedback.
        """
        try:
            # Emit atom gap request event for AutoToolPipeline
            from src.core.events import AtomGapRequestEvent

            gap_event = AtomGapRequestEvent(
                source_plugin=self.name,
                task=gap_description,
                context={
                    "original_message": original_message,
                    "session_id": session_id,
                    "conversation_id": conversation_id,
                    "timestamp": time.time(),
                },
            )

            await self.event_bus.publish(gap_event)
            logger.info(f"🚀 Emitted atom gap request: {gap_description}")

            # Provide immediate feedback to user
            response = self.prompt_manager.get_prompt(
                "planner.gap_response", gap_description=gap_description
            )

            await self._emit_chat_response(response, session_id, conversation_id)

            # Subscribe to atom_ready events to notify user when tool is available
            # (This could be enhanced to track specific gap requests)

        except Exception as e:
            logger.error(f"Error handling tool gap: {e}", exc_info=True)
            await self._emit_chat_response(
                "I recognize that you need a capability I don't have, but I encountered an error while trying to create it. Please try again later.",
                session_id,
                conversation_id,
            )

    async def _emit_chat_response(
        self, text: str, session_id: str, conversation_id: str
    ):
        """Emit a chat response directly to the user."""
        try:
            # Send response via the agent_reply channel for immediate display
            import json

            await self.event_bus._redis.publish(
                "agent_reply",
                json.dumps(
                    {
                        "text": text,
                        "session_id": session_id,
                        "source": "planner_fallback",
                    }
                ),
            )
            logger.info(f"💬 Chat response sent: {text[:50]}...")

        except Exception as e:
            logger.error(f"Failed to emit chat response: {e}")

    async def _emit_tool_call(
        self, tool_name: str, params: dict, session_id: str, conversation_id: str
    ):
        """Emit a tool call event with deduplication."""
        # Defensive normalization: convert legacy tool names
        if tool_name == "web_search":
            tool_name = "web_agent"
            # Also normalize parameter names for web searches
            if "input" in params and "query" not in params:
                params = {**params, "query": params.pop("input")}

        # Normalize memory_manager parameters
        elif tool_name == "memory_manager":
            # If LLM generated incorrect parameter structure, fix it
            if "input" in params and "action" not in params:
                # Extract content from input and set default action
                content = params.pop("input")
                params = {**params, "action": "save", "content": content}

        # CRITICAL: Dynamic tool name resolution
        # If the LLM uses a generic name like "calculator", map it to the actual dynamic tool
        else:
            # Check if this might be a reference to a dynamic tool
            dynamic_tools = await self._get_dynamic_tools()

            # Try to find a matching dynamic tool by searching for keywords
            if tool_name in ["calculator", "calc", "compute", "math"]:
                # Look for calculation-related dynamic tools
                for tool_key, tool_desc in dynamic_tools.items():
                    if any(
                        word in tool_desc.lower()
                        for word in [
                            "calculate",
                            "computation",
                            "math",
                            "interest",
                            "formula",
                        ]
                    ):
                        logger.info(
                            f"🔄 Mapping generic '{tool_name}' to dynamic tool '{tool_key}'"
                        )
                        tool_name = tool_key
                        break

            # Generic fallback: if tool_name doesn't match any known static tool,
            # check if there's a dynamic tool with a similar name
            elif tool_name not in ["web_agent", "memory_manager"]:  # Known static tools
                for tool_key in dynamic_tools.keys():
                    if (
                        tool_name.lower() in tool_key.lower()
                        or tool_key.lower() in tool_name.lower()
                    ):
                        logger.info(
                            f"🔄 Mapping '{tool_name}' to dynamic tool '{tool_key}'"
                        )
                        tool_name = tool_key
                        break

        call_id = f"plan_{int(time.time() * 1000)}_{tool_name}"

        if call_id in self._handled_tool_calls:
            logger.debug(f"Duplicate tool call {call_id} ignored")
            return  # Already dispatched

        self._handled_tool_calls.add(call_id)

        # Clean up old entries to prevent memory leak
        if len(self._handled_tool_calls) > 1000:
            # Keep only recent 500 entries
            sorted_calls = sorted(self._handled_tool_calls)
            self._handled_tool_calls = set(sorted_calls[-500:])

        try:
            await self.event_bus.publish(
                ToolCallEvent(
                    source_plugin=self.name,
                    tool_name=tool_name,
                    parameters=params,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    tool_call_id=call_id,
                )
            )
            logger.info(f"🔧 Emitted {tool_name} tool call: {call_id}")
        except Exception as e:
            logger.error(f"Failed to emit tool call: {e}")

    async def _handle_tool_result(self, event):
        """Forward tool results to PlanExecutor."""
        if self.executor:
            await self.executor._handle_tool_result(event)

    async def _handle_atom_ready(self, event):
        """Handle notifications of newly created tools."""
        try:
            data = event.model_dump() if hasattr(event, "model_dump") else event
            atom_info = data.get("atom", {})

            tool_name = atom_info.get("name", "New Tool")
            atom_info.get("description", "A new capability")

            logger.info(f"🎉 New tool available: {tool_name}")

            # For now, just log the event. In the future, we could:
            # 1. Store the tool info for future planning
            # 2. Notify active sessions that requested this capability
            # 3. Update the tool list for LLM prompts

        except Exception as e:
            logger.error(f"Error handling atom_ready event: {e}")

    async def _get_dynamic_tools(self) -> dict[str, str]:
        """Query the store for dynamically created tools and return their descriptions."""
        dynamic_tools = {}
        try:
            # Fallback: iterate through all atoms in the store
            for key, atom in self.store._atoms.items():
                if hasattr(atom, "value") and isinstance(atom.value, dict):
                    content = atom.value

                    # Check if this looks like a tool definition
                    if content.get("type") == "dynamic_tool" or (
                        "description" in content and "parameters" in content
                    ):
                        tool_name = content.get("name", key)
                        tool_description = content.get(
                            "description", "No description available"
                        )
                        parameters = content.get("parameters", {})

                        # Format the tool description for the LLM
                        param_desc = []
                        for param, details in parameters.items():
                            if isinstance(details, str):
                                param_desc.append(f"  {param}: {details}")
                            elif isinstance(details, dict):
                                param_type = details.get("type", "str")
                                desc = details.get("description", details)
                                param_desc.append(f"  {param} ({param_type}): {desc}")
                            else:
                                param_desc.append(f"  {param}: {details}")

                        param_text = (
                            "\n".join(param_desc)
                            if param_desc
                            else "  No parameters required"
                        )
                        dynamic_tools[key] = (
                            f"{tool_name}: {tool_description}\nParameters:\n{param_text}"
                        )

            logger.info(
                f"🔍 Found {len(dynamic_tools)} dynamic tools: {list(dynamic_tools.keys())}"
            )
            return dynamic_tools
        except Exception as e:
            logger.error(f"Error retrieving dynamic tools: {e}")
            return {}

    async def get_status(self) -> dict[str, Any]:
        """Get plugin status."""
        base_status = {
            "legacy_active_plans": len(self.active_plans),
            "total_plans_created": self.plan_counter,
            "planning_active": True,
        }

        if self.executor:
            executor_status = self.executor.get_stats()
            base_status.update(executor_status)

        return base_status
