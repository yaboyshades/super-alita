"""REUG v9.0 Execution Flow Implementation.

Concrete implementation of the state machine handlers for Super Alita's
cognitive architecture. Replaces linear tool chaining with robust,
state-driven execution.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast

from src.computational_env.executor import ComputationalEnvironment
from src.core.decision_policy_v1 import DecisionPolicyEngine
from src.core.observability import ObservabilityLevel, get_observability_manager
from src.core.plugin_interface import PluginInterface
from src.core.session import Session
from src.core.states import StateMachine, StateType, TransitionTrigger
from src.core.tool_types import ToolProvidingPlugin, ToolSpec
from src.script_of_thought.interpreter import ScriptOfThoughtInterpreter
from src.tools.dynamic_tools import ToolSchemaGenerator, dynamic_tool_registry

logger = logging.getLogger(__name__)


class REUGExecutionFlow:
    """
    REUG v9.0 Execution Flow Orchestrator

    Manages the cognitive turn lifecycle using state-driven execution,
    replacing the brittle linear orchestration with resilient state transitions.
    """

    def __init__(
        self,
        event_bus: Any,
        plugin_registry: dict[str, PluginInterface | ToolProvidingPlugin],
    ) -> None:
        """Initialize the REUG execution flow orchestrator."""
        self.event_bus = event_bus
        self.plugin_registry: dict[str, PluginInterface | ToolProvidingPlugin] = (
            plugin_registry
        )
        self.plugins: list[PluginInterface | ToolProvidingPlugin] = list(
            plugin_registry.values()
        )
        self.logger = logging.getLogger(__name__)

        # Decision Policy
        self.decision_policy = DecisionPolicyEngine()

        # Observability
        self.observability = get_observability_manager()
        self.observability.level = ObservabilityLevel.DETAILED

        # Computational environment & SoT
        self.computational_env = ComputationalEnvironment()
        self.script_interpreter = ScriptOfThoughtInterpreter(self.computational_env)

        # Session & state machine
        self.session = Session()
        self.state_machine = StateMachine(event_bus, self.session)
        self._enhance_state_handlers()

        # Runtime execution context
        self.current_session_id: str | None = None
        self.is_running = False

        # Metrics
        self.turns_processed = 0
        self.errors_recovered = 0
        self.total_execution_time = 0.0

    def _build_decision_context(self) -> dict[str, Any]:
        return {
            "session_id": self.current_session_id,
            "history": self.session.history if hasattr(self.session, "history") else [],
            "plugin_count": len(self.plugins),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _convert_plan_to_tools(self, plan) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        items = getattr(plan, "plan", []) or []
        for step in items:
            name = step.get("name") or step.get("tool") or "unknown_tool"
            tools.append(
                {
                    "name": name,
                    "description": step.get("description", ""),
                    "type": step.get("type", "decision_policy"),
                    "function": {"name": name, "parameters": step.get("args", {})},
                    "plugin_name": step.get("plugin", ""),
                    "sot_step_id": step.get("sot_step_id", ""),
                }
            )
        return tools

    def _enhance_state_handlers(self) -> None:
        """Bind concrete handlers into the underlying state machine."""
        self.state_machine.state_handlers.update(
            {
                StateType.READY: self._handle_ready_state,
                StateType.ENGAGE: self._handle_engage_state,
                StateType.UNDERSTAND: self._handle_understand_state,
                StateType.GENERATE: self._handle_generate_state,
                StateType.CREATE_DYNAMIC_TOOL: self._handle_create_dynamic_tool_state,
                StateType.ERROR_RECOVERY: self._handle_error_recovery_state,
                StateType.COMPLETE: self._handle_complete_state,
                StateType.SHUTDOWN: self._handle_shutdown_state,
            }
        )

    async def start_session(self, session_id: str | None = None) -> str:
        """Start a new cognitive session and emit a session_started event."""
        self.current_session_id = (
            session_id or f"session_{datetime.now(UTC).isoformat()}"
        )
        self.is_running = True

        # Reset state machine context
        self.state_machine.reset_context(self.current_session_id)
        self.logger.info("Started REUG v9.0 session: %s", self.current_session_id)

        if self.event_bus:
            await self.event_bus.emit(
                "session_started",
                source_plugin="reug_execution_flow",
                session_id=self.current_session_id,
                timestamp=datetime.now(UTC).isoformat(),
            )
        return self.current_session_id

    async def process_user_input(self, user_input: str) -> dict[str, Any]:
        """
        Process user input through the REUG v9.0 state machine

        Args:
            user_input: User's input message

        Returns:
            dict containing response and execution metadata
        """
        start_time = datetime.now(UTC)

        try:
            # Transition to ENGAGE state with user input
            await self.state_machine.transition(
                TransitionTrigger.USER_INPUT, {"user_input": user_input}
            )

            # Run the state machine until completion
            result = await self._run_state_machine()

            # Calculate execution time
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            self.total_execution_time += execution_time
            self.turns_processed += 1

            return {
                "response": result.get(
                    "response", "I encountered an issue processing your request."
                ),
                "success": result.get("success", False),
                "session_id": self.current_session_id,
                "turn_id": self.state_machine.context.turn_id,
                "execution_time": execution_time,
                "state_transitions": self.state_machine.state_transitions_count,
                "error_count": self.state_machine.context.error_count,
                "metadata": result.get("metadata", {}),
            }

        except Exception as e:  # noqa: BLE001
            self.logger.error("Fatal error in REUG execution flow: %s", e)

            # Emergency fallback
            await self.state_machine.transition(TransitionTrigger.FATAL_ERROR)

            return {
                "response": "I encountered a system error and need to restart the conversation.",
                "success": False,
                "session_id": self.current_session_id,
                "error": str(e),
                "execution_time": (datetime.now(UTC) - start_time).total_seconds(),
            }

    async def _run_state_machine(self) -> dict[str, Any]:
        """Run the state machine until completion or error"""
        max_transitions = 20  # Prevent infinite loops
        transitions_count = 0

        while (
            self.state_machine.current_state != StateType.COMPLETE
            and self.state_machine.current_state != StateType.SHUTDOWN
            and transitions_count < max_transitions
        ):
            # Handle current state
            next_trigger = await self.state_machine.handle_current_state()

            if next_trigger is None:
                # State is waiting for external input
                break

            # Execute transition
            success = await self.state_machine.transition(next_trigger)
            if not success:
                self.logger.warning("State transition failed: %s", next_trigger)
                break

            transitions_count += 1

        # Check for completion
        if self.state_machine.current_state == StateType.COMPLETE:
            return {
                "response": self.state_machine.context.response
                or "Task completed successfully.",
                "success": True,
                "metadata": {
                    "tools_used": self.state_machine.context.tools_selected,
                    "intent": self.state_machine.context.detected_intent,
                    "transitions": transitions_count,
                },
            }
        elif transitions_count >= max_transitions:
            self.logger.error("State machine exceeded maximum transitions")
            return {
                "response": "I got stuck in a processing loop. Let me try a different approach.",
                "success": False,
                "metadata": {"error": "max_transitions_exceeded"},
            }
        else:
            return {
                "response": "I'm still processing your request.",
                "success": False,
                "metadata": {"current_state": self.state_machine.current_state.name},
            }

    # Concrete State Handler Implementations

    async def _handle_ready_state(self) -> TransitionTrigger | None:
        """Handle READY state - waiting for user input"""
        self.logger.debug("READY state: Waiting for user input")

        # This state waits for external USER_INPUT trigger
        # No automatic transition
        return None

    async def _handle_engage_state(self) -> TransitionTrigger | None:
        """Handle ENGAGE state - process user input and detect intent"""
        self.logger.debug("ENGAGE state: Processing user input and detecting intent")

        try:
            user_input = self.state_machine.context.user_input
            if not user_input:
                return TransitionTrigger.ERROR_OCCURRED

            # Simple intent detection (can be enhanced with ML models)
            detected_intent = await self._detect_intent(user_input)

            # Update context
            self.state_machine.context.detected_intent = detected_intent

            # Emit intent detection event (if event bus available)
            if self.event_bus:
                await self.event_bus.emit(
                    "intent_detected",
                    source_plugin="reug_execution_flow",
                    session_id=self.current_session_id,
                    intent=detected_intent,
                    user_input_length=len(user_input),
                )

            return TransitionTrigger.INTENT_DETECTED

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error in ENGAGE state: %s", e)
            self.state_machine.context.error_count += 1
            return TransitionTrigger.ERROR_OCCURRED

    async def _handle_understand_state(self) -> TransitionTrigger | None:
        """Handle UNDERSTAND state - parse SoT, load context and route tools"""
        self.logger.debug(
            "UNDERSTAND state: Parsing Script of Thought, building context and routing tools"
        )

        try:
            intent = self.state_machine.context.detected_intent or "general"
            user_input = self.state_machine.context.user_input or ""

            # **NEW: Parse Script of Thought if applicable**
            sot_parse_result = None
            if self._is_script_of_thought_format(user_input):
                self.logger.debug("Detected Script of Thought format, parsing...")
                try:
                    sot_parse_result = self.script_interpreter.parser.parse_script(
                        user_input
                    )
                    self.state_machine.context.sot_parse_result = sot_parse_result
                    if hasattr(sot_parse_result, "steps"):
                        self.logger.info(
                            "Parsed SoT with %d steps", len(sot_parse_result.steps)
                        )

                    # Emit SoT parsing event
                    if self.event_bus and hasattr(sot_parse_result, "steps"):
                        await self.event_bus.emit(
                            "sot_parsed",
                            source_plugin="reug_execution_flow",
                            session_id=self.current_session_id,
                            step_count=len(sot_parse_result.steps),
                            has_dependencies=len(sot_parse_result.steps)
                            > 1,  # heuristic
                        )
                except Exception as e:  # noqa: BLE001
                    self.logger.warning("Failed to parse as Script of Thought: %s", e)

            # Load memory context
            memory_context = await self._load_memory_context(intent, user_input)
            self.state_machine.context.memory_context = memory_context

            # Use Decision Policy instead of legacy router
            ctx = self._build_decision_context()
            plan = await self.decision_policy.decide_and_plan(user_input, ctx)
            selected_tools = self._convert_plan_to_tools(plan)
            self.state_machine.context.tools_selected = selected_tools

            # **ADDED: Log the chosen tools at TOOLS_ROUTED stage**
            tool_names = [tool.get("name", "unknown") for tool in selected_tools]
            self.logger.info("Tool routed: %s", tool_names)

            # Emit context loaded event (if event bus available)
            if self.event_bus:
                await self.event_bus.emit(
                    "context_loaded",
                    source_plugin="reug_execution_flow",
                    session_id=self.current_session_id,
                    tools_selected=tool_names,
                    memory_context_size=len(memory_context),
                    sot_enabled=sot_parse_result is not None,
                )

            return TransitionTrigger.TOOLS_ROUTED

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error in UNDERSTAND state: %s", e)
            self.state_machine.context.error_count += 1
            return TransitionTrigger.ERROR_OCCURRED

    async def _handle_generate_state(self) -> TransitionTrigger | None:
        """Handle GENERATE state - execute tools and create response (with computational environment)"""
        self.logger.debug(
            "GENERATE state: Executing tools via computational environment and generating response"
        )

        try:
            tools = self.state_machine.context.tools_selected
            user_input = self.state_machine.context.user_input or ""
            memory_context = self.state_machine.context.memory_context
            sot_parse_result = self.state_machine.context.sot_parse_result

            # **NEW: Execute via Script of Thought interpreter if available**
            if sot_parse_result:
                self.logger.debug("Executing tools via Script of Thought interpreter")
                try:
                    # Use SoT interpreter for orchestrated execution
                    sot_results = await self.script_interpreter.execute_script(
                        sot_parse_result
                    )
                    self.state_machine.context.sot_execution_state = sot_results

                    # Extract computational environment results if any
                    comp_env_results = {}
                    for step_id, step_result in sot_results.items():
                        if hasattr(step_result, "computational_output"):
                            comp_env_results[step_id] = step_result.computational_output

                    self.state_machine.context.comp_env_results = comp_env_results

                    # Emit SoT execution event
                    if self.event_bus:
                        await self.event_bus.emit(
                            "sot_executed",
                            source_plugin="reug_execution_flow",
                            session_id=self.current_session_id,
                            steps_executed=len(sot_results),
                            comp_env_used=bool(comp_env_results),
                        )

                    tool_results = sot_results  # Use SoT results as tool results

                except Exception as e:  # noqa: BLE001
                    self.logger.warning(
                        "SoT execution failed, falling back to standard tool execution: %s",
                        e,
                    )
                    # Fall back to standard tool execution
                    # Legacy path expects names; tools here is list[ToolSpec]
                    legacy_names = [t.get("name", "unknown") for t in tools]
                    tool_results = await self._execute_tools(
                        legacy_names, user_input, memory_context
                    )
            else:
                # **NEW: Enhanced standard tool execution with computational environment**
                tool_results = await self._execute_tools_with_comp_env(
                    tools, user_input, memory_context
                )

            self.state_machine.context.tool_results = tool_results

            # Generate response based on tool results
            response = await self._generate_response(
                user_input,
                self.state_machine.context.detected_intent or "general",
                tool_results,
                memory_context,
            )
            self.state_machine.context.response = response

            # Emit response ready event (if event bus available)
            if self.event_bus:
                await self.event_bus.emit(
                    "response_generated",
                    source_plugin="reug_execution_flow",
                    session_id=self.current_session_id,
                    response_length=len(response) if response else 0,
                    tools_executed=len(tool_results),
                    comp_env_used=bool(self.state_machine.context.comp_env_results),
                )

            return TransitionTrigger.RESPONSE_READY

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error in GENERATE state: %s", e)
            self.state_machine.context.error_count += 1
            return TransitionTrigger.TOOL_FAILURE

    async def _handle_error_recovery_state(self) -> TransitionTrigger | None:
        """Handle ERROR_RECOVERY state - attempt to recover from errors"""
        self.logger.debug("ERROR_RECOVERY state: Attempting error recovery")

        try:
            error_count = self.state_machine.context.error_count

            if error_count >= 3:
                # Too many errors, complete with failure
                self.state_machine.context.response = (
                    "I encountered multiple errors and cannot complete this request. "
                    "Please try rephrasing your question or try again later."
                )
                return TransitionTrigger.RECOVERY_SUCCESS

            # Attempt recovery based on error context
            recovery_successful = await self._attempt_recovery()

            if recovery_successful:
                self.errors_recovered += 1
                self.logger.info(
                    "Successfully recovered from error (attempt %d)", error_count
                )

                # Emit recovery event (if event bus available)
                if self.event_bus:
                    await self.event_bus.emit(
                        "error_recovered",
                        source_plugin="reug_execution_flow",
                        session_id=self.current_session_id,
                        error_count=error_count,
                        recovery_method="state_machine_retry",
                    )

                return TransitionTrigger.RECOVERY_SUCCESS
            else:
                return TransitionTrigger.FATAL_ERROR

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error in ERROR_RECOVERY state: %s", e)
            return TransitionTrigger.FATAL_ERROR

    async def _handle_complete_state(self) -> TransitionTrigger | None:
        """Handle COMPLETE state - finalize turn and prepare for next"""
        self.logger.debug("COMPLETE state: Finalizing cognitive turn")

        try:
            # Store memory for future turns
            await self._store_turn_memory()

            # Emit turn complete event (if event bus available)
            if self.event_bus:
                await self.event_bus.emit(
                    "turn_completed",
                    source_plugin="reug_execution_flow",
                    session_id=self.current_session_id,
                    turn_id=self.state_machine.context.turn_id,
                    success=bool(self.state_machine.context.response),
                    error_count=self.state_machine.context.error_count,
                )

            return TransitionTrigger.TURN_COMPLETE

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error in COMPLETE state: %s", e)
            # Even if completion fails, we should transition to ready
            return TransitionTrigger.TURN_COMPLETE

    async def _handle_shutdown_state(self) -> TransitionTrigger | None:
        """Handle SHUTDOWN state - graceful system shutdown"""
        self.logger.info("SHUTDOWN state: Performing graceful shutdown")

        self.is_running = False

        # Emit shutdown event (if event bus available)
        if self.event_bus:
            await self.event_bus.emit(
                "session_shutdown",
                source_plugin="reug_execution_flow",
                session_id=self.current_session_id,
                turns_processed=self.turns_processed,
                total_execution_time=self.total_execution_time,
            )

        return None  # Terminal state

    # Helper Methods

    async def _detect_intent(self, user_input: str) -> str:
        """Detect user intent from input (simplified version)"""
        # In a full implementation, this would use NLP models
        # For now, use simple keyword matching

        input_lower = user_input.lower()

        if any(word in input_lower for word in ["create", "make", "build", "generate"]):
            return "create"
        elif any(
            word in input_lower
            for word in ["fix", "debug", "error", "issue", "problem"]
        ):
            return "debug"
        elif any(
            word in input_lower for word in ["explain", "how", "what", "why", "help"]
        ):
            return "explain"
        elif any(word in input_lower for word in ["search", "find", "look", "locate"]):
            return "search"
        elif any(
            word in input_lower for word in ["analyze", "review", "check", "examine"]
        ):
            return "analyze"
        else:
            return "general"

    async def _load_memory_context(
        self, intent: str, user_input: str
    ) -> dict[str, Any]:
        """Load relevant memory context for the current turn"""
        # In a full implementation, this would query memory systems
        # For now, return basic context

        return {
            "intent": intent,
            "user_input_hash": hash(user_input),
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self.current_session_id,
        }

    async def _route_tools(
        self, intent: str, user_input: str, memory_context: dict[str, Any]
    ) -> list[str]:
        """Route to appropriate tools based on intent and context"""

        # Map intents to tool categories
        intent_tool_mapping = {
            "create": ["llm_planner", "creator"],
            "debug": ["llm_planner", "analyzer"],
            "explain": ["llm_planner", "explainer"],
            "search": ["llm_planner", "search"],
            "analyze": ["llm_planner", "analyzer"],
            "general": ["llm_planner"],
        }

        selected_tools = intent_tool_mapping.get(intent, ["llm_planner"])

        # Filter to only available plugins
        available_tools = []
        for tool in selected_tools:
            if tool in self.plugin_registry:
                available_tools.append(tool)

        return available_tools or ["llm_planner"]  # Always fallback to llm_planner

    async def _execute_tools(
        self,
        tools: list[str] | list[ToolSpec],
        user_input: str,
        memory_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute selected tools (legacy path) ensuring ToolSpec normalization.

        This legacy executor accepted a list of tool names. For consistency with the
        enhanced routing, we now also accept a list of ToolSpec objects. Each name is
        coerced into a minimal ToolSpec so downstream handling remains uniform.
        """
        # Normalize to ToolSpec list
        normalized: list[ToolSpec] = []
        if tools and isinstance(tools, list):  # preserve list semantics
            if tools and isinstance(tools[0], str):
                for name in cast(list[str], tools):
                    normalized.append(
                        cast(
                            ToolSpec,
                            {
                                "name": name,
                                "description": f"Legacy routed tool {name}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "description": "Legacy tool",
                                    "parameters": {},
                                },
                                "plugin_name": name,
                            },
                        )
                    )
            else:
                normalized = cast(list[ToolSpec], tools)

        results: dict[str, Any] = {}
        for tool in normalized:
            tool_name = tool.get("name", "unknown")
            try:
                plugin_name = tool.get("plugin_name", tool_name)
                plugin = self.plugin_registry.get(plugin_name)
                if not plugin:
                    self.logger.warning("Tool %s not found in registry", plugin_name)
                    results[tool_name] = {"error": f"Plugin {plugin_name} not found"}
                    continue
                if hasattr(plugin, "process_request"):
                    result = await plugin.process_request(user_input, memory_context)
                    results[tool_name] = {"success": True, "output": result}
                else:
                    self.logger.warning(
                        "Plugin %s lacks process_request; marking skipped", plugin_name
                    )
                    results[tool_name] = {
                        "success": False,
                        "error": "process_request not implemented",
                    }
            except Exception as e:  # noqa: BLE001
                self.logger.error("Error executing tool %s: %s", tool_name, e)
                results[tool_name] = {"success": False, "error": str(e)}
        return results

    async def _generate_response(
        self,
        user_input: str,
        intent: str,
        tool_results: dict[str, Any],
        memory_context: dict[str, Any],
    ) -> str:
        """Generate final response from tool results"""

        # **ADDED: Guard against empty tool output**
        if not tool_results:
            self.logger.warning("Empty tool response (no tool results)")
            return "⚠️ No response generated (no tools executed)"

        # Check for errors in tool results
        errors: list[str] = []
        for tool, result in tool_results.items():
            if isinstance(result, dict) and "error" in result:
                err_val = result.get("error") or "Unknown error"
                errors.append(f"{tool}: {err_val}")

        if errors:
            return f"I encountered some issues: {'; '.join(errors)}"

        # Generate success response
        successful_tools = [
            tool
            for tool, result in tool_results.items()
            if not (isinstance(result, dict) and "error" in result)
        ]

        if successful_tools:
            response = f"I've processed your {intent} request using {', '.join(successful_tools)}. The task has been completed."
        else:
            response = (
                "I've processed your request, though some tools encountered issues."
            )

        # **ADDED: Additional guard against empty response**
        if not response or not response.strip():
            self.logger.warning("Empty tool response (response string empty)")
            return "⚠️ No response generated (tool output empty)"

        return response

    async def _attempt_recovery(self) -> bool:
        """Attempt to recover from errors"""
        # Simple recovery strategy - clear partial results and retry

        self.state_machine.context.tool_results.clear()

        # In a more sophisticated implementation, this would:
        # - Analyze the error type
        # - Try alternative tools
        # - Adjust parameters
        # - Use fallback strategies

        return True  # Assume recovery is always possible for now

    async def _store_turn_memory(self) -> None:
        """Store turn results in memory for future reference"""
        # In a full implementation, this would persist to memory systems
        # For now, just log the turn completion

        turn_summary: dict[str, Any] = {
            "turn_id": self.state_machine.context.turn_id,
            "intent": self.state_machine.context.detected_intent,
            "tools_used": [
                t.get("name", "unknown")
                for t in self.state_machine.context.tools_selected
            ],
            "success": bool(self.state_machine.context.response),
            "error_count": self.state_machine.context.error_count,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.logger.info("Turn completed: %s", turn_summary)

    def _is_script_of_thought_format(self, user_input: str) -> bool:
        """Detect if user input follows Script of Thought format"""
        # Simple heuristic: look for step markers like "Step 1:", "1.", etc.
        import re

        step_patterns = [
            r"(?:Step|step)\s*\d+[:\.]",  # Step 1: or step 1.
            r"^\d+[:\.]",  # 1: or 1.
            r"(?:Step|step)\s*[a-zA-Z]",  # Step A or step a
            r"^[a-zA-Z]\)",  # A) or a)
        ]

        for pattern in step_patterns:
            if re.search(pattern, user_input, re.MULTILINE):
                return True

        # Also check for dependency keywords
        dependency_keywords = [
            "depends on",
            "requires",
            "after",
            "before",
            "prerequisite",
        ]
        for keyword in dependency_keywords:
            if keyword.lower() in user_input.lower():
                return True

        return False

    async def _route_tools_enhanced(
        self,
        intent: str,
        user_input: str,
        memory_context: dict[str, Any],
        sot_parse_result=None,
    ) -> list[ToolSpec]:
        """Route to appropriate tools based on intent and SoT parse results."""

        # Track operation for concurrency safety
        op_id = self.session.register_operation(
            f"route_tools_{intent}_{user_input[:50]}"
        )

        try:
            available_tools: list[ToolSpec] = []

            # Get tools from all plugins
            for plugin in self.plugins:
                try:
                    if isinstance(plugin, ToolProvidingPlugin):  # structural check
                        tools_candidate = plugin.get_tools()
                        if tools_candidate:
                            for tool in tools_candidate:
                                available_tools.append(tool)
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(
                        "Plugin %s get_tools() failed: %s",
                        getattr(plugin, "name", "unknown"),
                        e,
                    )

            # If we have SoT parse results, use them for intelligent routing
            if sot_parse_result and hasattr(sot_parse_result, "steps"):
                selected_tools: list[ToolSpec] = []
                for step in sot_parse_result.steps:
                    # Route tools for each step
                    step_tools = self._route_tools_for_step(step, available_tools)
                    selected_tools.extend(step_tools)

                # **FALLBACK PATH: If no tools selected from SoT, add fallback**
                if not selected_tools:
                    self.logger.info(
                        "No tools selected from SoT routing - adding fallback"
                    )
                    selected_tools = self._get_fallback_tools()

                return selected_tools
            else:
                # Standard routing: use intent-based selection
                selected_tools = []  # list[ToolSpec]

                # Try plugin-based tool routing
                for plugin in self.plugins:
                    try:
                        if isinstance(plugin, ToolProvidingPlugin):
                            routed = plugin.route_tools(
                                available_tools, user_input, memory_context
                            )
                            if routed:
                                selected_tools.extend(routed)
                    except Exception as e:  # noqa: BLE001
                        self.logger.warning(
                            "Plugin %s route_tools() failed: %s",
                            getattr(plugin, "name", "unknown"),
                            e,
                        )

                # **FALLBACK PATH: If no tools selected by routing, ensure we have something**
                if not selected_tools:
                    self.logger.info("No tools selected by routing - adding fallback")
                    selected_tools = self._get_fallback_tools()

                return selected_tools

        finally:
            self.session.complete_operation(op_id)

    def _route_tools_for_step(
        self, step: Any, available_tools: list[ToolSpec]
    ) -> list[ToolSpec]:
        """Route tools for a specific SoT step."""
        step_tools = []

        # Route based on step type
        if step.step_type == "code":
            # Prioritize code execution tools
            for tool in available_tools:
                if any(
                    keyword in tool.get("name", "").lower()
                    for keyword in ["code", "execute", "python", "run"]
                ):
                    step_tools.append(
                        cast(ToolSpec, {**tool, "sot_step_id": step.step_id})
                    )
        elif step.step_type == "search":
            # Prioritize search tools
            for tool in available_tools:
                if any(
                    keyword in tool.get("name", "").lower()
                    for keyword in ["search", "web", "find", "lookup"]
                ):
                    step_tools.append(
                        cast(ToolSpec, {**tool, "sot_step_id": step.step_id})
                    )
        elif step.step_type == "analysis":
            # Prioritize analysis tools
            for tool in available_tools:
                if any(
                    keyword in tool.get("name", "").lower()
                    for keyword in ["analyze", "data", "chart", "graph"]
                ):
                    step_tools.append(
                        cast(ToolSpec, {**tool, "sot_step_id": step.step_id})
                    )

        # If no specific tools found, include general tools
        if not step_tools:
            step_tools = available_tools[:1]  # At least one tool per step

        return step_tools

    async def _execute_tools_with_comp_env(
        self,
        tools: Sequence[ToolSpec],
        user_input: str,
        memory_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute tools with computational environment integration"""
        results = {}

        for tool in tools:
            tool_name = tool.get("name", "unknown_tool")
            # plugin_name = tool.get('plugin_name', 'unknown_plugin')  # Currently unused

            try:
                # Check if this tool requires computational environment
                if self._tool_needs_comp_env(tool):
                    self.logger.debug(
                        f"Executing {tool_name} via computational environment"
                    )

                    # Extract code or computational task from the tool/request
                    code_to_execute = self._extract_code_from_tool_request(
                        tool, user_input
                    )

                    if code_to_execute:
                        # Execute via computational environment
                        comp_result = await self.computational_env.execute_code(
                            code_to_execute
                        )

                        # Add memory context to result if needed
                        comp_result["memory_context"] = memory_context
                        results[tool_name] = {
                            "success": comp_result.get("success", False),
                            "output": comp_result.get("stdout", ""),
                            "error": comp_result.get("error"),
                            "computational_output": comp_result,
                            "execution_type": "computational_environment",
                        }
                    else:
                        # Fall back to standard execution
                        results[tool_name] = await self._execute_single_tool(
                            tool, user_input, memory_context
                        )
                else:
                    # Standard tool execution
                    results[tool_name] = await self._execute_single_tool(
                        tool, user_input, memory_context
                    )

            except Exception as e:
                self.logger.error(f"Error executing tool {tool_name}: {e}")
                results[tool_name] = {
                    "success": False,
                    "error": str(e),
                    "execution_type": "error",
                }

        return results

    def _tool_needs_comp_env(self, tool: ToolSpec) -> bool:
        """Determine if a tool should be executed via computational environment"""
        tool_name = tool.get("name", "").lower()
        tool_description = tool.get("description", "").lower()

        # Check for computational keywords
        comp_keywords = [
            "code",
            "python",
            "execute",
            "run",
            "compute",
            "calculate",
            "script",
            "data",
            "analysis",
        ]

        for keyword in comp_keywords:
            if keyword in tool_name or keyword in tool_description:
                return True

        return False

    def _extract_code_from_tool_request(
        self, tool: ToolSpec, user_input: str
    ) -> str | None:
        """Extract executable code from tool request"""
        # Look for code blocks in user input
        import re

        # Pattern for code blocks
        code_patterns = [
            r"```(?:python)?\n(.*?)\n```",  # Triple backticks
            r"`([^`]+)`",  # Single backticks
            r"python:\s*(.*?)(?:\n|$)",  # python: prefix
        ]

        for pattern in code_patterns:
            matches = re.findall(pattern, user_input, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()

        # If no explicit code block, but tool suggests computation,
        # return the user input for interpretation
        if self._tool_needs_comp_env(tool):
            return user_input

        return None

    async def _execute_single_tool(
        self, tool: ToolSpec, user_input: str, memory_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a single tool via its plugin"""
        plugin_name = tool.get("plugin_name", "unknown_plugin")
        # tool_name = tool.get('name', 'unknown_tool')  # Currently unused

        if plugin_name in self.plugin_registry:
            plugin = self.plugin_registry[plugin_name]
            try:
                result = await plugin.process_request(user_input, memory_context)
                return {"success": True, "output": result, "execution_type": "plugin"}
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "execution_type": "plugin_error",
                }
        else:
            return {
                "success": False,
                "error": f"Plugin {plugin_name} not found",
                "execution_type": "plugin_not_found",
            }

    async def _handle_create_dynamic_tool_state(self) -> TransitionTrigger | None:
        """Handle dynamic tool creation from natural language description"""
        context = self.state_machine.context

        try:
            # Extract tool description from user input
            user_input = context.user_input or ""

            # Look for tool creation keywords
            if not (
                "create tool" in user_input.lower()
                or "build tool" in user_input.lower()
                or "make tool" in user_input.lower()
            ):
                logger.warning("No tool creation intent detected")
                return TransitionTrigger.ERROR_OCCURRED

            # Initialize tool schema generator
            generator: Any = ToolSchemaGenerator()

            # Generate schema from description
            schema = await generator.generate_schema_from_description(user_input)

            if not schema:
                logger.error("Failed to generate tool schema from description")
                context.error_message = "Could not understand tool description"
                return TransitionTrigger.ERROR_OCCURRED

            # For now, create a simple placeholder implementation
            # In a real system, this would generate actual code
            async def placeholder_implementation(**kwargs: Any) -> dict[str, Any]:
                return {
                    "message": f"Tool {schema.name} executed with parameters: {kwargs}",
                    "status": "placeholder_execution",
                }

            # Register the dynamic tool
            success = dynamic_tool_registry.register_tool(
                schema, placeholder_implementation
            )

            if success:
                # Update context with tool creation results
                context.dynamic_tool_name = schema.name
                context.dynamic_tool_schema = schema
                context.response_content = {
                    "tool_created": True,
                    "tool_name": schema.name,
                    "tool_description": schema.description,
                    "parameters": [
                        {
                            "name": p.name,
                            "type": p.type.value,
                            "description": p.description,
                            "required": p.required,
                        }
                        for p in schema.parameters
                    ],
                }

                logger.info(f"Successfully created dynamic tool: {schema.name}")
                return TransitionTrigger.DYNAMIC_TOOL_CREATED
            else:
                logger.error(f"Failed to register dynamic tool: {schema.name}")
                context.error_message = "Failed to register tool in registry"
                return TransitionTrigger.ERROR_OCCURRED

        except Exception as e:
            logger.error(f"Error in CREATE_DYNAMIC_TOOL state: {e}")
            context.error_message = f"Tool creation failed: {str(e)}"
            return TransitionTrigger.ERROR_OCCURRED

    def _get_fallback_tools(self) -> list[ToolSpec]:
        """Provide fallback tools when no tools are selected by routing."""
        return [
            cast(
                ToolSpec,
                {
                    "name": "direct_response",
                    "description": "Generate a direct response without external tools",
                    "type": "function",
                    "function": {
                        "name": "direct_response",
                        "description": "Generate response based on the user input",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "response": {
                                    "type": "string",
                                    "description": "The response to the user",
                                }
                            },
                            "required": ["response"],
                        },
                    },
                    "plugin_name": "reug_execution_flow",
                },
            )
        ]

    async def shutdown(self) -> None:
        """Shutdown the execution flow"""
        if self.state_machine.current_state != StateType.SHUTDOWN:
            await self.state_machine.transition(TransitionTrigger.SHUTDOWN_REQUESTED)

        self.is_running = False

    def get_metrics(self) -> dict[str, Any]:
        """Get execution flow metrics"""
        return {
            "session_id": self.current_session_id,
            "is_running": self.is_running,
            "turns_processed": self.turns_processed,
            "errors_recovered": self.errors_recovered,
            "total_execution_time": self.total_execution_time,
            "average_turn_time": self.total_execution_time
            / max(1, self.turns_processed),
            "state_machine_metrics": self.state_machine.get_metrics(),
        }
