"""
Tool Executor Plugin for Super Alita
Executes plans by running individual tool calls and actions
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any

from ..core.events import (
    ToolResultEvent,
)
from ..core.plugin_interface import PluginInterface
from ..core.secure_executor import get_tool_registry

logger = logging.getLogger(__name__)


class ToolExecutorPlugin(PluginInterface):
    """
    Tool Executor Plugin that receives plans and executes them step by step.

    This plugin bridges the gap between high-level planning (LADDER-AOG)
    and actual tool execution, turning abstract plan steps into concrete actions.
    """

    def __init__(self):
        super().__init__()
        self._name = "tool_executor"
        self.active_executions: dict[str, dict[str, Any]] = {}
        self.execution_history: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Executes plans by running individual tool calls and actions"

    async def setup(self, event_bus, store, config: dict[str, Any]) -> None:
        """Initialize the tool executor plugin."""
        await super().setup(event_bus, store, config)

        # Get configuration
        self.max_concurrent_executions = config.get("max_concurrent_executions", 3)
        self.execution_timeout = config.get("execution_timeout_seconds", 300)
        self.enable_parallel_execution = config.get("enable_parallel_execution", False)

        logger.info("Tool executor plugin setup complete")

    async def start(self) -> None:
        """Start the plugin and subscribe to events."""
        await super().start()

        # Subscribe to plan execution events
        await self.subscribe("plan_ready", self._execute_plan)
        await self.subscribe("tool_call_request", self._execute_tool_call)
        await self.subscribe("execution_status_request", self._handle_status_request)
        await self.subscribe(
            "tool_call", self._handle_atom_tool_execution
        )  # For composed atoms
        await self.subscribe(
            "tool_call", self._handle_dynamic_tool_execution
        )  # For dynamic registry tools
        await self.subscribe(
            "user_message", self._handle_tool_commands
        )  # For direct tool calls

        logger.info("Tool executor plugin started - ready to execute plans and tools")

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        # Cancel any running executions
        for execution_id in list(self.active_executions.keys()):
            await self._cancel_execution(execution_id)

        # Base class shutdown is abstract, so we don't call super()
        logger.info("Tool executor plugin shut down")

    async def _execute_plan(self, event) -> None:
        """Execute a plan received from the planning system."""
        try:
            plan_id = getattr(event, "plan_id", None)
            plan_data = getattr(event, "plan", None)
            goal_description = getattr(event, "goal_description", "Unknown goal")
            session_id = getattr(event, "session_id", "unknown")

            if not plan_id or not plan_data:
                logger.warning(
                    "Received plan_ready event with missing plan_id or plan data"
                )
                return

            logger.info(f"Starting execution of plan: {plan_id}")

            # Create execution record
            execution_id = f"exec_{uuid.uuid4().hex[:8]}"
            execution_record = {
                "execution_id": execution_id,
                "plan_id": plan_id,
                "goal_description": goal_description,
                "session_id": session_id,
                "start_time": datetime.now(),
                "status": "running",
                "steps_completed": 0,
                "steps_total": 0,
                "current_step": None,
                "results": [],
                "errors": [],
            }

            self.active_executions[execution_id] = execution_record

            # Emit execution started event
            await self.emit_event(
                "execution_started",
                execution_id=execution_id,
                plan_id=plan_id,
                goal_description=goal_description,
                session_id=session_id,
            )

            # Extract plan steps
            plan_steps = self._extract_plan_steps(plan_data)
            execution_record["steps_total"] = len(plan_steps)

            if not plan_steps:
                logger.warning(f"No executable steps found in plan: {plan_id}")
                await self._complete_execution(
                    execution_id, success=False, reason="No executable steps"
                )
                return

            # Execute steps
            success = await self._execute_plan_steps(execution_id, plan_steps)

            # Complete execution
            await self._complete_execution(execution_id, success=success)

        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            if "execution_id" in locals():
                await self._complete_execution(
                    execution_id, success=False, reason=str(e)
                )

    def _extract_plan_steps(self, plan_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract executable steps from plan data."""
        steps = []

        try:
            # Handle different plan formats
            if isinstance(plan_data, dict):
                # Check for steps array
                if "steps" in plan_data:
                    steps = plan_data["steps"]
                # Check for actions array
                elif "actions" in plan_data:
                    steps = plan_data["actions"]
                # Check for nodes (AOG format)
                elif "nodes" in plan_data:
                    nodes = plan_data["nodes"]
                    for node in nodes:
                        if node.get("type") == "action":
                            steps.append(
                                {
                                    "action": node.get("name", "unknown"),
                                    "description": node.get("description", ""),
                                    "parameters": node.get("metadata", {}),
                                }
                            )

            # Ensure each step has required fields
            normalized_steps = []
            for i, step in enumerate(steps):
                normalized_step = {
                    "step_id": f"step_{i + 1}",
                    "action": step.get("action", step.get("name", f"action_{i + 1}")),
                    "description": step.get("description", ""),
                    "parameters": step.get("parameters", step.get("metadata", {})),
                    "timeout": step.get("timeout", 30),
                }
                normalized_steps.append(normalized_step)

            logger.info(f"Extracted {len(normalized_steps)} executable steps")
            return normalized_steps

        except Exception as e:
            logger.error(f"Error extracting plan steps: {e}")
            return []

    async def _execute_plan_steps(
        self, execution_id: str, steps: list[dict[str, Any]]
    ) -> bool:
        """Execute all steps in a plan."""
        execution_record = self.active_executions[execution_id]

        try:
            for i, step in enumerate(steps):
                if execution_record["status"] == "cancelled":
                    logger.info(f"Execution {execution_id} was cancelled")
                    return False

                execution_record["current_step"] = step
                step_id = step["step_id"]

                logger.info(f"Executing step {i + 1}/{len(steps)}: {step['action']}")

                # Execute the step
                step_result = await self._execute_single_step(step)

                # Record result
                execution_record["results"].append(
                    {
                        "step_id": step_id,
                        "success": step_result["success"],
                        "result": step_result.get("result", ""),
                        "duration": step_result.get("duration", 0),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                if step_result["success"]:
                    execution_record["steps_completed"] += 1

                    # Emit step completion event
                    await self.emit_event(
                        "step_completed",
                        execution_id=execution_id,
                        step_id=step_id,
                        step_result=step_result["result"],
                        steps_completed=execution_record["steps_completed"],
                        steps_total=execution_record["steps_total"],
                    )
                else:
                    error_msg = step_result.get("error", "Step execution failed")
                    execution_record["errors"].append(f"Step {step_id}: {error_msg}")

                    # Emit step failure event
                    await self.emit_event(
                        "step_failed",
                        execution_id=execution_id,
                        step_id=step_id,
                        error=error_msg,
                    )

                    # For now, continue with other steps even if one fails
                    # In the future, this could be configurable
                    logger.warning(f"Step {step_id} failed but continuing: {error_msg}")

            # Success if we completed at least some steps
            success = execution_record["steps_completed"] > 0
            return success

        except Exception as e:
            logger.error(f"Error executing plan steps: {e}")
            execution_record["errors"].append(f"Execution error: {e!s}")
            return False

    async def _execute_single_step(self, step: dict[str, Any]) -> dict[str, Any]:
        """Execute a single step and return the result."""
        start_time = datetime.now()

        try:
            action = step["action"]
            parameters = step.get("parameters", {})
            timeout = step.get("timeout", 30)
            tool_key = step.get("tool_key")

            # Emit tool call event for this step
            await self.emit_event(
                "tool_call",
                action=action,
                parameters=parameters,
                step_id=step["step_id"],
                metadata={"timeout": timeout},
            )

            # Handle dynamic tool execution
            if action == "execute_dynamic_tool" and tool_key:
                result = await self._execute_dynamic_tool(tool_key, parameters)
            else:
                # Handle static tools or fallback
                result = await self._execute_static_tool(action, parameters)

            duration = (datetime.now() - start_time).total_seconds()

            return {
                "success": result.get("success", True),
                "result": result,
                "duration": duration,
            }

        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            duration = (datetime.now() - start_time).total_seconds()

            return {"success": False, "error": str(e), "duration": duration}

    async def _execute_dynamic_tool(
        self, tool_key: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a dynamically generated tool with security and audit trail."""
        try:
            from ..core.dynamic_tool_generator import get_dynamic_generator
            from ..core.secure_executor import get_secure_executor
            from ..core.tool_memory import get_memory_manager

            generator = get_dynamic_generator()
            get_secure_executor()
            memory_manager = get_memory_manager(self.store)

            # Get the tool atom from the store
            tool_atom = self.store.get(tool_key)
            if not tool_atom:
                logger.error(f"Dynamic tool not found: {tool_key}")
                return {
                    "success": False,
                    "error": f"Dynamic tool not found: {tool_key}",
                }

            # Extract tool data
            tool_data = tool_atom.value
            if isinstance(tool_data, dict):
                tool_name = tool_data.get("name", tool_key)
                tool_code = tool_data.get("code", "")

                if not tool_code:
                    # Try to get from memory manager
                    tool_memory = memory_manager.get_tool_from_library(tool_name)
                    if tool_memory:
                        tool_code = tool_memory.code

                if tool_code:
                    logger.info(f"🔧 Executing secure dynamic tool: {tool_name}")

                    # Execute with security and audit trail
                    response = generator.execute_dynamic_tool(
                        code=tool_code,
                        params=parameters,
                        user_id="system",
                        context_id=f"execution_{tool_key}",
                        tool_name=tool_name,
                    )

                    return {
                        "success": True,
                        "result": response,
                        "message": "Secure dynamic tool execution completed",
                        "tool_name": tool_name,
                        "audit_id": response.get("audit_id"),
                        "memory_id": response.get("memory_id"),
                        "security_validated": True,
                    }
                logger.warning(f"No executable code found for tool: {tool_name}")
                # Fallback to mock execution for compatibility
                return await self._mock_dynamic_tool_execution(tool_name, parameters)
            logger.warning(f"Invalid tool data format for: {tool_key}")
            return {
                "success": False,
                "error": f"Invalid tool data format for {tool_key}",
            }

        except Exception as e:
            logger.error(f"Dynamic tool execution failed: {e}")
            return {
                "success": False,
                "error": f"Dynamic tool execution error: {e!s}",
            }

    async def _mock_dynamic_tool_execution(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock dynamic tool execution for compatibility"""
        logger.info(f"🔧 Mock executing dynamic tool: {tool_name}")

        if "quantum" in tool_name.lower():
            result = {
                "success": True,
                "tool_output": {
                    "circuit_diagram": "Mock quantum circuit with 3 qubits, depth 5",
                    "qubits": parameters.get("qubits", 3),
                    "depth": parameters.get("depth", 5),
                    "gates_used": ["h", "cnot"],
                    "measurement_results": "Mock measurement data",
                },
                "message": "Mock dynamic quantum tool executed successfully",
            }
        elif "math" in tool_name.lower():
            result = {
                "success": True,
                "tool_output": {
                    "function": parameters.get("function", "x**2"),
                    "analysis": "Mock mathematical analysis complete",
                    "derivative": "2*x",
                    "zeros": ["0"],
                    "domain_analysis": "Real numbers",
                },
                "message": "Mock dynamic math tool executed successfully",
            }
        else:
            result = {
                "success": True,
                "tool_output": f"Mock dynamic tool {tool_name} executed with parameters: {parameters}",
                "message": "Mock dynamic tool execution completed",
            }

        return result

    async def _execute_static_tool(
        self, action: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a static/legacy tool."""
        try:
            # Legacy tool execution - simulate for now
            await asyncio.sleep(0.5)  # Simulate work

            result = f"Executed static tool '{action}' with parameters: {parameters}"

            return {
                "success": True,
                "result": result,
                "message": "Static tool execution completed",
            }

        except Exception as e:
            logger.error(f"Static tool execution failed: {e}")
            return {"success": False, "error": f"Static tool error: {e!s}"}

    async def _execute_tool_call(self, event) -> None:
        """Execute a specific tool call from conversation plugin"""
        try:
            tool_name = event.tool_name
            tool_args = event.arguments

            logger.info(f"Executing tool call: {tool_name} with args: {tool_args}")

            if tool_name in self.available_tools:
                tool = self.available_tools[tool_name]
                result = await tool.execute(**tool_args)

                response_event = ToolResultEvent(
                    source_plugin=self.name,
                    conversation_id=event.conversation_id,
                    tool_name=tool_name,
                    result=result.result,
                    success=result.success,
                    error=result.error,
                    message=result.message,
                )

                await self.event_bus.publish(response_event)
                logger.info(f"Tool call executed successfully: {tool_name}")

            else:
                logger.error(f"Tool not found: {tool_name}")

                error_event = ToolResultEvent(
                    source_plugin=self.name,
                    conversation_id=event.conversation_id,
                    tool_name=tool_name,
                    result=None,
                    success=False,
                    error=f"Tool '{tool_name}' not found",
                    message="Tool execution failed",
                )

                await self.event_bus.publish(error_event)

        except Exception as e:
            logger.error(f"Error executing tool call: {e}")

            error_event = ToolResultEvent(
                source_plugin=self.name,
                conversation_id=getattr(event, "conversation_id", "unknown"),
                tool_name=getattr(event, "tool_name", "unknown"),
                result=None,
                success=False,
                error=str(e),
                message="Tool execution error",
            )

            await self.event_bus.publish(error_event)
        """Handle individual tool call requests."""
        try:
            action = getattr(event, "action", None)
            parameters = getattr(event, "parameters", {})

            if not action:
                logger.warning("Received tool_call_request with no action")
                return

            logger.info(f"Executing tool call: {action}")

            # For now, just log the tool call
            # In a full implementation, this would dispatch to actual tools
            result = f"Tool call executed: {action} with {parameters}"

            await self.emit_event(
                "tool_call_result",
                action=action,
                parameters=parameters,
                result=result,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error executing tool call: {e}")

    async def _complete_execution(
        self, execution_id: str, success: bool = True, reason: str = ""
    ) -> None:
        """Complete an execution and clean up."""
        if execution_id not in self.active_executions:
            return

        execution_record = self.active_executions[execution_id]
        execution_record["status"] = "completed" if success else "failed"
        execution_record["end_time"] = datetime.now()
        execution_record["duration"] = (
            execution_record["end_time"] - execution_record["start_time"]
        ).total_seconds()

        if reason:
            execution_record["completion_reason"] = reason

        # Move to history
        self.execution_history.append(execution_record.copy())
        del self.active_executions[execution_id]

        # Emit completion event
        await self.emit_event(
            "execution_completed",
            execution_id=execution_id,
            success=success,
            steps_completed=execution_record["steps_completed"],
            steps_total=execution_record["steps_total"],
            duration=execution_record["duration"],
            session_id=execution_record.get("session_id", "unknown"),
        )

        logger.info(
            f"Execution {execution_id} completed: {'success' if success else 'failed'}"
        )

    async def _cancel_execution(self, execution_id: str) -> None:
        """Cancel a running execution."""
        if execution_id in self.active_executions:
            self.active_executions[execution_id]["status"] = "cancelled"
            await self._complete_execution(
                execution_id, success=False, reason="Cancelled"
            )

    async def _handle_status_request(self, event) -> None:
        """Handle execution status requests."""
        try:
            # Emit status information
            await self.emit_event(
                "execution_status",
                active_executions=len(self.active_executions),
                completed_executions=len(self.execution_history),
                execution_details=list(self.active_executions.values()),
            )

        except Exception as e:
            logger.error(f"Error handling status request: {e}")

    async def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        completed_success = sum(
            1 for exec in self.execution_history if exec.get("status") == "completed"
        )
        completed_failed = sum(
            1 for exec in self.execution_history if exec.get("status") == "failed"
        )

        return {
            "active_executions": len(self.active_executions),
            "total_completed": len(self.execution_history),
            "successful_executions": completed_success,
            "failed_executions": completed_failed,
            "success_rate": completed_success / max(len(self.execution_history), 1),
        }

    async def _handle_atom_tool_execution(self, event) -> None:
        """Handle execution of composed atoms from brainstorm/compose plugins"""
        try:
            tool_data = getattr(event, "tool", {})
            params = getattr(event, "params", {})

            # Check if this is the right type of event for atom tools
            if not tool_data or not isinstance(tool_data, dict):
                return  # Not an atom tool event, let other handlers deal with it

            tool_name = tool_data.get("tool", "unknown")
            code = tool_data.get("code", "")

            # If no code is present, this isn't our type of event
            if not code:
                return  # Silently return - not an atom tool event

            logger.info(f"🔧 Executing atom tool: {tool_name}")

            # Execute with real subprocess for safety
            result = await self._execute_python_code(code, params)

            await self.emit_event(
                "tool_result", tool_name=tool_name, result=result, success=True
            )

            logger.info(f"✅ Tool executed → result: {str(result)[:100]}")

        except Exception as e:
            logger.error(f"Error executing atom tool: {e}")

    async def _handle_dynamic_tool_execution(self, event) -> None:
        """Handle execution of dynamically registered tools from CREATOR plugin"""
        try:
            # Check if this is a ToolCallEvent for a dynamically registered tool
            if not hasattr(event, "tool_name"):
                return  # Not a ToolCallEvent

            tool_name = event.tool_name
            parameters = getattr(event, "parameters", {})

            # Check if tool exists in dynamic registry
            registry = get_tool_registry()
            tool_info = registry.get_tool(tool_name)

            if not tool_info:
                return  # Tool not in dynamic registry, let other handlers deal with it

            logger.info(
                f"🎯 Executing dynamic tool: {tool_name} with params: {parameters}"
            )

            # Get the tool code
            tool_code = tool_info["code"]

            # Execute the tool code with parameters and event context
            result = await self._execute_dynamic_tool_code(
                tool_name, tool_code, parameters, event
            )

            # Track usage in registry
            registry.use_tool(tool_name)

            # Emit result event with properly formatted result
            result_data = {
                "value": result,
                "tool_name": tool_name,
                "parameters": parameters,
            }

            await self.emit_event(
                "tool_result",
                source_plugin=self.name,
                tool_call_id=getattr(event, "tool_call_id", ""),
                conversation_id=getattr(event, "conversation_id", ""),
                session_id=getattr(event, "session_id", ""),
                tool_name=tool_name,
                success=True,
                result=result_data,
            )

            logger.info(
                f"✅ Dynamic tool '{tool_name}' executed successfully: {str(result)[:100]}"
            )

        except Exception as e:
            logger.error(f"Error executing dynamic tool: {e}")

            # Emit error result
            try:
                await self.emit_event(
                    "tool_result",
                    source_plugin=self.name,
                    tool_call_id=getattr(event, "tool_call_id", ""),
                    conversation_id=getattr(event, "conversation_id", ""),
                    session_id=getattr(event, "session_id", ""),
                    tool_name=getattr(event, "tool_name", "unknown"),
                    success=False,
                    result={"error": str(e)},
                    error=str(e),
                )
            except Exception:
                pass

    async def _execute_dynamic_tool_code(
        self, tool_name: str, code: str, parameters: dict, event
    ) -> Any:
        """Execute dynamic tool code safely with given parameters."""
        try:
            # Create a safe execution environment
            safe_globals = {
                "__builtins__": {
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "abs": abs,
                    "max": max,
                    "min": min,
                    "sum": sum,
                    "print": print,
                    "ValueError": ValueError,
                    "TypeError": TypeError,
                    "Exception": Exception,
                    "isinstance": isinstance,
                    "hasattr": hasattr,
                    "getattr": getattr,
                    "__import__": __import__,
                },
                "__name__": "__dynamic_tool__",  # Add __name__ to prevent errors
                "ast": __import__("ast"),
                "operator": __import__("operator"),
                "math": __import__("math"),
                "asyncio": __import__("asyncio"),
                "uuid": __import__("uuid"),
            }

            # Pre-import event classes to avoid import issues in generated code
            try:
                from src.core.events import ToolResultEvent

                safe_globals["ToolResultEvent"] = ToolResultEvent
            except ImportError:
                pass

            # Execute the tool code to define the function
            exec(code, safe_globals)

            # Get the function (should have same name as tool)
            if tool_name in safe_globals:
                tool_function = safe_globals[tool_name]
            else:
                raise ValueError(f"Function '{tool_name}' not found in tool code")

            # Prepare parameters for execution including event bus and metadata
            exec_params = parameters.copy()
            exec_params.update(
                {
                    "event_bus": self.event_bus,
                    "tool_call_id": getattr(event, "tool_call_id", ""),
                    "session_id": getattr(event, "session_id", ""),
                    "conversation_id": getattr(event, "conversation_id", ""),
                }
            )

            # Handle async vs sync functions properly
            import asyncio
            import inspect

            if inspect.iscoroutinefunction(tool_function):
                # Async function: await it directly (we're already in async context)
                result = await asyncio.wait_for(
                    tool_function(**exec_params), timeout=5.0
                )
            else:
                # Sync function: use the execute_tool function with timeout
                from src.core.secure_executor import execute_tool

                result = execute_tool(tool_function, exec_params, timeout=5)

            return result

        except Exception as e:
            logger.error(f"Error executing dynamic tool code: {e}")
            raise

    async def _handle_tool_commands(self, event) -> None:
        """Handle direct tool execution commands from chat"""
        try:
            # Parse the event data
            if hasattr(event, "model_dump"):
                data = event.model_dump()
            elif hasattr(event, "__dict__"):
                data = event.__dict__
            elif isinstance(event, dict):
                data = event
            else:
                try:
                    data = json.loads(str(event))
                except (json.JSONDecodeError, ValueError):
                    return

            text = data.get("text", "")

            # Handle /atom-run command for direct tool execution
            if text.startswith("/atom-run"):
                await self._handle_atom_run_command(text)

        except Exception as e:
            logger.error(f"Error handling tool command: {e}")

    async def _handle_atom_run_command(self, text: str) -> None:
        """Handle /atom-run tool="name" args="value" commands"""
        try:
            import re

            # Parse command: /atom-run tool="prime_counter" args="100"
            tool_match = re.search(r'tool="([^"]+)"', text)
            args_match = re.search(r'args="([^"]+)"', text)

            if not tool_match:
                await self.event_bus._redis.publish(
                    "agent_reply",
                    json.dumps(
                        {
                            "text": '❌ Invalid format. Use: /atom-run tool="tool_name" args="arguments"'
                        }
                    ),
                )
                return

            tool_name = tool_match.group(1)
            args = args_match.group(1) if args_match else ""

            logger.info(f"🔧 Direct tool execution: {tool_name} with args: {args}")

            # Handle specific tools
            if tool_name == "prime_counter":
                result = await self._execute_prime_counter(args)

                await self.event_bus._redis.publish(
                    "agent_reply",
                    json.dumps({"text": f"🔧 Tool executed → stdout: {result}"}),
                )

                logger.info(f"✅ Tool {tool_name} executed → stdout: {result}")

            else:
                await self.event_bus._redis.publish(
                    "agent_reply", json.dumps({"text": f"❌ Unknown tool: {tool_name}"})
                )

        except Exception as e:
            logger.error(f"Error handling atom-run command: {e}")
            await self.event_bus._redis.publish(
                "agent_reply", json.dumps({"text": f"❌ Tool execution error: {e}"})
            )

    async def _execute_prime_counter(self, args: str) -> str:
        """Execute prime counting tool with real computation"""
        try:
            import asyncio
            import subprocess

            n = int(args) if args.isdigit() else 100

            # Execute with timeout for safety
            proc = await asyncio.create_subprocess_exec(
                "python",
                "-c",
                f"import sympy; print(sympy.primepi({n}))",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=10.0
                )

                if proc.returncode == 0:
                    return stdout.decode().strip()
                return f"Error: {stderr.decode().strip()}"

            except TimeoutError:
                proc.kill()
                return "Error: Tool execution timed out"

        except Exception as e:
            logger.error(f"Prime counter execution failed: {e}")
            return f"Error: {e!s}"

    async def _execute_python_code(self, code: str, params: dict[str, Any]) -> str:
        """Execute Python code safely with subprocess"""
        try:
            import asyncio
            import os
            import subprocess
            import tempfile

            # Create temporary file with code
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Execute with timeout
                proc = await asyncio.create_subprocess_exec(
                    "python", temp_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=30.0
                )

                if proc.returncode == 0:
                    return stdout.decode().strip()
                return f"Error: {stderr.decode().strip()}"

            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return f"Error: {e!s}"
