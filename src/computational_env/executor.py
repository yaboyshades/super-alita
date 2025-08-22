"""
Computational Environment Executor for Super Alita

Orchestrates code execution, tool invocation, and result management.
"""

import asyncio
import logging
from typing import Any

from .sandbox import CodeSandbox
from .tools_registry import ToolsRegistry

logger = logging.getLogger(__name__)


class ExecutionContext:
    """Context for code/tool execution"""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or self._generate_session_id()
        self.variables: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        import uuid

        return str(uuid.uuid4())

    def add_variable(self, name: str, value: Any) -> None:
        """Add a variable to the execution context"""
        self.variables[name] = value

    def get_variable(self, name: str) -> Any:
        """Get a variable from the execution context"""
        return self.variables.get(name)

    def remove_variable(self, name: str) -> None:
        """Remove a variable from the execution context"""
        self.variables.pop(name, None)

    def clear_variables(self) -> None:
        """Clear all variables"""
        self.variables.clear()

    def add_to_history(self, entry: dict[str, Any]) -> None:
        """Add an entry to execution history"""
        self.history.append({**entry, "timestamp": asyncio.get_event_loop().time()})

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get execution history"""
        if limit:
            return self.history[-limit:]
        return self.history.copy()

    def clear_history(self) -> None:
        """Clear execution history"""
        self.history.clear()


class ComputationalEnvironment:
    """
    Main computational environment orchestrator

    Integrates sandbox, tools registry, and execution context
    to provide a complete computational environment.
    """

    def __init__(self, sandbox_timeout: float = 30.0):
        self.sandbox = CodeSandbox(timeout=sandbox_timeout)
        self.tools_registry = ToolsRegistry(self.sandbox)
        self.contexts: dict[str, ExecutionContext] = {}
        self.logger = logging.getLogger(__name__)

    def create_context(self, session_id: str | None = None) -> ExecutionContext:
        """Create a new execution context"""
        context = ExecutionContext(session_id)
        self.contexts[context.session_id] = context
        self.logger.info(f"Created execution context: {context.session_id}")
        return context

    def get_context(self, session_id: str) -> ExecutionContext | None:
        """Get an existing execution context"""
        return self.contexts.get(session_id)

    def remove_context(self, session_id: str) -> None:
        """Remove an execution context"""
        if session_id in self.contexts:
            del self.contexts[session_id]
            self.logger.info(f"Removed execution context: {session_id}")

    async def execute_code(
        self,
        code: str,
        session_id: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute Python code in the computational environment

        Args:
            code: Python code to execute
            session_id: Session ID for context (creates new if None)
            variables: Additional variables to include

        Returns:
            dict: Execution result with context updates
        """
        # Get or create context
        if session_id:
            context = self.get_context(session_id)
            if not context:
                context = self.create_context(session_id)
        else:
            context = self.create_context()

        # Prepare execution environment
        exec_variables = context.variables.copy()
        if variables:
            exec_variables.update(variables)

        try:
            # Execute code
            result = await self.sandbox.execute_code(code, exec_variables)

            # Update context with new/modified variables
            # This is simplified - in practice, we'd need to track what changed
            if result["success"] and result.get("return_value") is not None:
                context.add_variable("_last_result", result["return_value"])

            # Add to history
            context.add_to_history(
                {
                    "type": "code_execution",
                    "code": code,
                    "success": result["success"],
                    "error": result.get("error"),
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                }
            )

            return {
                **result,
                "session_id": context.session_id,
                "context_variables": list(context.variables.keys()),
            }

        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            context.add_to_history(
                {
                    "type": "code_execution",
                    "code": code,
                    "success": False,
                    "error": str(e),
                }
            )

            return {
                "success": False,
                "error": str(e),
                "session_id": context.session_id,
                "stdout": "",
                "stderr": "",
                "return_value": None,
            }

    async def execute_tool(
        self, tool_name: str, parameters: dict[str, Any], session_id: str | None = None
    ) -> dict[str, Any]:
        """
        Execute a tool in the computational environment

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            session_id: Session ID for context

        Returns:
            dict: Tool execution result with context
        """
        # Get or create context
        if session_id:
            context = self.get_context(session_id)
            if not context:
                context = self.create_context(session_id)
        else:
            context = self.create_context()

        try:
            # Execute tool
            result = await self.tools_registry.execute_tool(tool_name, **parameters)

            # Add to history
            context.add_to_history(
                {
                    "type": "tool_execution",
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "success": result.success,
                    "error": result.error,
                    "data": result.data,
                }
            )

            # Update context if tool returned data
            if result.success and result.data is not None:
                context.add_variable(f"_tool_{tool_name}_result", result.data)

            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "metadata": result.metadata,
                "session_id": context.session_id,
                "context_variables": list(context.variables.keys()),
            }

        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            context.add_to_history(
                {
                    "type": "tool_execution",
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "success": False,
                    "error": str(e),
                }
            )

            return {
                "success": False,
                "error": str(e),
                "session_id": context.session_id,
                "data": None,
                "metadata": {},
            }

    async def execute_script_of_thought(
        self, script: str, session_id: str | None = None
    ) -> dict[str, Any]:
        """
        Execute a Script of Thought in the computational environment

        Args:
            script: Script of Thought to execute
            session_id: Session ID for context

        Returns:
            dict: Execution result with all step results
        """
        from ..script_of_thought.interpreter import ScriptOfThoughtInterpreter

        # Get or create context
        if session_id:
            context = self.get_context(session_id)
            if not context:
                context = self.create_context(session_id)
        else:
            context = self.create_context()

        try:
            # Create interpreter with this environment
            interpreter = ScriptOfThoughtInterpreter(self)

            # Execute script
            result = await interpreter.execute_script(script, context.session_id)

            # Add to history
            context.add_to_history(
                {
                    "type": "script_execution",
                    "script": script,
                    "success": result["success"],
                    "error": result.get("error"),
                    "steps_completed": len(result.get("step_results", [])),
                }
            )

            return {
                **result,
                "session_id": context.session_id,
                "context_variables": list(context.variables.keys()),
            }

        except Exception as e:
            self.logger.error(f"Script execution failed: {e}")
            context.add_to_history(
                {
                    "type": "script_execution",
                    "script": script,
                    "success": False,
                    "error": str(e),
                }
            )

            return {
                "success": False,
                "error": str(e),
                "session_id": context.session_id,
                "step_results": [],
            }

    def list_available_tools(self) -> dict[str, dict[str, Any]]:
        """List all available tools"""
        return self.tools_registry.list_tools()

    def get_context_info(self, session_id: str) -> dict[str, Any] | None:
        """Get information about an execution context"""
        context = self.get_context(session_id)
        if not context:
            return None

        return {
            "session_id": context.session_id,
            "variables": list(context.variables.keys()),
            "history_length": len(context.history),
            "metadata": context.metadata,
        }

    def list_contexts(self) -> list[str]:
        """List all active context session IDs"""
        return list(self.contexts.keys())

    async def cleanup_context(self, session_id: str) -> None:
        """Clean up a context and its resources"""
        context = self.get_context(session_id)
        if context:
            context.clear_variables()
            context.clear_history()
            self.remove_context(session_id)

    async def cleanup_all_contexts(self) -> None:
        """Clean up all contexts"""
        for session_id in list(self.contexts.keys()):
            await self.cleanup_context(session_id)


async def example_usage():
    """Example of using the computational environment"""
    env = ComputationalEnvironment()

    # Create a context
    context = env.create_context()
    session_id = context.session_id
    print(f"Created session: {session_id}")

    # Execute some code
    result1 = await env.execute_code(
        "import math; x = 42; y = math.sqrt(x); print(f'sqrt({x}) = {y}'); y",
        session_id=session_id,
    )
    print("Code result:", result1)

    # Execute a tool
    result2 = await env.execute_tool(
        "data_analysis",
        {"operation": "summary", "data": [1, 2, 3, 4, 5]},
        session_id=session_id,
    )
    print("Tool result:", result2)

    # Execute mathematical expression
    result3 = await env.execute_tool(
        "math_expression",
        {"expression": "sin(pi/2) + cos(0)", "variables": {"pi": 3.14159}},
        session_id=session_id,
    )
    print("Math result:", result3)

    # Show context info
    info = env.get_context_info(session_id)
    print("Context info:", info)

    # Cleanup
    await env.cleanup_context(session_id)


if __name__ == "__main__":
    asyncio.run(example_usage())
