"""
Security-First Dynamic Code Execution for Super Alita
Provides safe execution of dynamically generated code with audit trails
"""

import ast
import inspect
import logging
import re
import types
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuditLog:
    """Audit log entry for dynamic code execution"""

    uuid: str
    code: str
    params: dict[str, Any]
    user_id: str
    context_id: str
    timestamp: str
    execution_result: str | None = None
    error: str | None = None


class SecureCodeExecutor:
    """Security-first dynamic code executor with audit trails"""

    def __init__(self):
        self.audit_logs: list[AuditLog] = []
        self.safe_builtins = {
            # Math operations
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "pow": pow,
            "divmod": divmod,
            # Data structures
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "reversed": reversed,
            # Type checking
            "isinstance": isinstance,
            "type": type,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            # Safe modules (can be extended)
            "math": __import__("math"),
            "json": __import__("json"),
            "random": __import__("random"),
        }

    def sanitize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Sanitize parameters to prevent code injection"""
        sanitized = {}

        for key, value in params.items():
            # Validate key format (alphanumeric + underscore only)
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                logger.warning(f"Skipping invalid parameter key: {key}")
                continue

            # Validate value types (only safe types allowed)
            if isinstance(value, int | float | str | bool | list | dict):
                # For strings, check for potential code injection
                if isinstance(value, str):
                    if any(
                        dangerous in value.lower()
                        for dangerous in [
                            "import",
                            "exec",
                            "eval",
                            "open",
                            "file",
                            "__",
                        ]
                    ):
                        logger.warning(
                            f"Skipping potentially dangerous string parameter: {key}"
                        )
                        continue

                sanitized[key] = value
            else:
                logger.warning(
                    f"Skipping parameter with unsafe type: {key} ({type(value)})"
                )

        return sanitized

    def validate_code_safety(self, code: str) -> bool:
        """Validate that code is safe to execute"""
        try:
            # Parse the code to check for dangerous operations
            tree = ast.parse(code)

            # Check for dangerous node types
            dangerous_nodes = (
                ast.Import,
                ast.ImportFrom,
                ast.Exec,
                ast.Eval,
                ast.Call,  # We'll check calls more specifically
            )

            for node in ast.walk(tree):
                if isinstance(node, dangerous_nodes):
                    if isinstance(node, ast.Call):
                        # Check for dangerous function calls
                        if hasattr(node.func, "id"):
                            if node.func.id in ["eval", "exec", "compile", "open"]:
                                logger.error(
                                    f"Dangerous function call detected: {node.func.id}"
                                )
                                return False
                    else:
                        logger.error(
                            f"Dangerous operation detected: {type(node).__name__}"
                        )
                        return False

            return True

        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            return False

    def safe_exec(
        self, code: str, params: dict[str, Any], function_name: str = None
    ) -> Callable:
        """Execute code in a restricted environment"""
        if not self.validate_code_safety(code):
            raise SecurityError("Code failed safety validation")

        # Create restricted execution environment
        safe_globals = {
            "__builtins__": self.safe_builtins,
            "__name__": "__restricted__",
        }

        # Add sanitized parameters to locals
        safe_locals = params.copy()

        try:
            # Execute the code
            exec(code, safe_globals, safe_locals)

            # Extract the function (try common names if not specified)
            if function_name:
                if function_name not in safe_locals:
                    raise RuntimeError(
                        f"Function '{function_name}' not found in executed code"
                    )
                return safe_locals[function_name]
            # Try to find a callable in the locals
            callables = {
                k: v
                for k, v in safe_locals.items()
                if callable(v) and not k.startswith("_")
            }

            if not callables:
                raise RuntimeError("No callable function found in executed code")

            if len(callables) == 1:
                return list(callables.values())[0]
            # Return the first non-builtin callable
            for name, func in callables.items():
                if not isinstance(func, types.BuiltinFunctionType):
                    return func

            raise RuntimeError("Multiple functions found, specify function_name")

        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            raise

    def log_audit(
        self,
        code: str,
        params: dict[str, Any],
        user_id: str,
        context_id: str,
        execution_result: str = None,
        error: str = None,
    ) -> AuditLog:
        """Create audit log entry"""
        audit_log = AuditLog(
            uuid=str(uuid.uuid4()),
            code=code,
            params=params,
            user_id=user_id,
            context_id=context_id,
            timestamp=datetime.now().isoformat(),
            execution_result=execution_result,
            error=error,
        )

        self.audit_logs.append(audit_log)
        logger.info(f"Audit log created: {audit_log.uuid}")

        return audit_log

    def execute_with_audit(
        self,
        code: str,
        params: dict[str, Any],
        user_id: str,
        context_id: str,
        function_name: str = None,
    ) -> tuple[Any, AuditLog]:
        """Execute code with full audit trail"""
        sanitized_params = self.sanitize_params(params)
        execution_result = None
        error = None

        try:
            func = self.safe_exec(code, sanitized_params, function_name)
            execution_result = f"Function created successfully: {func.__name__ if hasattr(func, '__name__') else 'anonymous'}"
            return func, self.log_audit(
                code, sanitized_params, user_id, context_id, execution_result
            )

        except Exception as e:
            error = str(e)
            logger.error(f"Code execution failed: {error}")
            self.log_audit(code, sanitized_params, user_id, context_id, None, error)
            raise CodeExecutionError(f"Execution failed: {error}") from e

    def get_audit_logs(
        self, user_id: str = None, context_id: str = None
    ) -> list[AuditLog]:
        """Retrieve audit logs with optional filtering"""
        logs = self.audit_logs

        if user_id:
            logs = [log for log in logs if log.user_id == user_id]

        if context_id:
            logs = [log for log in logs if log.context_id == context_id]

        return logs

    def run_unit_test(
        self, code: str, params: dict[str, Any], test_inputs: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Run automated unit test on generated code"""
        try:
            func = self.safe_exec(code, params)

            # Use provided test inputs or default params
            test_params = test_inputs or params

            # Execute the function with test parameters
            test_result = func(**test_params)

            return {
                "status": "success",
                "result": test_result,
                "function_name": getattr(func, "__name__", "anonymous"),
            }

        except Exception as e:
            return {
                "status": "failure",
                "error": str(e),
                "error_type": type(e).__name__,
            }


class DynamicToolRegistry:
    """Registry for managing dynamic tools with persistence"""

    def __init__(self):
        self.registry: dict[str, dict[str, Any]] = {}
        self.usage_stats: dict[str, int] = {}

    def register_tool(
        self,
        tool_name: str,
        code: str,
        params_schema: dict[str, Any] = None,
        description: str = None,
        author: str = None,
    ) -> None:
        """Register a dynamic tool with metadata"""
        self.registry[tool_name] = {
            "code": code,
            "params_schema": params_schema or {},
            "description": description or f"Dynamic tool: {tool_name}",
            "author": author,
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
        }

        self.usage_stats[tool_name] = 0
        logger.info(f"Registered dynamic tool: {tool_name}")

    def get_tool(self, tool_name: str) -> dict[str, Any] | None:
        """Retrieve tool metadata and code"""
        return self.registry.get(tool_name)

    def use_tool(self, tool_name: str) -> None:
        """Track tool usage"""
        if tool_name in self.usage_stats:
            self.usage_stats[tool_name] += 1

    def get_popular_tools(self, limit: int = 10) -> list[tuple[str, int]]:
        """Get most frequently used tools"""
        return sorted(self.usage_stats.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]

    def list_tools(self) -> list[str]:
        """List all registered tools"""
        return list(self.registry.keys())

    def reload(self) -> None:
        """Re-scan plugin directory and register new tools."""
        import importlib
        import sys
        from pathlib import Path

        # Reload any dynamically created tool modules
        plugins_dir = Path("src/plugins")
        for tool_file in plugins_dir.glob("*_atom.py"):
            module_name = f"src.plugins.{tool_file.stem}"

            if module_name in sys.modules:
                try:
                    importlib.reload(sys.modules[module_name])
                    logger.info(f"Reloaded module: {module_name}")
                except Exception as e:
                    logger.warning(f"Failed to reload module {module_name}: {e}")
            else:
                try:
                    importlib.import_module(module_name)
                    logger.info(f"Loaded new module: {module_name}")
                except Exception as e:
                    logger.warning(f"Failed to load module {module_name}: {e}")

        logger.info(f"Registry reload complete. Available tools: {self.list_tools()}")


class SecurityError(Exception):
    """Raised when code fails security validation"""


class CodeExecutionError(Exception):
    """Raised when code execution fails"""


# Global instances
_secure_executor = SecureCodeExecutor()
_tool_registry = DynamicToolRegistry()
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool_exec")


def get_secure_executor() -> SecureCodeExecutor:
    """Get global secure executor instance"""
    return _secure_executor


def get_tool_registry() -> DynamicToolRegistry:
    """Get global tool registry instance"""
    return _tool_registry


def execute_tool(tool_fn: Callable, kwargs: dict[str, Any], timeout: int = 5) -> Any:
    """
    Execute a tool (sync or async) with a hard wall-clock timeout.

    Args:
        tool_fn: The tool function to execute
        kwargs: Parameters to pass to the tool
        timeout: Maximum execution time in seconds

    Returns:
        The result of the tool execution

    Raises:
        TimeoutError: If execution exceeds timeout
        Exception: Any exception raised by the tool
    """
    if inspect.iscoroutinefunction(tool_fn):
        import asyncio

        return asyncio.run(asyncio.wait_for(tool_fn(**kwargs), timeout))
    # Synchronous tool: run in thread so we can time it out
    fut = _pool.submit(tool_fn, **kwargs)
    return fut.result(timeout=timeout)
