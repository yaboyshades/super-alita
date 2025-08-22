"""
Secure Computational Environment for Super Alita

Provides safe code execution with sandboxing, timeouts, and restricted imports.
"""

import ast
import asyncio
import logging
import sys
import traceback
from io import StringIO
from typing import Any

logger = logging.getLogger(__name__)


class CodeSandbox:
    """
    Secure Python code execution sandbox

    Features:
    - AST parsing for syntax validation
    - Restricted builtins and imports
    - Execution timeout
    - Captured stdout/stderr
    - Safe eval for expressions
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Safe builtins - remove dangerous functions
        self.safe_builtins = {
            name: builtin
            for name, builtin in __builtins__.items()
            if name
            not in {
                "eval",
                "exec",
                "compile",
                "open",
                "input",
                "raw_input",
                "file",
                "reload",
                "__import__",
                "help",
                "dir",
                "globals",
                "locals",
                "vars",
                "exit",
                "quit",
            }
        }

        # Add safe functions
        self.safe_builtins.update(
            {
                "abs": abs,
                "all": all,
                "any": any,
                "bin": bin,
                "bool": bool,
                "chr": chr,
                "dict": dict,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "hex": hex,
                "int": int,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "oct": oct,
                "ord": ord,
                "pow": pow,
                "print": print,
                "range": range,
                "reversed": reversed,
                "round": round,
                "set": set,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "type": type,
                "zip": zip,
            }
        )

        # Allowed modules
        self.allowed_modules = {
            "math",
            "random",
            "datetime",
            "json",
            "collections",
            "itertools",
            "functools",
            "operator",
            "re",
            "string",
            "decimal",
            "fractions",
            "statistics",
            "pathlib",
            "pandas",
            "numpy",
            "matplotlib",
            "seaborn",
            "plotly",
        }

    def validate_code(self, code: str) -> tuple[bool, str | None]:
        """
        Validate code using AST parsing

        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)

            # Check for dangerous operations
            dangerous_nodes: list[str] = []

            for node in ast.walk(tree):
                # Check for imports
                if isinstance(node, ast.Import | ast.ImportFrom):
                    if isinstance(node, ast.ImportFrom):
                        module_name = node.module
                    else:
                        module_name = node.names[0].name if node.names else None

                    if module_name and not self._is_module_allowed(module_name):
                        dangerous_nodes.append(f"Forbidden import: {module_name}")

                # Check for dangerous function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in {"eval", "exec", "compile", "__import__"}:
                            dangerous_nodes.append(f"Forbidden function: {func_name}")

                # Check for attribute access to dangerous attributes
                elif isinstance(node, ast.Attribute):
                    attr_name = node.attr
                    if attr_name in {
                        "__globals__",
                        "__locals__",
                        "__dict__",
                        "__class__",
                    }:
                        dangerous_nodes.append(
                            f"Forbidden attribute access: {attr_name}"
                        )

            if dangerous_nodes:
                return False, "; ".join(dangerous_nodes)

            return True, None

        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"

    def _is_module_allowed(self, module_name: str) -> bool:
        """Check if module is in allowlist"""
        # Check full module name
        if module_name in self.allowed_modules:
            return True

        # Check parent modules
        parts = module_name.split(".")
        for i in range(len(parts)):
            parent_module = ".".join(parts[: i + 1])
            if parent_module in self.allowed_modules:
                return True

        return False

    async def execute_code(
        self, code: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute Python code safely

        Args:
            code: Python code to execute
            context: Additional variables to include in execution context

        Returns:
            dict: Execution result with stdout, stderr, return_value, success
        """
        # Validate code first
        is_valid, error_msg = self.validate_code(code)
        if not is_valid:
            return {
                "success": False,
                "error": f"Code validation failed: {error_msg}",
                "stdout": "",
                "stderr": error_msg or "",
                "return_value": None,
            }

        # Prepare execution environment
        execution_globals = self.safe_builtins.copy()
        execution_locals = context.copy() if context else {}

        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_stdout = StringIO()
        captured_stderr = StringIO()

        try:
            # Redirect output
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr

            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_code_async(code, execution_globals, execution_locals),
                timeout=self.timeout,
            )

            return {
                "success": True,
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "return_value": result,
                "error": None,
            }

        except TimeoutError:
            return {
                "success": False,
                "error": f"Code execution timed out after {self.timeout} seconds",
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "return_value": None,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue() + traceback.format_exc(),
                "return_value": None,
            }
        finally:
            # Restore output
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    async def _execute_code_async(
        self, code: str, globals_dict: dict[str, Any], locals_dict: dict[str, Any]
    ) -> Any:
        """Execute code in async context"""
        # Parse code to determine if it's an expression or statement
        try:
            # Try as expression first
            tree = ast.parse(code, mode="eval")
            return eval(compile(tree, "<string>", "eval"), globals_dict, locals_dict)
        except SyntaxError:
            # Not an expression, execute as statements
            tree = ast.parse(code, mode="exec")
            exec(compile(tree, "<string>", "exec"), globals_dict, locals_dict)

            # Try to return the last expression if any
            last_expr = None
            for node in ast.walk(tree):
                if isinstance(node, ast.Expr):
                    last_expr = node

            if last_expr and isinstance(
                last_expr.value, ast.Name | ast.Call | ast.BinOp
            ):
                try:
                    return eval(
                        compile(ast.Expression(last_expr.value), "<string>", "eval"),
                        globals_dict,
                        locals_dict,
                    )
                except Exception:
                    pass

            return None

    async def evaluate_expression(
        self, expression: str, context: dict[str, Any] | None = None
    ) -> Any:
        """
        Safely evaluate a Python expression

        Args:
            expression: Python expression to evaluate
            context: Variables available in expression

        Returns:
            Evaluation result
        """
        # Validate as expression
        try:
            ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression: {e}") from e

        # Prepare context
        execution_globals = self.safe_builtins.copy()
        execution_locals = context.copy() if context else {}

        try:
            result = await asyncio.wait_for(
                self._eval_expression_async(
                    expression, execution_globals, execution_locals
                ),
                timeout=self.timeout,
            )
            return result
        except TimeoutError as e:
            raise RuntimeError(
                f"Expression evaluation timed out after {self.timeout} seconds"
            ) from e

    async def _eval_expression_async(
        self, expression: str, globals_dict: dict[str, Any], locals_dict: dict[str, Any]
    ) -> Any:
        """Evaluate expression in async context"""
        tree = ast.parse(expression, mode="eval")
        return eval(compile(tree, "<string>", "eval"), globals_dict, locals_dict)


# Custom import hook for additional security
class RestrictedImporter:
    """Custom import hook that restricts module imports"""

    def __init__(self, allowed_modules: set[str]):
        self.allowed_modules = allowed_modules

    def __call__(
        self, name: str, globals_dict=None, locals_dict=None, fromlist=(), level=0
    ):
        if not self._is_module_allowed(name):
            raise ImportError(f"Import of module '{name}' is not allowed")

        return __import__(name, globals_dict, locals_dict, fromlist, level)

    def _is_module_allowed(self, module_name: str) -> bool:
        """Check if module is allowed"""
        if module_name in self.allowed_modules:
            return True

        # Check parent modules
        parts = module_name.split(".")
        for i in range(len(parts)):
            parent_module = ".".join(parts[: i + 1])
            if parent_module in self.allowed_modules:
                return True

        return False


async def example_usage():
    """Example of using the code sandbox"""
    sandbox = CodeSandbox(timeout=5.0)

    # Test valid code
    code1 = """
import math
x = 5
y = math.sqrt(x)
print(f"Square root of {x} is {y}")
result = x + y
"""

    result1 = await sandbox.execute_code(code1)
    print("Result 1:", result1)

    # Test invalid code (should be rejected)
    code2 = """
import os
os.system('ls')
"""

    result2 = await sandbox.execute_code(code2)
    print("Result 2:", result2)

    # Test expression evaluation
    try:
        expr_result = await sandbox.evaluate_expression("2 + 3 * 4", {"x": 10})
        print("Expression result:", expr_result)
    except Exception as e:
        print("Expression error:", e)


if __name__ == "__main__":
    asyncio.run(example_usage())
