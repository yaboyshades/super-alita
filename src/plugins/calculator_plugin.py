#!/usr/bin/env python3
"""
Calculator Plugin for Super Alita
Provides basic arithmetic calculation capabilities
"""

import ast
import logging
import operator
import re
from typing import Any

from src.core.events import ToolCallEvent, ToolResultEvent
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class CalculatorPlugin(PluginInterface):
    """
    Calculator plugin that provides safe arithmetic evaluation.

    Supported operations:
    - Basic arithmetic: +, -, *, /, %, **
    - Parentheses for grouping
    - Mathematical functions: abs, round, min, max

    Security: Uses AST parsing to prevent code injection
    """

    def __init__(self):
        super().__init__()
        self._safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        self._safe_functions = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "int": int,
            "float": float,
        }

    @property
    def name(self) -> str:
        return "calculator"

    async def setup(self, event_bus, store, config: dict[str, Any]) -> None:
        """Initialize the calculator plugin."""
        await super().setup(event_bus, store, config)
        logger.info("CalculatorPlugin setup complete")

    async def start(self) -> None:
        """Start the calculator plugin."""
        await super().start()

        # Subscribe to tool_call events for calculator
        await self.subscribe("tool_call", self._handle_tool_call)

        logger.info("CalculatorPlugin started - ready for arithmetic calculations")

    async def _handle_tool_call(self, event: ToolCallEvent) -> None:
        """Handle calculator tool calls."""
        if not isinstance(event, ToolCallEvent) or event.tool_name != "calculator":
            return  # Not for us

        logger.info(f"Processing calculator tool call: {event.tool_call_id}")

        try:
            # Extract expression from parameters
            expression = (
                event.parameters.get("expression")
                or event.parameters.get("expr")
                or event.parameters.get("input", "")
            )

            if not expression:
                raise ValueError(
                    "No expression provided. Use 'expression', 'expr', or 'input' parameter."
                )

            # Clean and evaluate the expression
            result = self._safe_eval(expression)

            # Emit successful result
            await self._emit_tool_result(
                event,
                success=True,
                result={
                    "expression": expression,
                    "result": result,
                    "type": type(result).__name__,
                },
                message=f"Calculated: {expression} = {result}",
            )

        except Exception as e:
            logger.error(f"Calculator error: {e}")
            await self._emit_tool_result(
                event,
                success=False,
                result=None,
                message=f"Calculation failed: {e!s}",
            )

    def _safe_eval(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression using AST parsing.

        Args:
            expression: Mathematical expression as string

        Returns:
            Numerical result of the calculation

        Raises:
            ValueError: If expression is invalid or unsafe
        """
        # Clean the expression
        expression = expression.strip()

        # Remove common prefixes that might confuse the parser
        if expression.lower().startswith(("calculate", "calc", "=")):
            expression = re.sub(
                r"^(calculate|calc|=)\s*", "", expression, flags=re.IGNORECASE
            )

        try:
            # Parse the expression into an AST
            node = ast.parse(expression, mode="eval")
            return self._eval_node(node.body)

        except SyntaxError:
            raise ValueError(f"Invalid mathematical expression: {expression}") from None
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e!s}") from e

    def _eval_node(self, node):
        """Recursively evaluate AST nodes safely."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        if isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self._safe_operators.get(type(node.op))
            if op is None:
                raise ValueError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )
            return op(operand)
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self._safe_operators.get(type(node.op))
            if op is None:
                raise ValueError(
                    f"Unsupported binary operator: {type(node.op).__name__}"
                )
            return op(left, right)
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self._safe_functions:
                raise ValueError(f"Unsupported function: {func_name}")
            args = [self._eval_node(arg) for arg in node.args]
            return self._safe_functions[func_name](*args)
        if isinstance(node, ast.Name):
            # Only allow mathematical constants
            if node.id == "pi":
                import math

                return math.pi
            if node.id == "e":
                import math

                return math.e
            raise ValueError(f"Unsupported variable: {node.id}")
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    async def _emit_tool_result(
        self, original_event: ToolCallEvent, success: bool, result: Any, message: str
    ) -> None:
        """Emit a ToolResultEvent in response to a tool call."""
        try:
            result_event = ToolResultEvent(
                source_plugin=self.name,
                conversation_id=original_event.conversation_id,
                session_id=original_event.session_id,
                tool_call_id=original_event.tool_call_id,
                success=success,
                result=result if success else {"error": message},
            )

            await self.event_bus.publish(result_event)
            logger.debug(f"Emitted tool result for call {original_event.tool_call_id}")

        except Exception as e:
            logger.error(f"Failed to emit tool result: {e}")

    async def shutdown(self) -> None:
        """Shutdown the calculator plugin."""
        logger.info("CalculatorPlugin shutdown complete")


# Plugin registration for auto-discovery
def create_plugin():
    """Factory function for plugin auto-discovery."""
    return CalculatorPlugin()
