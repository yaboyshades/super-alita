"""
Tools Registry for Super Alita

Manages available computational tools and their execution within the sandbox.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .sandbox import CodeSandbox

logger = logging.getLogger(__name__)


class ToolResult:
    """Result of tool execution"""

    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


class Tool(ABC):
    """Abstract base class for computational tools"""

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters"""
        pass

    def validate_parameters(self, **kwargs: Any) -> bool:
        """Validate input parameters against schema"""
        # Basic validation - can be extended by subclasses
        required_params = {
            name
            for name, config in self.parameters.items()
            if config.get("required", False)
        }

        provided_params = set(kwargs.keys())
        missing_params = required_params - provided_params

        return not missing_params


class PythonCodeTool(Tool):
    """Tool for executing Python code in sandbox"""

    name = "python_code"
    description = "Execute Python code safely in a sandboxed environment"
    parameters = {
        "code": {
            "type": "string",
            "required": True,
            "description": "Python code to execute",
        },
        "context": {
            "type": "object",
            "required": False,
            "description": "Variables available in execution context",
        },
    }

    def __init__(self, sandbox: "CodeSandbox"):
        self.sandbox = sandbox

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute Python code"""
        if not self.validate_parameters(**kwargs):
            return ToolResult(success=False, error="Invalid parameters")

        code = kwargs["code"]
        context = kwargs.get("context", {})

        try:
            result = await self.sandbox.execute_code(code, context)
            return ToolResult(
                success=result["success"],
                data={
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                    "return_value": result["return_value"],
                },
                error=result.get("error"),
                metadata={"execution_time": result.get("execution_time")},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class MathExpressionTool(Tool):
    """Tool for evaluating mathematical expressions"""

    name = "math_expression"
    description = "Evaluate mathematical expressions safely"
    parameters = {
        "expression": {
            "type": "string",
            "required": True,
            "description": "Mathematical expression to evaluate",
        },
        "variables": {
            "type": "object",
            "required": False,
            "description": "Variables to use in expression",
        },
    }

    def __init__(self, sandbox: "CodeSandbox"):
        self.sandbox = sandbox

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Evaluate mathematical expression"""
        if not self.validate_parameters(**kwargs):
            return ToolResult(success=False, error="Invalid parameters")

        expression = kwargs["expression"]
        variables = kwargs.get("variables", {})

        try:
            # Add math functions to context
            import math

            context: dict[str, Any] = {
                **variables,
                **{
                    name: getattr(math, name)
                    for name in dir(math)
                    if not name.startswith("_")
                },
            }

            result = await self.sandbox.evaluate_expression(expression, context)
            return ToolResult(
                success=True, data=result, metadata={"expression": expression}
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class DataAnalysisTool(Tool):
    """Tool for basic data analysis operations"""

    name = "data_analysis"
    description = "Perform basic data analysis operations"
    parameters = {
        "operation": {
            "type": "string",
            "required": True,
            "description": "Analysis operation to perform",
        },
        "data": {"type": "array", "required": True, "description": "Data to analyze"},
        "options": {
            "type": "object",
            "required": False,
            "description": "Additional options",
        },
    }

    def __init__(self, sandbox: "CodeSandbox"):
        self.sandbox = sandbox

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Perform data analysis"""
        if not self.validate_parameters(**kwargs):
            return ToolResult(success=False, error="Invalid parameters")

        operation = kwargs["operation"]
        data = kwargs["data"]
        options = kwargs.get("options", {})

        try:
            # Generate analysis code based on operation
            code = self._generate_analysis_code(operation, data, options)
            context = {"data": data, "options": options}

            result = await self.sandbox.execute_code(code, context)
            return ToolResult(
                success=result["success"],
                data=result["return_value"],
                error=result.get("error"),
                metadata={
                    "operation": operation,
                    "data_length": len(data) if isinstance(data, list) else None,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _generate_analysis_code(
        self, operation: str, data: list[Any], options: dict[str, Any]
    ) -> str:
        """Generate Python code for analysis operation"""
        if operation == "mean":
            return "sum(data) / len(data) if data else 0"
        elif operation == "median":
            return """
sorted_data = sorted(data)
n = len(sorted_data)
if n == 0:
    result = 0
elif n % 2 == 0:
    result = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
else:
    result = sorted_data[n//2]
result
"""
        elif operation == "mode":
            return """
from collections import Counter
if not data:
    result = None
else:
    counts = Counter(data)
    max_count = max(counts.values())
    result = [k for k, v in counts.items() if v == max_count]
result
"""
        elif operation == "std":
            return """
import math
if len(data) < 2:
    result = 0
else:
    mean_val = sum(data) / len(data)
    variance = sum((x - mean_val) ** 2 for x in data) / (len(data) - 1)
    result = math.sqrt(variance)
result
"""
        elif operation == "summary":
            return """
import math

if not data:
    result = {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
else:
    count = len(data)
    mean_val = sum(data) / count
    min_val = min(data)
    max_val = max(data)

    if count > 1:
        # Calculate variance step by step
        deviations = []
        for x in data:
            deviations.append((x - mean_val) ** 2)
        variance = sum(deviations) / (count - 1)
        std_val = math.sqrt(variance)
    else:
        std_val = 0

    result = {
        "count": count,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val
    }

# Return the result
result
"""
        else:
            raise ValueError(f"Unknown operation: {operation}")


class ToolsRegistry:
    """Registry for managing available tools"""

    def __init__(self, sandbox: "CodeSandbox"):
        self.sandbox = sandbox
        self.tools: dict[str, Tool] = {}
        self.logger = logging.getLogger(__name__)

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default computational tools"""
        self.register_tool(PythonCodeTool(self.sandbox))
        self.register_tool(MathExpressionTool(self.sandbox))
        self.register_tool(DataAnalysisTool(self.sandbox))

    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")

    def unregister_tool(self, tool_name: str):
        """Unregister a tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            self.logger.info(f"Unregistered tool: {tool_name}")

    def get_tool(self, tool_name: str) -> Tool | None:
        """Get a tool by name"""
        return self.tools.get(tool_name)

    def list_tools(self) -> dict[str, dict[str, Any]]:
        """List all available tools with their metadata"""
        return {
            name: {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for name, tool in self.tools.items()
        }

    async def execute_tool(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{tool_name}' not found")

        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolResult(success=False, error=str(e))

    async def execute_multiple_tools(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[ToolResult]:
        """Execute multiple tools concurrently"""
        tasks = []
        for call in tool_calls:
            tool_name = call.get("tool")
            parameters = call.get("parameters", {})
            if tool_name:
                tasks.append(self.execute_tool(tool_name, **parameters))

        if tasks:
            return await asyncio.gather(*tasks, return_exceptions=True)
        return []

    def validate_tool_call(
        self, tool_name: str, **kwargs: Any
    ) -> tuple[bool, str | None]:
        """Validate a tool call without executing it"""
        tool = self.get_tool(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' not found"

        if not tool.validate_parameters(**kwargs):
            return False, "Invalid parameters for tool"

        return True, None


async def example_usage():
    """Example of using the tools registry"""
    from .sandbox import CodeSandbox

    # Create sandbox and registry
    sandbox = CodeSandbox()
    registry = ToolsRegistry(sandbox)

    # List available tools
    tools = registry.list_tools()
    print("Available tools:")
    for name, info in tools.items():
        print(f"  {name}: {info['description']}")

    # Execute Python code
    result1 = await registry.execute_tool(
        "python_code", code="x = 5; y = x ** 2; print(f'x={x}, y={y}'); y"
    )
    print("\nPython code result:", result1.to_dict())

    # Execute math expression
    result2 = await registry.execute_tool(
        "math_expression", expression="sqrt(16) + log(e)", variables={"e": 2.718281828}
    )
    print("\nMath expression result:", result2.to_dict())

    # Execute data analysis
    result3 = await registry.execute_tool(
        "data_analysis", operation="summary", data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    print("\nData analysis result:", result3.to_dict())


if __name__ == "__main__":
    asyncio.run(example_usage())
