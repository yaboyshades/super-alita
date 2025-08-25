"""
Dynamic Tool Protocol for Super Alita

Provides live schema/contract-first tool creation with runtime tool registration
and validation. Supports dynamic tool generation from natural language descriptions.
"""

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import jsonschema

logger = logging.getLogger(__name__)


class ToolParameterType(Enum):
    """Supported tool parameter types"""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"  # Changed from FLOAT to NUMBER for JSON Schema compatibility
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""

    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    default: Any = None
    enum_values: list[Any] | None = None
    properties: dict[str, "ToolParameter"] | None = None


@dataclass
class ToolSchema:
    """Schema definition for a dynamic tool"""

    name: str
    description: str
    parameters: list[ToolParameter]
    return_type: str
    examples: list[dict[str, Any]]
    version: str = "1.0.0"
    created_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format"""
        properties: dict[str, Any] = {}
        required = []

        for param in self.parameters:
            prop_def: dict[str, Any] = {
                "type": param.type.value,
                "description": param.description,
            }

            if param.enum_values:
                prop_def["enum"] = param.enum_values

            if param.type == ToolParameterType.ARRAY and param.properties:
                prop_def["items"] = {
                    "type": "object",
                    "properties": {
                        p.name: {"type": p.type.value, "description": p.description}
                        for p in param.properties.values()
                    },
                }
            elif param.type == ToolParameterType.OBJECT and param.properties:
                prop_def["properties"] = {
                    p.name: {"type": p.type.value, "description": p.description}
                    for p in param.properties.values()
                }

            properties[param.name] = prop_def

            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


class DynamicTool:
    """A dynamically created tool with runtime validation"""

    def __init__(self, schema: ToolSchema, implementation: Callable):
        self.schema = schema
        self.implementation = implementation
        self.json_schema = schema.to_json_schema()
        self.call_count = 0
        self.last_called = None

    async def execute(self, **kwargs) -> dict[str, Any]:
        """Execute the tool with parameter validation"""
        try:
            # Validate parameters against schema
            jsonschema.validate(kwargs, self.json_schema)

            # Update usage statistics
            self.call_count += 1
            self.last_called = datetime.now(UTC)

            # Execute the tool
            if inspect.iscoroutinefunction(self.implementation):
                result = await self.implementation(**kwargs)
            else:
                result = self.implementation(**kwargs)

            return {
                "success": True,
                "result": result,
                "tool_name": self.schema.name,
                "execution_time": datetime.now(UTC).isoformat(),
            }

        except jsonschema.ValidationError as e:
            return {
                "success": False,
                "error": f"Parameter validation failed: {e.message}",
                "tool_name": self.schema.name,
            }
        except Exception as e:
            logger.error(f"Error executing dynamic tool {self.schema.name}: {e}")
            return {"success": False, "error": str(e), "tool_name": self.schema.name}


class DynamicToolRegistry:
    """Registry for managing dynamic tools"""

    def __init__(self):
        self.tools: dict[str, DynamicTool] = {}
        self.schemas: dict[str, ToolSchema] = {}
        self.creation_log: list[dict[str, Any]] = []

    def register_tool(self, schema: ToolSchema, implementation: Callable) -> bool:
        """Register a new dynamic tool"""
        try:
            # Validate the implementation signature matches schema
            sig = inspect.signature(implementation)
            schema_params = {p.name for p in schema.parameters}
            impl_params = set(sig.parameters.keys())

            if not schema_params.issubset(impl_params):
                missing = schema_params - impl_params
                raise ValueError(f"Implementation missing parameters: {missing}")

            # Create and register the tool
            tool = DynamicTool(schema, implementation)
            self.tools[schema.name] = tool
            self.schemas[schema.name] = schema

            # Log the creation
            self.creation_log.append(
                {
                    "tool_name": schema.name,
                    "created_at": schema.created_at.isoformat()
                    if schema.created_at
                    else datetime.now(UTC).isoformat(),
                    "version": schema.version,
                    "parameter_count": len(schema.parameters),
                }
            )

            logger.info(f"Registered dynamic tool: {schema.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register tool {schema.name}: {e}")
            return False

    def get_tool(self, name: str) -> DynamicTool | None:
        """Get a registered tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names"""
        return list(self.tools.keys())

    def get_tool_schema(self, name: str) -> ToolSchema | None:
        """Get the schema for a tool"""
        return self.schemas.get(name)

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the registry"""
        if name in self.tools:
            del self.tools[name]
            del self.schemas[name]
            logger.info(f"Removed dynamic tool: {name}")
            return True
        return False

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_tools": len(self.tools),
            "creation_log": self.creation_log,
            "tool_usage": {
                name: {
                    "call_count": tool.call_count,
                    "last_called": tool.last_called.isoformat()
                    if tool.last_called
                    else None,
                }
                for name, tool in self.tools.items()
            },
        }


class ToolSchemaGenerator:
    """Generates tool schemas from natural language descriptions"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def generate_schema_from_description(
        self, description: str, examples: list[str] | None = None
    ) -> ToolSchema | None:
        """Generate a tool schema from natural language description"""
        # This would integrate with an LLM to parse natural language
        # For now, we'll provide a simple template-based approach

        # Extract basic information from description
        name = self._extract_tool_name(description)
        parameters = self._extract_parameters(description)
        return_type = self._extract_return_type(description)

        if not name:
            return None

        return ToolSchema(
            name=name,
            description=description,
            parameters=parameters,
            return_type=return_type,
            examples=[],  # Convert examples to proper format later
        )

    def _extract_tool_name(self, description: str) -> str:
        """Extract tool name from description"""
        # Simple extraction - would be enhanced with LLM processing
        words = description.lower().split()
        if "called" in words:
            idx = words.index("called")
            if idx + 1 < len(words):
                return words[idx + 1].strip('",.')

        # Fallback to first few words
        return "_".join(words[:3]).replace(" ", "_")

    def _extract_parameters(self, description: str) -> list[ToolParameter]:
        """Extract parameters from description"""
        # Simple pattern matching - would be enhanced with LLM processing
        parameters = []

        # Look for common parameter patterns
        if "takes" in description.lower() or "accepts" in description.lower():
            # This would be much more sophisticated with LLM parsing
            parameters.append(
                ToolParameter(
                    name="input",
                    type=ToolParameterType.STRING,
                    description="Input parameter extracted from description",
                    required=True,
                )
            )

        return parameters

    def _extract_return_type(self, description: str) -> str:
        """Extract return type from description"""
        # Simple pattern matching
        if "returns" in description.lower():
            return "object"
        return "string"


# Global registry instance
dynamic_tool_registry = DynamicToolRegistry()


async def create_tool_from_description(
    description: str, implementation: Callable
) -> bool:
    """Convenience function to create a tool from description and implementation"""
    generator = ToolSchemaGenerator()
    schema = await generator.generate_schema_from_description(description)

    if schema:
        return dynamic_tool_registry.register_tool(schema, implementation)

    return False


# Example usage and built-in tools
async def example_dynamic_tool_usage():
    """Example of creating and using dynamic tools"""

    # Create a simple calculator tool
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers together"""
        return a + b

    # Define schema manually
    calculator_schema = ToolSchema(
        name="add_calculator",
        description="Adds two numbers together",
        parameters=[
            ToolParameter("a", ToolParameterType.NUMBER, "First number", required=True),
            ToolParameter(
                "b", ToolParameterType.NUMBER, "Second number", required=True
            ),
        ],
        return_type="float",
        examples=[{"a": 5.0, "b": 3.0, "expected": 8.0}],
    )

    # Register the tool
    success = dynamic_tool_registry.register_tool(calculator_schema, add_numbers)
    if success:
        print("Calculator tool registered successfully")

        # Execute the tool
        tool = dynamic_tool_registry.get_tool("add_calculator")
        if tool:
            result = await tool.execute(a=10.5, b=7.3)
            print(f"Calculator result: {result}")

    # Print registry stats
    stats = dynamic_tool_registry.get_registry_stats()
    print(f"Registry stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    asyncio.run(example_dynamic_tool_usage())
