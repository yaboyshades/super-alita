# Tool Implementations - Agent Instructions

## Overview
The `src/tools/` directory contains specific tool implementations for Super Alita:
- **Core Tools** - Essential tools for system operation
- **External Tools** - Integrations with external services
- **Custom Tools** - User-defined and dynamically generated tools
- **Utility Tools** - Helper tools for common operations

## Key Files & Responsibilities

### Tool Components
- Core tool implementations (to be added)
- Tool registration and discovery utilities
- Tool execution and management systems

## Development Guidelines

### Tool Implementation Pattern
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pydantic import BaseModel, Field
from src.core.plugin_interface import PluginInterface

class ToolInput(BaseModel):
    """Base input schema for tools"""
    pass

class ToolOutput(BaseModel):
    """Base output schema for tools"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BaseTool(ABC):
    """Base class for all tools"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.version = "1.0.0"
        self.tags: List[str] = []

    @abstractmethod
    async def execute(self, input_data: ToolInput) -> ToolOutput:
        """Execute the tool with given input"""
        pass

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Get tool input schema"""
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """Get tool output schema"""
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input against schema"""
        try:
            schema_class = self._get_input_model_class()
            schema_class(**input_data)
            return True
        except Exception:
            return False

    def add_tag(self, tag: str):
        """Add tag to tool"""
        if tag not in self.tags:
            self.tags.append(tag)

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata"""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'tags': self.tags,
            'input_schema': self.get_input_schema(),
            'output_schema': self.get_output_schema()
        }
```

### Calculator Tool Example
```python
from pydantic import Field
import operator
from typing import Union

class CalculatorInput(ToolInput):
    """Calculator tool input"""
    operation: str = Field(..., description="Mathematical operation (+, -, *, /, ^)")
    operand1: Union[int, float] = Field(..., description="First operand")
    operand2: Union[int, float] = Field(..., description="Second operand")

class CalculatorOutput(ToolOutput):
    """Calculator tool output"""
    result: Union[int, float] = None

class CalculatorTool(BaseTool):
    """Basic calculator tool"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs basic mathematical operations"
        )
        self.add_tag("math")
        self.add_tag("utility")

        self.operations = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '^': operator.pow
        }

    async def execute(self, input_data: CalculatorInput) -> CalculatorOutput:
        """Execute calculation"""
        try:
            if input_data.operation not in self.operations:
                return CalculatorOutput(
                    success=False,
                    error=f"Unsupported operation: {input_data.operation}"
                )

            if input_data.operation == '/' and input_data.operand2 == 0:
                return CalculatorOutput(
                    success=False,
                    error="Division by zero"
                )

            operation_func = self.operations[input_data.operation]
            result = operation_func(input_data.operand1, input_data.operand2)

            return CalculatorOutput(
                success=True,
                result=result,
                metadata={
                    'operation': input_data.operation,
                    'operands': [input_data.operand1, input_data.operand2]
                }
            )

        except Exception as e:
            return CalculatorOutput(
                success=False,
                error=f"Calculation error: {str(e)}"
            )

    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema"""
        return CalculatorInput.model_json_schema()

    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema"""
        return CalculatorOutput.model_json_schema()

    def _get_input_model_class(self):
        return CalculatorInput
```

### Web Search Tool Example
```python
import aiohttp
from urllib.parse import quote
from typing import List

class WebSearchInput(ToolInput):
    """Web search tool input"""
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, description="Maximum number of results")
    include_snippets: bool = Field(True, description="Include result snippets")

class SearchResult(BaseModel):
    """Single search result"""
    title: str
    url: str
    snippet: Optional[str] = None

class WebSearchOutput(ToolOutput):
    """Web search tool output"""
    results: List[SearchResult] = Field(default_factory=list)
    total_results: int = 0

class WebSearchTool(BaseTool):
    """Web search tool using external API"""

    def __init__(self, api_key: str, search_engine: str = "google"):
        super().__init__(
            name="web_search",
            description="Search the web for information"
        )
        self.api_key = api_key
        self.search_engine = search_engine
        self.add_tag("search")
        self.add_tag("information")

    async def execute(self, input_data: WebSearchInput) -> WebSearchOutput:
        """Execute web search"""
        try:
            if not input_data.query.strip():
                return WebSearchOutput(
                    success=False,
                    error="Empty search query"
                )

            # Perform search via API
            results = await self._perform_search(
                input_data.query,
                input_data.max_results,
                input_data.include_snippets
            )

            return WebSearchOutput(
                success=True,
                results=results,
                total_results=len(results),
                metadata={
                    'query': input_data.query,
                    'search_engine': self.search_engine
                }
            )

        except Exception as e:
            return WebSearchOutput(
                success=False,
                error=f"Search error: {str(e)}"
            )

    async def _perform_search(
        self, query: str, max_results: int, include_snippets: bool
    ) -> List[SearchResult]:
        """Perform actual web search"""
        results = []

        async with aiohttp.ClientSession() as session:
            # Example API call (replace with actual search API)
            search_url = f"https://api.example-search.com/search"
            params = {
                'q': query,
                'num': max_results,
                'key': self.api_key
            }

            async with session.get(search_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Search API error: {response.status}")

                data = await response.json()

                for item in data.get('items', []):
                    result = SearchResult(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', '') if include_snippets else None
                    )
                    results.append(result)

        return results

    def get_input_schema(self) -> Dict[str, Any]:
        return WebSearchInput.model_json_schema()

    def get_output_schema(self) -> Dict[str, Any]:
        return WebSearchOutput.model_json_schema()

    def _get_input_model_class(self):
        return WebSearchInput
```

### Tool Registry and Management
```python
from typing import Dict, List, Type, Optional
import importlib
import inspect

class ToolRegistry:
    """Registry for managing tools"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_categories: Dict[str, List[str]] = {}

    def register_tool(self, tool: BaseTool, category: str = "general"):
        """Register a tool"""
        self.tools[tool.name] = tool

        if category not in self.tool_categories:
            self.tool_categories[category] = []
        self.tool_categories[category].append(tool.name)

        logger.info(f"Registered tool: {tool.name} in category: {category}")

    def unregister_tool(self, tool_name: str):
        """Unregister a tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]

            # Remove from categories
            for category, tools in self.tool_categories.items():
                if tool_name in tools:
                    tools.remove(tool_name)

            logger.info(f"Unregistered tool: {tool_name}")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(tool_name)

    def list_tools(self, category: Optional[str] = None, tag: Optional[str] = None) -> List[str]:
        """List available tools"""
        if category:
            return self.tool_categories.get(category, [])

        if tag:
            return [
                name for name, tool in self.tools.items()
                if tag in tool.tags
            ]

        return list(self.tools.keys())

    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool metadata"""
        tool = self.get_tool(tool_name)
        return tool.get_metadata() if tool else None

    def discover_tools(self, module_path: str):
        """Discover and register tools from module"""
        try:
            module = importlib.import_module(module_path)

            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, BaseTool) and
                    obj != BaseTool):

                    # Instantiate and register tool
                    tool_instance = obj()
                    self.register_tool(tool_instance)

        except Exception as e:
            logger.error(f"Failed to discover tools from {module_path}: {e}")

class ToolExecutor:
    """Tool execution management"""

    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry
        self.execution_history: List[Dict[str, Any]] = []

    async def execute_tool(
        self, tool_name: str, input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolOutput:
        """Execute tool with input data"""
        tool = self.registry.get_tool(tool_name)

        if not tool:
            return ToolOutput(
                success=False,
                error=f"Tool not found: {tool_name}"
            )

        # Validate input
        if not tool.validate_input(input_data):
            return ToolOutput(
                success=False,
                error=f"Invalid input for tool: {tool_name}"
            )

        start_time = time.time()

        try:
            # Create input object
            input_model_class = tool._get_input_model_class()
            tool_input = input_model_class(**input_data)

            # Execute tool
            result = await tool.execute(tool_input)

            execution_time = time.time() - start_time

            # Record execution
            self._record_execution(tool_name, input_data, result, execution_time, context)

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            error_result = ToolOutput(
                success=False,
                error=f"Tool execution error: {str(e)}"
            )

            self._record_execution(tool_name, input_data, error_result, execution_time, context)

            return error_result

    def _record_execution(
        self, tool_name: str, input_data: Dict[str, Any],
        result: ToolOutput, execution_time: float, context: Optional[Dict[str, Any]]
    ):
        """Record tool execution for history/analytics"""
        execution_record = {
            'tool_name': tool_name,
            'input_data': input_data,
            'result': result.dict(),
            'execution_time': execution_time,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'context': context or {}
        }

        self.execution_history.append(execution_record)

        # Keep only recent history (e.g., last 1000 executions)
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics"""
        if not self.execution_history:
            return {}

        stats = {
            'total_executions': len(self.execution_history),
            'successful_executions': sum(
                1 for record in self.execution_history
                if record['result']['success']
            ),
            'average_execution_time': sum(
                record['execution_time'] for record in self.execution_history
            ) / len(self.execution_history),
            'tool_usage': {}
        }

        # Tool usage statistics
        tool_usage = {}
        for record in self.execution_history:
            tool_name = record['tool_name']
            if tool_name not in tool_usage:
                tool_usage[tool_name] = {'count': 0, 'total_time': 0}
            tool_usage[tool_name]['count'] += 1
            tool_usage[tool_name]['total_time'] += record['execution_time']

        for tool_name, usage in tool_usage.items():
            usage['average_time'] = usage['total_time'] / usage['count']

        stats['tool_usage'] = tool_usage

        return stats
```

## Testing Guidelines

### Tool Testing Pattern
```python
import pytest
from unittest.mock import AsyncMock, patch
from src.tools.calculator_tool import CalculatorTool, CalculatorInput

@pytest.mark.asyncio
async def test_calculator_tool_basic_operations():
    """Test calculator tool basic operations"""
    calculator = CalculatorTool()

    # Test addition
    input_data = CalculatorInput(operation="+", operand1=5, operand2=3)
    result = await calculator.execute(input_data)

    assert result.success is True
    assert result.result == 8

    # Test multiplication
    input_data = CalculatorInput(operation="*", operand1=4, operand2=7)
    result = await calculator.execute(input_data)

    assert result.success is True
    assert result.result == 28

@pytest.mark.asyncio
async def test_calculator_tool_error_handling():
    """Test calculator tool error handling"""
    calculator = CalculatorTool()

    # Test division by zero
    input_data = CalculatorInput(operation="/", operand1=10, operand2=0)
    result = await calculator.execute(input_data)

    assert result.success is False
    assert "Division by zero" in result.error

    # Test invalid operation
    input_data = CalculatorInput(operation="invalid", operand1=1, operand2=2)
    result = await calculator.execute(input_data)

    assert result.success is False
    assert "Unsupported operation" in result.error

@pytest.mark.asyncio
async def test_web_search_tool():
    """Test web search tool"""
    # Mock the HTTP response
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            'items': [
                {
                    'title': 'Test Result',
                    'link': 'https://example.com',
                    'snippet': 'Test snippet'
                }
            ]
        }
        mock_get.return_value.__aenter__.return_value = mock_response

        search_tool = WebSearchTool(api_key="test_key")
        input_data = WebSearchInput(query="test query", max_results=5)

        result = await search_tool.execute(input_data)

        assert result.success is True
        assert len(result.results) == 1
        assert result.results[0].title == "Test Result"

def test_tool_registry():
    """Test tool registry functionality"""
    registry = ToolRegistry()
    calculator = CalculatorTool()

    # Test registration
    registry.register_tool(calculator, "math")
    assert "calculator" in registry.tools
    assert "calculator" in registry.tool_categories["math"]

    # Test retrieval
    retrieved_tool = registry.get_tool("calculator")
    assert retrieved_tool == calculator

    # Test listing
    math_tools = registry.list_tools(category="math")
    assert "calculator" in math_tools

    # Test metadata
    metadata = registry.get_tool_metadata("calculator")
    assert metadata["name"] == "calculator"
    assert "math" in metadata["tags"]

@pytest.mark.asyncio
async def test_tool_executor():
    """Test tool executor"""
    registry = ToolRegistry()
    calculator = CalculatorTool()
    registry.register_tool(calculator)

    executor = ToolExecutor(registry)

    # Test successful execution
    input_data = {"operation": "+", "operand1": 5, "operand2": 3}
    result = await executor.execute_tool("calculator", input_data)

    assert result.success is True
    assert result.result == 8

    # Check execution history
    assert len(executor.execution_history) == 1
    assert executor.execution_history[0]["tool_name"] == "calculator"

    # Test statistics
    stats = executor.get_execution_stats()
    assert stats["total_executions"] == 1
    assert stats["successful_executions"] == 1
```

### Integration Testing
```python
@pytest.mark.integration
async def test_tool_plugin_integration():
    """Test tool integration with plugin system"""
    from src.plugins.tool_executor_plugin import ToolExecutorPlugin

    event_bus = AsyncMock()
    plugin = ToolExecutorPlugin(event_bus)

    # Initialize plugin with tools
    await plugin.initialize()

    # Test tool execution via plugin
    event = create_event(
        "tool_execution_request",
        tool_name="calculator",
        parameters={"operation": "+", "operand1": 2, "operand2": 3}
    )

    await plugin.handle_event(event)

    # Verify tool execution event was emitted
    event_bus.publish.assert_called()
    emitted_event = event_bus.publish.call_args[0][0]
    assert emitted_event["type"] == "tool_execution_result"
    assert emitted_event["data"]["success"] is True
```

## Security Guidelines

### Input Validation and Sanitization
```python
import re
from typing import Any, Dict, List

class ToolSecurityValidator:
    """Security validation for tool inputs"""

    def __init__(self):
        self.dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'__import__',
            r'open\s*\(',
            r'file\s*\('
        ]

        self.max_input_size = 1024 * 1024  # 1MB
        self.allowed_schemes = {'http', 'https', 'ftp'}

    def validate_tool_input(self, tool_name: str, input_data: Dict[str, Any]) -> List[str]:
        """Validate tool input for security issues"""
        violations = []

        # Check input size
        input_str = str(input_data)
        if len(input_str.encode('utf-8')) > self.max_input_size:
            violations.append("Input data too large")

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                violations.append(f"Dangerous pattern detected: {pattern}")

        # Validate URLs if present
        violations.extend(self._validate_urls(input_data))

        # Tool-specific validation
        violations.extend(self._validate_tool_specific(tool_name, input_data))

        return violations

    def _validate_urls(self, data: Any) -> List[str]:
        """Validate URLs in input data"""
        violations = []

        if isinstance(data, dict):
            for value in data.values():
                violations.extend(self._validate_urls(value))
        elif isinstance(data, list):
            for item in data:
                violations.extend(self._validate_urls(item))
        elif isinstance(data, str):
            # Simple URL detection
            if data.startswith(('http://', 'https://', 'ftp://')):
                scheme = data.split('://')[0]
                if scheme not in self.allowed_schemes:
                    violations.append(f"Disallowed URL scheme: {scheme}")

        return violations

    def _validate_tool_specific(self, tool_name: str, input_data: Dict[str, Any]) -> List[str]:
        """Tool-specific security validation"""
        violations = []

        if tool_name == "web_search":
            query = input_data.get("query", "")
            if len(query) > 500:
                violations.append("Search query too long")

        elif tool_name == "file_tool":
            file_path = input_data.get("file_path", "")
            if ".." in file_path or file_path.startswith("/"):
                violations.append("Potentially dangerous file path")

        return violations

class SecureToolExecutor(ToolExecutor):
    """Tool executor with security validation"""

    def __init__(self, tool_registry: ToolRegistry):
        super().__init__(tool_registry)
        self.security_validator = ToolSecurityValidator()

    async def execute_tool(
        self, tool_name: str, input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolOutput:
        """Execute tool with security validation"""
        # Security validation
        violations = self.security_validator.validate_tool_input(tool_name, input_data)

        if violations:
            return ToolOutput(
                success=False,
                error=f"Security violations: {', '.join(violations)}"
            )

        # Execute with base implementation
        return await super().execute_tool(tool_name, input_data, context)
```

### Tool Sandboxing
```python
import tempfile
import shutil
from pathlib import Path

class SandboxedToolExecutor(SecureToolExecutor):
    """Tool executor with sandboxing"""

    def __init__(self, tool_registry: ToolRegistry):
        super().__init__(tool_registry)
        self.sandbox_dir = None

    async def __aenter__(self):
        """Enter sandbox context"""
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix="tool_sandbox_"))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit sandbox context"""
        if self.sandbox_dir and self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir)

    async def execute_tool_sandboxed(
        self, tool_name: str, input_data: Dict[str, Any]
    ) -> ToolOutput:
        """Execute tool in sandboxed environment"""
        if not self.sandbox_dir:
            raise RuntimeError("Sandbox not initialized")

        # Add sandbox context
        context = {
            'sandbox_dir': str(self.sandbox_dir),
            'restricted_mode': True
        }

        return await self.execute_tool(tool_name, input_data, context)
```

## Performance Guidelines

### Async Tool Execution
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class PerformantToolExecutor(ToolExecutor):
    """Tool executor optimized for performance"""

    def __init__(self, tool_registry: ToolRegistry, max_workers: int = 4):
        super().__init__(tool_registry)
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.execution_cache: Dict[str, Any] = {}

    async def execute_tools_parallel(
        self, tool_requests: List[Dict[str, Any]]
    ) -> List[ToolOutput]:
        """Execute multiple tools in parallel"""
        tasks = []

        for request in tool_requests:
            task = self.execute_tool(
                request['tool_name'],
                request['input_data'],
                request.get('context')
            )
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def execute_tool_cached(
        self, tool_name: str, input_data: Dict[str, Any], ttl: int = 300
    ) -> ToolOutput:
        """Execute tool with result caching"""
        cache_key = self._generate_cache_key(tool_name, input_data)

        # Check cache
        if cache_key in self.execution_cache:
            cached_result, timestamp = self.execution_cache[cache_key]
            if time.time() - timestamp < ttl:
                return cached_result

        # Execute tool
        result = await self.execute_tool(tool_name, input_data)

        # Cache successful results
        if result.success:
            self.execution_cache[cache_key] = (result, time.time())

        return result

    def _generate_cache_key(self, tool_name: str, input_data: Dict[str, Any]) -> str:
        """Generate cache key for tool execution"""
        import hashlib
        import json

        data_str = json.dumps({
            'tool_name': tool_name,
            'input_data': input_data
        }, sort_keys=True)

        return hashlib.md5(data_str.encode()).hexdigest()
```

## Common Patterns

### Tool Composition
```python
class CompositeToolExecutor(ToolExecutor):
    """Execute multiple tools in sequence or parallel"""

    async def execute_tool_chain(
        self, tool_chain: List[Dict[str, Any]]
    ) -> List[ToolOutput]:
        """Execute tools in sequence, passing output to next tool"""
        results = []
        previous_output = None

        for tool_config in tool_chain:
            tool_name = tool_config['tool_name']
            input_data = tool_config['input_data'].copy()

            # Inject previous output if configured
            if previous_output and tool_config.get('use_previous_output'):
                input_data.update(previous_output.dict())

            result = await self.execute_tool(tool_name, input_data)
            results.append(result)

            if not result.success:
                break  # Stop chain on failure

            previous_output = result

        return results

    async def execute_tool_workflow(
        self, workflow_config: Dict[str, Any]
    ) -> Dict[str, ToolOutput]:
        """Execute complex tool workflow"""
        results = {}

        # Execute parallel tools first
        if 'parallel_tools' in workflow_config:
            parallel_results = await self.execute_tools_parallel(
                workflow_config['parallel_tools']
            )

            for i, result in enumerate(parallel_results):
                tool_name = workflow_config['parallel_tools'][i]['tool_name']
                results[f"parallel_{tool_name}"] = result

        # Execute sequential tools
        if 'sequential_tools' in workflow_config:
            sequential_results = await self.execute_tool_chain(
                workflow_config['sequential_tools']
            )

            for i, result in enumerate(sequential_results):
                tool_name = workflow_config['sequential_tools'][i]['tool_name']
                results[f"sequential_{tool_name}"] = result

        return results
```

### Dynamic Tool Creation
```python
class DynamicToolCreator:
    """Create tools dynamically from specifications"""

    def create_tool_from_spec(self, tool_spec: Dict[str, Any]) -> BaseTool:
        """Create tool from specification"""

        class DynamicTool(BaseTool):
            def __init__(self, spec):
                super().__init__(spec['name'], spec['description'])
                self.spec = spec

            async def execute(self, input_data):
                # Execute based on specification
                if self.spec['type'] == 'http_api':
                    return await self._execute_http_api(input_data)
                elif self.spec['type'] == 'script':
                    return await self._execute_script(input_data)
                else:
                    raise NotImplementedError(f"Tool type: {self.spec['type']}")

            def get_input_schema(self):
                return self.spec.get('input_schema', {})

            def get_output_schema(self):
                return self.spec.get('output_schema', {})

        return DynamicTool(tool_spec)
```

## Debugging Tips
- **Execution tracing** - Trace tool execution flow and performance
- **Input/output logging** - Log all tool inputs and outputs
- **Error aggregation** - Aggregate and analyze tool execution errors
- **Performance monitoring** - Monitor tool execution times and resource usage
- **Security auditing** - Audit tool security validation effectiveness
