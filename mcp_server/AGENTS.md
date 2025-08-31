# MCP Server Development - Agent Instructions

## Overview
The `mcp_server/` directory contains a standalone Model Context Protocol server:
- **MCP Implementation** - Full MCP protocol support for VS Code integration
- **Tool Registry** - Dynamic tool discovery and execution
- **VS Code Integration** - Seamless editor integration via MCP
- **Custom Tools** - Extensible tool system for Super Alita capabilities

## Project Structure
```
mcp_server/
├── src/mcp_server/
│   ├── tools/           # Individual MCP tools
│   ├── server.py        # Main MCP server implementation
│   └── __init__.py
├── pyproject.toml       # Separate Python project
└── README.md
```

## MCP Protocol Overview

### MCP Architecture
```
VS Code ←→ MCP Client ←→ MCP Server ←→ Super Alita Tools
```

### Core MCP Concepts
- **Tools** - Discrete capabilities exposed to VS Code
- **Resources** - File or data sources the server can access
- **Prompts** - Reusable prompt templates
- **Sampling** - LLM interaction capabilities

## Development Environment

### Setup MCP Server
```bash
# Bootstrap MCP development environment
pwsh .\Setup-MCP.ps1 -Bootstrap

# Health check
pwsh .\Setup-MCP.ps1 -Doctor

# Add new tool
pwsh .\Setup-MCP.ps1 -AddTool YourToolName
```

### VS Code Integration
```json
// claude_desktop_config.json
{
  "mcpServers": {
    "myCustomPythonAgent": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "/path/to/super-alita/mcp_server"
    }
  }
}
```

## Tool Development

### MCP Tool Template
```python
# src/mcp_server/tools/your_tool.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class YourToolInput(BaseModel):
    """Input schema for your tool"""
    parameter1: str = Field(description="Description of parameter1")
    parameter2: Optional[int] = Field(default=10, description="Optional parameter")
    dry_run: bool = Field(default=True, description="Preview changes without executing")

class YourToolOutput(BaseModel):
    """Output schema for your tool"""
    success: bool
    result: str
    error: Optional[str] = None

async def your_tool(input_data: YourToolInput) -> YourToolOutput:
    """
    Your tool description.

    Args:
        input_data: Tool input parameters

    Returns:
        Tool execution result
    """
    try:
        if input_data.dry_run:
            # Return preview/plan without executing
            return YourToolOutput(
                success=True,
                result="Dry run: would perform operation X with parameter1=" + input_data.parameter1
            )

        # Actual tool implementation
        result = perform_operation(input_data.parameter1, input_data.parameter2)

        return YourToolOutput(
            success=True,
            result=f"Operation completed: {result}"
        )

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return YourToolOutput(
            success=False,
            result="",
            error=str(e)
        )

# Tool metadata for MCP registration
TOOL_METADATA = {
    "name": "your_tool",
    "description": "Brief description of your tool",
    "input_schema": YourToolInput.model_json_schema(),
    "output_schema": YourToolOutput.model_json_schema()
}
```

### Tool Registration
```python
# Tools are auto-discovered from the tools/ directory
# Each tool file should export:
# - Tool function (async)
# - TOOL_METADATA dict
# - Input/Output Pydantic models

# Example registration in server.py
from mcp_server.tools.your_tool import your_tool, TOOL_METADATA

# Register with MCP server
server.register_tool(
    name=TOOL_METADATA["name"],
    description=TOOL_METADATA["description"],
    input_schema=TOOL_METADATA["input_schema"],
    handler=your_tool
)
```

## Core MCP Tools

### File System Tools
```python
# Example: file_analyzer.py
async def analyze_file(input_data: FileAnalysisInput) -> FileAnalysisOutput:
    """Analyze file content and structure"""

    file_path = Path(input_data.file_path).resolve()

    # Validate workspace boundaries
    workspace_root = Path(os.getenv("workspaceFolder", ".")).resolve()
    if not is_within_workspace(file_path, workspace_root):
        raise ValueError("File path outside workspace boundaries")

    if input_data.dry_run:
        return FileAnalysisOutput(
            success=True,
            result=f"Would analyze file: {file_path}"
        )

    # Safe file analysis
    analysis = perform_file_analysis(file_path)

    return FileAnalysisOutput(
        success=True,
        result=analysis,
        metadata={"file_size": file_path.stat().st_size}
    )
```

### Code Generation Tools
```python
# Example: code_generator.py
async def generate_code(input_data: CodeGenerationInput) -> CodeGenerationOutput:
    """Generate code based on specifications"""

    if input_data.dry_run:
        # Return unified diff preview
        diff = generate_code_diff(input_data.specification)
        return CodeGenerationOutput(
            success=True,
            result=f"Preview of changes:\n{diff}",
            dry_run=True
        )

    # Generate and validate code
    generated_code = create_code_from_spec(input_data.specification)
    validation_result = validate_generated_code(generated_code)

    if not validation_result.is_valid:
        return CodeGenerationOutput(
            success=False,
            error=f"Generated code validation failed: {validation_result.errors}"
        )

    return CodeGenerationOutput(
        success=True,
        result=generated_code,
        metadata={"language": input_data.language, "lines": len(generated_code.split('\n'))}
    )
```

## Security Guidelines

### Workspace Boundaries
```python
import os
from pathlib import Path

def is_within_workspace(file_path: Path, workspace_root: Path) -> bool:
    """Ensure file path is within workspace boundaries"""
    try:
        file_path.resolve().relative_to(workspace_root.resolve())
        return True
    except ValueError:
        return False

def validate_file_access(file_path: str) -> Path:
    """Validate and resolve file path safely"""
    path = Path(file_path).resolve()
    workspace_root = Path(os.getenv("workspaceFolder", ".")).resolve()

    if not is_within_workspace(path, workspace_root):
        raise ValueError(f"File access denied: {file_path} is outside workspace")

    return path
```

### Input Validation
```python
def validate_tool_input(input_data: BaseModel) -> None:
    """Validate tool input parameters"""

    # Pydantic handles basic validation
    # Add custom validation for security

    if hasattr(input_data, 'file_path'):
        validate_file_access(input_data.file_path)

    if hasattr(input_data, 'command'):
        # Never allow arbitrary command execution
        if any(danger in input_data.command.lower() for danger in ['rm', 'del', 'format']):
            raise ValueError("Dangerous command detected")
```

### Safe Defaults
```python
# Always default to dry_run=True
class SafeToolInput(BaseModel):
    dry_run: bool = Field(default=True, description="Preview mode (safer)")

# Always validate file paths
def safe_file_operation(file_path: str, operation: str) -> Dict:
    """Safely perform file operations"""

    # Validate path
    validated_path = validate_file_access(file_path)

    # Check if file exists
    if not validated_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Perform operation with error handling
    try:
        result = perform_operation(validated_path, operation)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## Testing Guidelines

### MCP Tool Testing
```python
import pytest
from mcp_server.tools.your_tool import your_tool, YourToolInput

@pytest.mark.asyncio
async def test_tool_dry_run():
    """Test tool in dry run mode"""
    input_data = YourToolInput(
        parameter1="test_value",
        dry_run=True
    )

    result = await your_tool(input_data)

    assert result.success
    assert "Dry run" in result.result
    assert result.error is None

@pytest.mark.asyncio
async def test_tool_execution():
    """Test actual tool execution"""
    input_data = YourToolInput(
        parameter1="test_value",
        dry_run=False
    )

    result = await your_tool(input_data)

    assert result.success
    assert result.error is None

@pytest.mark.asyncio
async def test_tool_error_handling():
    """Test tool error handling"""
    input_data = YourToolInput(
        parameter1="invalid_value",
        dry_run=False
    )

    result = await your_tool(input_data)

    assert not result.success
    assert result.error is not None
```

### Integration Testing
```python
@pytest.mark.integration
async def test_mcp_server_tool_registration():
    """Test tool registration with MCP server"""
    from mcp_server.server import create_mcp_server

    server = create_mcp_server()

    # Verify tool is registered
    tools = server.list_tools()
    tool_names = [tool.name for tool in tools]

    assert "your_tool" in tool_names

@pytest.mark.integration
async def test_vs_code_integration():
    """Test VS Code integration end-to-end"""
    # This would test the full MCP protocol flow
    # Requires MCP test framework
    pass
```

## Performance Guidelines

### Async Operations
```python
import asyncio
import aiofiles

async def process_multiple_files(file_paths: List[str]) -> List[Dict]:
    """Process multiple files concurrently"""

    async def process_single_file(file_path: str) -> Dict:
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            return {"file": file_path, "size": len(content)}

    # Process files concurrently
    tasks = [process_single_file(path) for path in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append({"error": str(result)})
        else:
            processed_results.append(result)

    return processed_results
```

### Caching
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_file_analysis(file_path: str, file_hash: str) -> Dict:
    """Cache file analysis results"""
    # Use file hash to invalidate cache when file changes
    return perform_expensive_analysis(file_path)

async def analyze_file_with_cache(file_path: str) -> Dict:
    """Analyze file with caching"""

    # Calculate file hash for cache key
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    # Use cached result if available
    return cached_file_analysis(file_path, file_hash)
```

## Debugging & Monitoring

### Logging Configuration
```python
import logging
import sys

def setup_mcp_logging(debug: bool = False) -> None:
    """Configure logging for MCP server"""

    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mcp_server.log')
        ]
    )

    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
```

### Tool Metrics
```python
import time
from functools import wraps

def track_tool_metrics(func):
    """Decorator to track tool execution metrics"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        tool_name = func.__name__

        try:
            result = await func(*args, **kwargs)

            # Log success metrics
            duration = time.time() - start_time
            logger.info(f"Tool {tool_name} completed in {duration:.2f}s")

            return result

        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            logger.error(f"Tool {tool_name} failed after {duration:.2f}s: {e}")
            raise

    return wrapper

# Usage
@track_tool_metrics
async def your_tool(input_data: YourToolInput) -> YourToolOutput:
    # Tool implementation
    pass
```

## Common Patterns

### Unified Response Format
```python
class StandardToolOutput(BaseModel):
    """Standard output format for all tools"""
    success: bool
    result: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    dry_run: bool = False

def create_success_response(result: str, metadata: Dict = None, dry_run: bool = False) -> StandardToolOutput:
    """Create standardized success response"""
    return StandardToolOutput(
        success=True,
        result=result,
        metadata=metadata or {},
        dry_run=dry_run
    )

def create_error_response(error: str, metadata: Dict = None) -> StandardToolOutput:
    """Create standardized error response"""
    return StandardToolOutput(
        success=False,
        result="",
        error=error,
        metadata=metadata or {}
    )
```

### Progressive Disclosure
```python
class ProgressiveToolInput(BaseModel):
    """Input with progressive detail levels"""
    operation: str
    quick_mode: bool = Field(default=True, description="Fast operation with basic output")
    detailed: bool = Field(default=False, description="Include detailed analysis")
    expert_mode: bool = Field(default=False, description="Full expert-level output")

async def progressive_tool(input_data: ProgressiveToolInput) -> StandardToolOutput:
    """Tool with progressive detail levels"""

    if input_data.quick_mode:
        result = perform_quick_operation(input_data.operation)
    elif input_data.detailed:
        result = perform_detailed_operation(input_data.operation)
    elif input_data.expert_mode:
        result = perform_expert_operation(input_data.operation)
    else:
        result = perform_default_operation(input_data.operation)

    return create_success_response(result)
```

### Configuration Management
```python
import os
from typing import Optional

class MCPConfig:
    """MCP server configuration"""

    def __init__(self):
        self.workspace_folder = os.getenv("workspaceFolder", ".")
        self.max_file_size = int(os.getenv("MCP_MAX_FILE_SIZE", "10485760"))  # 10MB
        self.enable_file_operations = os.getenv("MCP_ENABLE_FILE_OPS", "true").lower() == "true"
        self.debug_mode = os.getenv("MCP_DEBUG", "false").lower() == "true"

    def validate_file_size(self, file_path: str) -> bool:
        """Check if file size is within limits"""
        try:
            file_size = Path(file_path).stat().st_size
            return file_size <= self.max_file_size
        except OSError:
            return False

# Global config instance
config = MCPConfig()
```

## Deployment Guidelines

### Development Deployment
```bash
# Local development
cd mcp_server
python -m mcp_server

# With debug logging
MCP_DEBUG=true python -m mcp_server
```

### Production Deployment
```bash
# Production configuration
export MCP_ENABLE_FILE_OPS=true
export MCP_MAX_FILE_SIZE=52428800  # 50MB
export MCP_DEBUG=false

# Run with process manager
systemctl start mcp-server
```

### VS Code Configuration
```json
{
  "mcpServers": {
    "super-alita": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "/path/to/super-alita/mcp_server",
      "env": {
        "MCP_DEBUG": "false",
        "MCP_MAX_FILE_SIZE": "52428800"
      }
    }
  }
}
```
