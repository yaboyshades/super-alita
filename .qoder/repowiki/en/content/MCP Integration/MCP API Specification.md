# MCP API Specification

<cite>
**Referenced Files in This Document**   
- [fastmcp_server.py](file://mcp/fastmcp_server.py)
- [mcp_server.py](file://backend/mcp_server.py)
- [server.py](file://mcp_server/src/mcp_server/server.py)
- [deepcode_tool.py](file://mcp_server/src/mcp_server/tools/deepcode_tool.py)
- [puter_tool.py](file://mcp_server/src/mcp_server/tools/puter_tool.py)
- [format_and_scan.py](file://mcp_server/src/mcp_server/tools/format_and_scan.py)
- [mcp_integration.py](file://src/core/mcp_integration.py)
- [test_mcp_registration_canonical.py](file://tests/test_mcp_registration_canonical.py)
- [test_mcp_catalog.py](file://tests/test_mcp_catalog.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [API Endpoints](#api-endpoints)
3. [Tool Invocation API](#tool-invocation-api)
4. [Capability Discovery Endpoints](#capability-discovery-endpoints)
5. [Health Check Routes](#health-check-routes)
6. [JSON Schema Definitions](#json-schema-definitions)
7. [Security Considerations](#security-considerations)
8. [Client Implementation Guidelines](#client-implementation-guidelines)
9. [Performance Optimization Tips](#performance-optimization-tips)
10. [API Usage Examples](#api-usage-examples)

## Introduction
The Model Context Protocol (MCP) API provides a standardized interface for integrating AI models with external tools and services. This specification documents the HTTP methods, URL patterns, request/response schemas, and authentication mechanisms used in the MCP protocol. The API enables secure tool invocation, capability discovery, and health monitoring for AI agent systems.

**Section sources**
- [fastmcp_server.py](file://mcp/fastmcp_server.py#L1-L50)
- [mcp_server.py](file://backend/mcp_server.py#L1-L20)

## API Endpoints
The MCP API exposes several endpoints for tool registration, execution, and discovery. The base URL for all endpoints is determined by the server configuration and transport mode (stdio or SSE).

```mermaid
flowchart TD
A[Client Request] --> B{Endpoint Type}
B --> C[/tools/mcp/register]
B --> D[/tools/mcp/catalog]
B --> E[/tools/mcp/abstract]
B --> F[/health]
C --> G[Tool Registration]
D --> H[Tool Catalog]
E --> I[Abstract Tool Info]
F --> J[Health Status]
```

**Diagram sources**
- [fastmcp_server.py](file://mcp/fastmcp_server.py#L100-L220)
- [server.py](file://mcp_server/src/mcp_server/server.py#L50-L70)

**Section sources**
- [fastmcp_server.py](file://mcp/fastmcp_server.py#L100-L220)
- [mcp_server.py](file://backend/mcp_server.py#L30-L50)

## Tool Invocation API
The Tool Invocation API allows clients to execute registered tools through the MCP server. Tools are invoked via POST requests to specific endpoints with JSON payloads containing execution parameters.

### Request Format
All tool invocations follow the same request structure:
- **HTTP Method**: POST
- **Content-Type**: application/json
- **Authentication**: Bearer token in Authorization header
- **Body**: JSON object with tool-specific parameters

### Response Structure
Successful tool invocations return a 200 OK status with a JSON response containing:
- `success`: Boolean indicating execution status
- `result`: Tool-specific output data
- `error`: Error message if execution failed
- `execution_info`: Metadata about the execution

Error responses include appropriate HTTP status codes (400, 401, 404, 500) with JSON error details.

**Section sources**
- [fastmcp_server.py](file://mcp/fastmcp_server.py#L150-L220)
- [mcp_integration.py](file://src/core/mcp_integration.py#L100-L150)

## Capability Discovery Endpoints
The MCP API provides endpoints for discovering available tool capabilities and their schemas.

### GET /tools/mcp/catalog
Returns a comprehensive list of all registered tools with their full schemas.

**Response Schema**:
```json
[
  {
    "name": "string",
    "description": "string",
    "input_schema": "JSON Schema",
    "output_schema": "JSON Schema"
  }
]
```

### POST /tools/mcp/abstract
Returns minimal tool information for quick discovery.

**Response Schema**:
```json
{
  "tools": [
    {
      "name": "string",
      "description": "string"
    }
  ]
  }
```

### POST /tools/mcp/register
Registers a new tool with the MCP server.

**Request Schema**:
```json
{
  "tool_id": "string",
  "description": "string",
  "action": "string",
  "input_schema": "JSON Schema",
  "output_schema": "JSON Schema"
}
```

**Section sources**
- [test_mcp_catalog.py](file://tests/test_mcp_catalog.py#L105-L139)
- [test_mcp_registration_canonical.py](file://tests/test_mcp_registration_canonical.py#L27-L86)

## Health Check Routes
The MCP API includes health check endpoints to monitor server status and connectivity.

### GET /health
The primary health check endpoint that returns server status.

**Response**:
- **200 OK**: Server is operational
- **503 Service Unavailable**: Server is not ready

The health check verifies:
- Server process is running
- Required dependencies are available
- Authentication systems are functional

**Section sources**
- [mcp_integration.py](file://src/core/mcp_integration.py#L50-L70)
- [start_super_alita.py](file://start_super_alita.py#L150-L188)

## JSON Schema Definitions
The MCP API uses JSON Schema to define tool capabilities and execution requests.

### Tool Registration Schema
```json
{
  "type": "object",
  "required": ["tool_id", "action", "input_schema"],
  "properties": {
    "tool_id": {
      "type": "string",
      "description": "Unique identifier for the tool"
    },
    "description": {
      "type": "string",
      "description": "Human-readable description of the tool"
    },
    "action": {
      "type": "string",
      "description": "Action name that triggers the tool"
    },
    "input_schema": {
      "type": "object",
      "description": "JSON Schema defining expected input parameters"
    },
    "output_schema": {
      "type": "object",
      "description": "JSON Schema defining expected output structure"
    }
  }
}
```

### Tool Execution Schema
```json
{
  "type": "object",
  "additionalProperties": true,
  "description": "Parameters passed to the tool function"
}
```

**Section sources**
- [test_mcp_registration_canonical.py](file://tests/test_mcp_registration_canonical.py#L27-L86)
- [mcp_integration.py](file://src/core/mcp_integration.py#L80-L100)

## Security Considerations
The MCP API implements several security measures to protect against unauthorized access and malicious use.

### Authentication
- **Bearer Token Authentication**: Required for all endpoints when enabled
- **Allowlist Validation**: Tokens must be in the server's allowlist
- **Environment Control**: Authentication can be disabled for local development

### Input Validation
- **Schema Validation**: All inputs validated against JSON Schema
- **Path Boundary Checks**: File operations restricted to workspace directory
- **Command Whitelisting**: Only safe commands allowed for execution

### Rate Limiting
- **Request Throttling**: Configurable rate limits per client
- **Timeout Enforcement**: Execution timeouts prevent hanging operations
- **Resource Monitoring**: Memory and CPU usage tracked during execution

**Section sources**
- [fastmcp_server.py](file://mcp/fastmcp_server.py#L60-L90)
- [puter_tool.py](file://mcp_server/src/mcp_server/tools/puter_tool.py#L50-L100)

## Client Implementation Guidelines
When implementing MCP API clients, follow these guidelines for reliable integration.

### Connection Management
- Use persistent connections when possible
- Implement exponential backoff for retry logic
- Handle connection timeouts gracefully

### Error Handling
- Parse JSON error responses for detailed information
- Implement fallback mechanisms for critical operations
- Log errors with sufficient context for debugging

### Tool Discovery
- Cache tool catalog responses to reduce network requests
- Validate tool schemas before invocation
- Handle tool registration changes dynamically

**Section sources**
- [mcp_integration.py](file://src/core/mcp_integration.py#L40-L190)

## Performance Optimization Tips
Optimize MCP API performance with these recommendations.

### Caching Strategies
- Cache tool catalog responses with appropriate TTL
- Implement client-side caching for frequent tool invocations
- Use ETags for conditional requests when supported

### Batch Operations
- Combine multiple tool invocations when possible
- Use asynchronous processing for independent operations
- Implement connection pooling for high-frequency requests

### Resource Management
- Monitor memory usage during tool execution
- Limit concurrent requests to prevent server overload
- Optimize payload sizes for network efficiency

**Section sources**
- [fastmcp_server.py](file://mcp/fastmcp_server.py#L100-L220)
- [mcp_integration.py](file://src/core/mcp_integration.py#L100-L150)

## API Usage Examples
This section provides examples of API usage from Python and TypeScript clients.

### Python Client Example
```python
import requests
import json

# Register a new tool
tool_spec = {
    "tool_id": "calculator_v1",
    "description": "Simple calculator for basic math",
    "action": "calculate",
    "input_schema": {
        "type": "object",
        "required": ["operation", "a", "b"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {"type": "number"},
            "b": {"type": "number"}
        }
    },
    "output_schema": {
        "type": "object",
        "properties": {"result": {"type": "number"}}
    }
}

headers = {
    "Authorization": "Bearer your-api-token",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/tools/mcp/register",
    json=tool_spec,
    headers=headers
)

if response.status_code == 200:
    print("Tool registered successfully")
else:
    print(f"Registration failed: {response.text}")
```

### TypeScript Client Example
```typescript
interface ToolSpec {
  tool_id: string;
  description: string;
  action: string;
  input_schema: object;
  output_schema: object;
}

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

class MCPClient {
  private baseUrl: string;
  private token: string;

  constructor(baseUrl: string, token: string) {
    this.baseUrl = baseUrl;
    this.token = token;
  }

  async registerTool(toolSpec: ToolSpec): Promise<ApiResponse<void>> {
    try {
      const response = await fetch(`${this.baseUrl}/tools/mcp/register`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(toolSpec)
      });

      if (response.ok) {
        return { success: true };
      } else {
        const error = await response.text();
        return { success: false, error };
      }
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }

  async getToolCatalog(): Promise<ApiResponse<any[]>> {
    try {
      const response = await fetch(`${this.baseUrl}/tools/mcp/catalog`, {
        headers: {
          'Authorization': `Bearer ${this.token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        return { success: true, data };
      } else {
        const error = await response.text();
        return { success: false, error };
      }
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }
}

// Usage example
const client = new MCPClient('http://localhost:8000', 'your-api-token');

client.registerTool({
  tool_id: 'calculator_v1',
  description: 'Simple calculator for basic math',
  action: 'calculate',
  input_schema: {
    type: 'object',
    required: ['operation', 'a', 'b'],
    properties: {
      operation: {
        type: 'string',
        enum: ['add', 'subtract', 'multiply', 'divide']
      },
      a: { type: 'number' },
      b: { type: 'number' }
    }
  },
  output_schema: {
    type: 'object',
    properties: {
      result: { type: 'number' }
    }
  }
}).then(result => {
  if (result.success) {
    console.log('Tool registered successfully');
  } else {
    console.error('Registration failed:', result.error);
  }
});
```

**Section sources**
- [test_mcp_registration_canonical.py](file://tests/test_mcp_registration_canonical.py#L27-L86)
- [test_mcp_catalog.py](file://tests/test_mcp_catalog.py#L105-L139)