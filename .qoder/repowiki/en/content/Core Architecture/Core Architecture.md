
# Core Architecture

<cite>
**Referenced Files in This Document**   
- [main.py](file://src/main.py)
- [events.py](file://src/events.py)
- [agent_orchestrator.py](file://backend/agent_orchestrator.py)
- [mcp_server.py](file://backend/mcp_server.py)
- [constitutional_gateway.py](file://src/constitutional_gateway.py)
- [services.yaml](file://config/services.yaml)
- [docker-compose.yml](file://docker/docker-compose.yml)
- [server.py](file://mcp_server/src/mcp_server/server.py)
- [planner_ability.py](file://src/abilities/planner_ability.py)
- [redis_event_bus.py](file://src/adapters/redis_event_bus.py)
- [telemetry_pipeline.yaml](file://config/telemetry_pipeline.yaml)
- [security_policies.yaml](file://config/security_policies.yaml)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)

## Introduction
The Super Alita framework is a sophisticated AI orchestration system designed for autonomous agent development and execution. This document details the core architectural principles, component interactions, and system design of the framework. The architecture emphasizes modularity, event-driven communication, and specification-driven development to enable robust, extensible, and reliable multi-agent systems.

## Project Structure

The Super Alita framework follows a modular monorepo structure with clearly defined component boundaries and responsibilities. The organization supports both independent development and seamless integration of various subsystems.

```mermaid
graph TB
subgraph "Core Framework"
main[main.py]
events[events.py]
pipeline[pipeline.py]
end
subgraph "Backend Services"
agent_orchestrator[agent_orchestrator.py]
mcp_server[mcp_server.py]
context_server[context_server.py]
end
subgraph "Configuration"
config[config/]
services_yaml[services.yaml]
startup_yaml[startup.yaml]
end
subgraph "Docker Infrastructure"
docker[Docker/]
compose[docker-compose.yml]
redis[Docker Redis]
end
subgraph "MCP Server"
mcp[MCP Server]
mcp_server_py[server.py]
tools[tools/]
end
main --> agent_orchestrator
main --> mcp_server
agent_orchestrator --> context_server
config --> main
config --> agent_orchestrator
compose --> redis
compose --> agent_orchestrator
mcp_server_py --> tools
main --> mcp_server_py
style main fill:#4CAF50,stroke:#388E3C
style agent_orchestrator fill:#2196F3,stroke:#1976D2
style mcp_server_py fill:#9C27B0,stroke:#7B1FA2
style config fill:#FF9800,stroke:#F57C00
style compose fill:#607D8B,stroke:#455A64
```

**Diagram sources**
- [main.py](file://src/main.py)
- [agent_orchestrator.py](file://backend/agent_orchestrator.py)
- [mcp_server.py](file://backend/mcp_server.py)
- [services.yaml](file://config/services.yaml)
- [docker-compose.yml](file://docker/docker-compose.yml)

**Section sources**
- [main.py](file://src/main.py)
- [backend](file://backend)
- [config](file://config)
- [docker](file://docker)
- [mcp_server](file://mcp_server)

## Core Components

The Super Alita framework consists of several core components that work together to enable autonomous agent functionality. The system is built around a modular architecture that supports plugin-based extensibility and event-driven communication.

The main entry point is controlled by `main.py`, which initializes the core system components and orchestrates the startup sequence. The event system, defined in `events.py`, provides the foundation for asynchronous communication between components. The backend services, including the agent orchestrator and MCP server, handle specialized functionality for agent management and tool execution.

The configuration system, centered around YAML files in the config directory, enables declarative setup of services, security policies, and telemetry pipelines. The Docker infrastructure provides containerized deployment options with Redis for message brokering.

**Section sources**
- [main.py](file://src/main.py#L1-L50)
- [events.py](file://src/events.py#L1-L30)
- [services.yaml](file://config/services.yaml#L1-L20)
- [docker-compose.yml](file://docker/docker-compose.yml#L1-L15)

## Architecture Overview

The Super Alita framework employs an event-driven, microservices-inspired architecture with a plugin-based extensibility model. The system is designed to support specification-driven development, constitutional governance, and multi-agent orchestration patterns.

```mermaid
graph TD
Client[External Client] --> API[API Interface]
subgraph "Core System"
API --> EventBus[Event Bus<br/>Redis]
EventBus --> Orchestrator[Agent Orchestrator]
EventBus --> MCP[MCP Server]
EventBus --> Context[Context Server]
Orchestrator --> Planner[Planner Ability]
Orchestrator --> Consensus[Consensus System]
Orchestrator --> Constitutional[Constitutional Gateway]
MCP --> Tools[Tool Plugins]
MCP --> Deepcode[Deepcode Integration]
MCP --> Puter[Puter Integration]
Context --> Knowledge[Knowledge Graph]
Context --> Memory[Memory System]
Constitutional --> Policies[Constitutional Articles]
Constitutional --> Scorer[Constitutional Scorer]
end
subgraph "Infrastructure"
Redis[(Redis)]
Telemetry[Telemetry Pipeline]
Security[Security Policies]
end
Orchestrator --> Telemetry
MCP --> Telemetry
Context --> Telemetry
Constitutional --> Telemetry
Security --> Orchestrator
Security --> MCP
Security --> Context
Redis -.-> EventBus
style EventBus fill:#FFC107,stroke:#FFA000
style Orchestrator fill:#2196F3,stroke:#1976D2
style MCP fill:#9C27B0,stroke:#7B1FA2
style Context fill:#00BCD4,stroke:#0097A7
style Constitutional fill:#E91E63,stroke:#C2185B
style Redis fill:#F44336,stroke:#D32F2F
style Telemetry fill:#4CAF50,stroke:#388E3C
style Security fill:#607D8B,stroke:#455A64
```

**Diagram sources**
- [agent_orchestrator.py](file://backend/agent_orchestrator.py)
- [mcp_server.py](file://backend/mcp_server.py)
- [context_server.py](file://backend/context_server.py)
- [constitutional_gateway.py](file://src/constitutional_gateway.py)
- [services.yaml](file://config/services.yaml)
- [telemetry_pipeline.yaml](file://config/telemetry_pipeline.yaml)
- [security_policies.yaml](file://config/security_policies.yaml)

## Detailed Component Analysis

### Event-Driven Architecture

The Super Alita framework is built on an event-driven architecture that enables loose coupling between components and supports asynchronous processing patterns. The event bus serves as the central nervous system of the framework, facilitating communication between agents, services, and external systems.

```mermaid
sequenceDiagram
participant Client as "External Client"
participant Main as "Main Application"
participant EventBus as "Event Bus (Redis)"
participant Orchestrator as "Agent Orchestrator"
participant MCP as "MCP Server"
participant Context as "Context Server"
Client->>Main : Initiate Agent Task
Main->>EventBus : Publish TASK_INITIATED event
EventBus->>Orchestrator : Deliver TASK_INITIATED
Orchestrator->>Orchestrator : Process task planning
Orchestrator->>EventBus : Publish TASK_PLANNED event
EventBus->>MCP : Deliver TASK_PLANNED
MCP->>MCP : Execute tool operations
MCP->>EventBus : Publish TOOL_EXECUTION_COMPLETE event
EventBus->>Context : Deliver TOOL_EXECUTION_COMPLETE
Context->>Context : Update context state
Context->>EventBus : Publish CONTEXT_UPDATED event
EventBus->>Orchestrator : Deliver CONTEXT_UPDATED
Orchestrator->>Orchestrator : Evaluate next steps
Orchestrator->>Main : Return task result
Main->>Client : Respond with result
```

**Diagram sources**
- [events.py](file://src/events.py)
- [agent_orchestrator.py](file://backend/agent_orchestrator.py)
- [mcp_server.py](file://backend/mcp_server.py)
- [context_server.py](file://backend/context_server.py)

### Plugin-Based Extensibility

The framework supports plugin-based extensibility through the MCP (Model Context Protocol) server architecture. This design allows for dynamic addition of capabilities without modifying the core system.

```mermaid
classDiagram
class MCP_Server {
+start_server()
+register_tool(tool)
+handle_request()
-validate_request()
-execute_tool()
-format_response()
}
class Tool_Plugin {
<<interface>>
+execute(params)
+get_metadata()
+validate_params()
}
class Deepcode_Tool {
+execute(params)
+get_metadata()
+validate_params()
}
class Puter_Tool {
+execute(params)
+get_metadata()
+validate_params()
}
class Format_and_Scan_Tool {
+execute(params)
+get_metadata()
+validate_params()
}
MCP_Server --> Tool_Plugin : "uses"
Tool_Plugin <|-- Deepcode_Tool
Tool_Plugin <|-- Puter_Tool
Tool_Plugin <|-- Format_and_Scan_Tool
MCP_Server --> Deepcode_Tool : "registers"
MCP_Server --> Puter_Tool : "registers"
MCP_Server --> Format_and_Scan_Tool : "registers"
```

**Diagram sources**
- [server.py](file://mcp_server/src/mcp_server/server.py)
- [deepcode_tool.py](file://mcp_server/src/mcp_server/tools/deepcode_tool.py)
- [puter_tool.py](file://mcp_server/src/mcp_server/tools/puter_tool.py)
- [format_and_scan.py](file://mcp_server/src/mcp_server/tools/format_and_scan.py)

### Specification-Driven Development

The framework implements specification-driven development patterns, where system behavior is defined through declarative specifications rather than imperative code. This approach enables greater consistency, testability, and maintainability.

```mermaid
flowchart TD
Spec[Specification File] --> Parser[Specification Parser]
Parser --> Validator[Specification Validator]
Validator --> Valid{"Valid?"}
Valid --> |No| Error[Return Validation Errors]
Valid --> |Yes| Generator[Code Generator]
Generator --> Implementation[Implementation Code]
Implementation --> Tester[Automated Tests]
Tester --> Coverage{"Coverage Complete?"}
Coverage --> |No| Revise[Revise Implementation]
Coverage --> |Yes| Deploy[Deploy to Environment]
Deploy --> Monitor[Runtime Monitoring]
Monitor --> Feedback[Feedback Loop]
Feedback --> Spec
Revise --> Implementation
style Spec fill:#FFD54F,stroke:#FBC02D
style Parser fill:#4FC3F7,stroke:#29B6F6
style Validator fill:#81C784,stroke:#66BB6A
style Generator fill:#BA68C8,stroke:#AB47BC
style Implementation fill:#FF8A65,stroke:#FF7043
style Tester fill:#90A4AE,stroke:#78909C
style Deploy fill:#26A69A,stroke:#00897B
style Monitor fill:#7986CB,stroke:#5C6BC0
```

**Diagram sources**
- [planner_ability.py](file://src/abilities/planner_ability.py)
- [pipeline.py](file://src/pipeline.py)
- [test_sdd_integration.py](file://tests/test_sdd_integration.py)

### Constitutional Governance

The constitutional governance system provides a framework for ensuring that agent behavior adheres to predefined principles and constraints. This system acts as a governance layer that monitors and validates agent actions against constitutional articles.

```mermaid
sequenceDiagram
participant Agent as "Autonomous Agent"
participant Gateway as "Constitutional Gateway"
participant Scorer as "Constitutional Scorer"
participant Articles as "Constitutional Articles"
participant Policy as "Security Policy"
Agent->>Gateway : Request Action Execution
Gateway->>Scorer : Evaluate Action Against Constitution
Scorer->>Articles : Retrieve Relevant Articles
Articles-->>Scorer : Return Article Content
Scorer->>Scorer : Calculate Compliance Score
Scorer-->>Gateway : Return Compliance Assessment
Gateway->>Policy : Check Security Policies
Policy-->>Gateway : Return Policy Check Result
Gateway->>Agent : Allow or Reject Action
Gateway->>Scorer : Log Constitutional Compliance
Scorer->>Scorer : Update Compliance Metrics
```

**Diagram sources**
- [constitutional_gateway.py](file://src/constitutional_gateway.py)
- [scorer.py](file://src/constitutional/scorer.py)
- [articles.py](file://src/constitutional/articles.py)
- [security_policies.yaml](file://config/security_policies.yaml)

## Dependency Analysis

The Super Alita framework has a well-defined dependency structure that supports modularity while maintaining necessary integration points between components.

```mermaid
graph TD
main[main.py] --> agent_orchestrator[agent_orchestrator.py]
main --> mcp_server[mcp_server.py]
main --> events[events.py]
main --> pipeline[pipeline.py]
agent_orchestrator --> planner[planner_ability.py]
agent_orchestrator --> consensus[consensus_grpc/server.py]
agent_orchestrator --> constitutional[constitutional_gateway.py]
agent_orchestrator --> context[context_server.py]
mcp_server --> tools[tools/]
mcp_server --> deepcode[deepcode/]
mcp_server --> puter[puter/]
events --> redis[redis_event_bus.py]
events --> adapter[event_bus_adapter.py]
pipeline --> sdd[sdd/]
pipeline --> telemetry[telemetry/]
config --> services[services.yaml]
config --> security[security_policies.yaml]
config --> telemetry_config[telemetry_pipeline.yaml]
docker --> compose[docker-compose.yml]
docker --> redis_config[docker-compose.redis.yml]
style main fill:#4CAF50,stroke:#388E3C
style agent_orchestrator fill:#2196F3,stroke:#1976D2
style mcp_server fill:#9C27B0,stroke:#7B1FA2
style events fill:#FF9800,stroke:#F57C00
style pipeline fill:#607D8B,stroke:#455A64
style config fill:#795548,stroke:#5D4037
style docker fill:#9E9E9E,stroke:#616161
```

**Diagram sources**
- [main.py](file://src/main.py)
- [agent_orchestrator.py](file://backend/agent_orchestrator.py)
- [mcp_server.py](file://backend/mcp_server.py)
- [events.py](file://src/events.py)
- [pipeline.py](file://src/pipeline.py)
- [services.yaml](file://config/services.yaml)
- [docker-compose.yml](file://docker/docker-compose.yml)

**Section sources**
- [main.py](file://src/main.py)
- [backend](file://backend)
- [src](file://src)
- [config](file://config)
- [docker](file://docker)
- [mcp_server](file://mcp_server)

## Performance Considerations

The Super Alita framework incorporates several performance optimization strategies to ensure efficient operation at scale. The event-driven architecture enables asynchronous processing, reducing blocking operations and improving throughput. Redis serves as a high-performance message broker for the