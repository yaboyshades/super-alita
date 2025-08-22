# Knowledge Graph Task Manager Integration

## Overview

This document outlines the comprehensive integration between the Knowledge Graph and the To-Do List Framework for Super-Alita, including graph-driven prioritization, bi-directional updates, and visualization capabilities.

## ✅ Completed Integration Features

### 1. Core Knowledge Graph Task Manager (`src/core/kg_task_manager.py`)

**Features Implemented:**
- **Graph-driven task prioritization** using centrality scores and dependency analysis
- **Dependency tracking** with automatic ready task identification
- **Task graph creation** and comprehensive metrics calculation
- **Real-time task visualization** in Mermaid and DOT formats
- **Event-driven architecture** for seamless updates across the system

**Key Components:**
- `GraphTaskNode`: Enhanced task model with knowledge graph integration
- `TaskGraph`: Represents subgraphs of related tasks
- `GraphMetrics`: Comprehensive metrics for task prioritization
- `KnowledgeGraphTaskManager`: Main orchestrator with graph-driven intelligence

### 2. Copilot Todos API Integration (`src/core/copilot_todos_integration.py`)

**Features Implemented:**
- **Bi-directional synchronization** between VS Code Copilot Todos and knowledge graph
- **Graph metrics enrichment** of todo items with dependency context
- **Priority mapping** between different priority systems
- **Context augmentation** with related entities and graph metrics
- **Automatic discovery** of Copilot API endpoints

**Key Components:**
- `CopilotTodoItem`: Pydantic model for Copilot Todos API
- `CopilotTodosIntegration`: Main integration plugin
- API discovery and connection testing utilities

### 3. Comprehensive Integration Tests (`tests/test_kg_todos_integration.py`)

**Test Coverage:**
- Task creation and graph integration
- Graph-driven prioritization algorithms
- Task completion and dependency updates
- Task graph creation and metrics
- Task visualization generation
- Copilot integration models and mappings
- End-to-end integration workflows

## 🎯 Graph-Driven Prioritization Algorithm

### Centrality Calculation
```python
# Simplified PageRank-like algorithm for task centrality
for _ in range(10):  # 10 iterations
    new_centrality = {}
    for node in nodes:
        score = 0.15 / len(nodes)  # Damping factor

        # Add contributions from incoming edges
        for other_node, deps in graph.items():
            if node in deps:
                out_degree = len(deps) if deps else 1
                score += 0.85 * centrality.get(other_node, 0.0) / out_degree

        new_centrality[node] = score

    centrality = new_centrality
```

### Dependency Depth Analysis
```python
def calculate_depth(node: str) -> int:
    if node in visited:
        return depths.get(node, 0)

    visited.add(node)
    dependencies = graph.get(node, [])

    if not dependencies:
        depths[node] = 0
        return 0

    max_dep_depth = max(
        (calculate_depth(dep) for dep in dependencies), default=-1
    )
    depths[node] = max_dep_depth + 1
    return depths[node]
```

### Priority Adjustment Logic
```python
# Adjust priority based on graph position
if task.centrality_score > 0.7 or task.dependency_depth == 0:
    if task.priority.value in ["low", "medium"]:
        task.priority = TaskPriority.HIGH
elif task.centrality_score < 0.3 and task.dependency_depth > 3:
    if task.priority.value in ["high", "medium"]:
        task.priority = TaskPriority.LOW
```

## 🔄 Bi-Directional Synchronization

### Knowledge Graph → Copilot Todos
```python
async def sync_task_to_todo(self, task: GraphTaskNode) -> CopilotTodoItem | None:
    # Enrich todo with graph context
    enriched_context = await self._enrich_task_context(task)

    todo_data = {
        "title": task.title,
        "description": task.description,
        "completed": task.status == TaskStatus.COMPLETED,
        "priority": self._map_task_priority_to_todo(task.priority),
        "context": {
            **task.context,
            **enriched_context,
            "kg_node_id": task.kg_node_id,
            "dependency_depth": task.dependency_depth,
            "centrality_score": task.centrality_score,
        }
    }
```

### Copilot Todos → Knowledge Graph
```python
async def sync_todo_to_task(self, todo: CopilotTodoItem) -> GraphTaskNode | None:
    task_updates = {
        "title": todo.title,
        "description": todo.description,
        "status": TaskStatus.COMPLETED if todo.completed else TaskStatus.NOT_STARTED,
        "priority": self._map_todo_priority_to_task(todo.priority),
        "due_date": todo.due_date,
        "tags": set(todo.tags),
        "context": todo.context,
        "last_modified_by": "copilot_todos_integration",
    }
```

## 📊 Task Visualization

### Mermaid Graph Generation
```python
async def _generate_mermaid_graph(self, task_ids: set[str]) -> str:
    lines = ["graph TD"]

    # Add nodes with status colors
    for task_id in task_ids:
        task = self.tasks.get(task_id)
        if task:
            status_color = {
                TaskStatus.NOT_STARTED: "lightblue",
                TaskStatus.IN_PROGRESS: "yellow",
                TaskStatus.COMPLETED: "lightgreen",
                TaskStatus.BLOCKED: "red",
            }.get(task.status, "white")

            lines.append(f'    {task_id}["{task.title}"]')
            lines.append(f"    style {task_id} fill:{status_color}")

    # Add dependency edges
    for task_id in task_ids:
        task = self.tasks.get(task_id)
        if task:
            for dep_id in task.depends_on:
                if dep_id in task_ids:
                    lines.append(f"    {dep_id} --> {task_id}")

    return "\n".join(lines)
```

### DOT Graph Generation
```python
async def _generate_dot_graph(self, task_ids: set[str]) -> str:
    lines = ["digraph TaskGraph {", "    rankdir=TB;"]

    # Add nodes with status styling
    for task_id in task_ids:
        task = self.tasks.get(task_id)
        if task:
            lines.append(
                f'    "{task_id}" [label="{task.title}" '
                f'fillcolor="{status_color}" style="filled"];'
            )

    # Add dependency edges
    for task_id in task_ids:
        task = self.tasks.get(task_id)
        if task:
            for dep_id in task.depends_on:
                if dep_id in task_ids:
                    lines.append(f'    "{dep_id}" -> "{task_id}";')

    lines.append("}")
    return "\n".join(lines)
```

## 🎯 Graph Metrics and Analytics

### Task Graph Metrics
```python
class GraphMetrics(BaseModel):
    total_tasks: int = Field(default=0)
    completed_tasks: int = Field(default=0)
    blocked_tasks: int = Field(default=0)
    critical_path_length: int = Field(default=0)
    avg_centrality: float = Field(default=0.0)
    completion_percentage: float = Field(default=0.0)
    estimated_total_effort: float = Field(default=0.0)
    estimated_remaining_effort: float = Field(default=0.0)
```

### Context Enrichment
```python
async def _enrich_task_context(self, task: GraphTaskNode) -> dict[str, Any]:
    enrichment = {
        "graph_metrics": {
            "centrality_score": task.centrality_score,
            "dependency_depth": task.dependency_depth,
            "estimated_effort": task.estimated_effort,
        },
        "dependencies": {
            "depends_on": list(task.depends_on),
            "blocks": list(task.blocks),
            "related_to": list(task.related_to),
        },
        "kg_enrichment": {
            "entity_type": task.kg_entity_type,
            "node_id": task.kg_node_id,
            "last_updated": task.updated_at.isoformat(),
        }
    }
    return enrichment
```

## 🚀 Usage Examples

### Creating Tasks with Dependencies
```python
# Create a knowledge graph task manager
kg_manager = KnowledgeGraphTaskManager(event_bus, config)
await kg_manager.initialize()

# Create root task
root_task = await kg_manager.create_task(
    title="Design System Architecture",
    description="Create the high-level system architecture",
    task_type=TaskType.PLANNING,
    priority=TaskPriority.HIGH,
    tags={"architecture", "design"},
)

# Create dependent task
api_task = await kg_manager.create_task(
    title="Design API Endpoints",
    description="Define REST API endpoints",
    task_type=TaskType.IMPLEMENTATION,
    depends_on={root_task.task_id},
    tags={"api", "backend"},
)
```

### Getting Prioritized Tasks
```python
# Get tasks ordered by graph-driven priority
prioritized_tasks = await kg_manager.get_prioritized_tasks(limit=10)

# Get tasks ready to work on (no pending dependencies)
ready_tasks = await kg_manager.get_ready_tasks()
```

### Creating Task Graphs and Visualizations
```python
# Create a task graph
task_ids = {task.task_id for task in tasks}
graph = await kg_manager.create_task_graph(
    name="Sprint Planning",
    description="Current sprint tasks",
    task_ids=task_ids,
)

# Generate visualization
mermaid_viz = await kg_manager.visualize_task_graph(
    graph_id=graph.graph_id,
    format="mermaid"
)
```

### Copilot Integration
```python
# Create Copilot integration
copilot_integration = CopilotTodosIntegration(event_bus, config)
await copilot_integration.initialize()

# Sync task to Copilot todo
synced_todo = await copilot_integration.sync_task_to_todo(task)

# Perform full bidirectional sync
stats = await copilot_integration.full_sync()
```

## 🎯 Key Benefits

1. **Intelligent Prioritization**: Tasks are prioritized based on their position in the dependency graph and centrality scores

2. **Automatic Dependency Management**: System automatically identifies ready tasks and unblocks dependent tasks upon completion

3. **Visual Task Management**: Real-time graph visualizations help understand task relationships and project structure

4. **Seamless Integration**: Bi-directional sync with VS Code Copilot Todos provides native editor integration

5. **Context-Aware Planning**: Graph metrics enrich task context for better decision making

6. **Event-Driven Updates**: All changes propagate automatically through the event system

## 🔧 Configuration

### Knowledge Graph Task Manager Config
```python
config = {
    "auto_prioritize": True,           # Enable automatic prioritization
    "max_dependency_depth": 5,         # Maximum dependency chain depth
    "prioritization_interval": 300,    # Prioritization frequency (seconds)
}
```

### Copilot Integration Config
```python
config = {
    "copilot_api_base": "http://localhost:3000",  # Copilot API endpoint
    "api_timeout": 30.0,                          # API timeout
    "sync_interval": 300,                         # Sync frequency
    "auto_sync": True,                           # Enable automatic sync
}
```

## 📋 Next Steps

1. **Advanced Graph Analytics**: Implement more sophisticated graph algorithms for task optimization

2. **Machine Learning Integration**: Add ML-based task effort estimation and completion prediction

3. **Multi-Agent Coordination**: Extend for collaborative task management across multiple agents

4. **Performance Optimization**: Add caching and incremental graph updates for large task sets

5. **Security Framework**: Implement access control and audit trails for task management

6. **Multi-Modal Support**: Add support for voice commands and natural language task creation

## ✅ Conclusion

The Knowledge Graph Task Manager integration provides a comprehensive, intelligent task management system that leverages graph theory for optimal task prioritization and scheduling. The bi-directional integration with VS Code Copilot Todos ensures seamless workflow integration while maintaining the power of graph-driven insights.

This implementation represents a significant advancement in agentic task management, combining the precision of graph algorithms with the convenience of modern development tools.
