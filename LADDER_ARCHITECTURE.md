# LADDER System Architecture

## Overview

This document provides the system architect's view of LADDER integration within the Cortex agent system, including component interactions, message flows, and deployment considerations.

## High-Level Architecture

```text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Layer    │    │   Knowledge      │    │   External      │
│                 │    │   Graph          │    │   Tools/APIs    │
│ • CLI/API       │    │                  │    │                 │
│ • Web UI        │    │ • Facts Storage  │    │ • Web Search    │
│ • Chat Interface│    │ • Memory         │    │ • Code Exec     │
└─────────────────┘    │ • Relationships  │    │ • File System   │
         │              └──────────────────┘    │ • Databases     │
         │                       │              └─────────────────┘
         ▼                       │                       │
┌─────────────────┐              │                       │
│  Orchestrator   │              │                       │
│                 │              │                       │
│ • Request       │              │                       │
│   Routing       │              │                       │
│ • Response      │              │                       │
│   Formatting    │              │                       │
│ • Error         │              │                       │
│   Handling      │              │                       │
└─────────────────┘              │                       │
         │                       │                       │
         ▼                       │                       │
┌─────────────────┐              │                       │
│ Ladder Adapter  │              │                       │
│                 │              │                       │
│ • Mode Control  │              │                       │
│ • Integration   │              │                       │
│ • Logging       │              │                       │
└─────────────────┘              │                       │
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────────────────────────────────────────────┐     │
│              LADDER Planner                             │     │
│                                                         │     │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐ │     │
│  │   Task        │  │  Decomposer  │  │   Bandit     │ │     │
│  │   Graph       │  │  Framework   │  │   Policy     │ │     │
│  │               │  │              │  │              │ │     │
│  │ • DAG Mgmt    │  │ • LLM-based  │  │ • UCB1       │ │     │
│  │ • Dependencies│  │ • Domain     │  │ • ε-greedy   │ │     │
│  │ • Scheduling  │  │   Specific   │  │ • Learning   │ │     │
│  └───────────────┘  └──────────────┘  └──────────────┘ │     │
│                                                         │     │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐ │     │
│  │   Energy      │  │  Execution   │  │   Tool       │ │     │
│  │   Manager     │  │  Controller  │  │   Selector   │ │◄────┘
│  │               │  │              │  │              │ │
│  │ • Priority    │  │ • Shadow/    │  │ • Selection  │ │
│  │ • Scheduling  │  │   Active     │  │ • Validation │ │
│  │ • Resources   │  │ • Safety     │  │ • Execution  │ │
│  └───────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Component Interaction Flow

### 1. Request Processing Flow

```text
User Request → Orchestrator → Ladder Adapter → LADDER Planner
                    ↓
            [Mode Check: Shadow/Active]
                    ↓
            Plan Creation Phase:
            • Task Decomposition
            • Dependency Analysis
            • Energy Assignment
                    ↓
            Execution Phase:
            • Task Scheduling
            • Tool Selection (Bandit)
            • Result Collection
                    ↓
            Response Assembly → User
```

### 2. Message Flow Sequence

```text
1. User → Orchestrator: "Build a web API with authentication"

2. Orchestrator → LadderAdapter: handle_request(query, mode)

3. LadderAdapter → LadderPlanner: create_plan(query)

4. LadderPlanner → Decomposer: decompose(root_task)
   Decomposer → LadderPlanner: [subtask1, subtask2, subtask3]

5. LadderPlanner → TaskGraph: build_dag(subtasks, dependencies)

6. For each ready task:
   LadderPlanner → BanditPolicy: select_tool(task, available_tools)
   BanditPolicy → LadderPlanner: selected_tool

7. LadderPlanner → ToolExecutor: execute(tool, task_input, mode)
   ToolExecutor → ExternalAPI: [if active mode]
   ToolExecutor → LadderPlanner: task_result

8. LadderPlanner → BanditPolicy: update_reward(tool, result_quality)

9. LadderPlanner → KnowledgeGraph: store_fact(task_result)

10. LadderPlanner → LadderAdapter: final_result
    LadderAdapter → Orchestrator: formatted_response
    Orchestrator → User: response
```

## Integration Points

### 1. Orchestrator Integration

The Cortex orchestrator integrates LADDER through several key extension points:

#### Configuration Initialization

```python
# In orchestrator startup
if settings.PLANNER == "ladder":
    ladder_planner = LadderPlanner(
        tool_registry=self.tool_registry,
        decomposers=self._load_decomposers(),
        mode=settings.LADDER_MODE,
        config=settings.LADDER_CONFIG
    )
    self.planner_adapter = LadderAdapter(ladder_planner)
```

#### Request Routing

```python
# In orchestrator main loop
async def handle_request(self, user_query: str) -> str:
    if self.planner_adapter and self._should_use_ladder(user_query):
        return await self.planner_adapter.handle_request(user_query)
    else:
        return await self._default_handler(user_query)
```

#### Tool Access Bridge

```python
# LadderAdapter provides tool access to LADDER
class LadderAdapter:
    def execute_tool(self, tool_name: str, input_data: Any, dry_run: bool) -> Any:
        tool = self.orchestrator.tool_registry.get_tool(tool_name)
        if dry_run:
            return tool.simulate(input_data)
        else:
            return tool.execute(input_data)
```

### 2. Shadow Mode Implementation

Shadow mode enables safe validation by running LADDER in parallel without real effects:

```python
class ExecutionController:
    def __init__(self, mode: str = "active"):
        self.mode = mode
        self.shadow_log = []

    def execute_task(self, task: Task, tool: str) -> Any:
        if self.mode == "shadow":
            # Log intended action without executing
            action = f"Would execute {tool} for: {task.description}"
            self.shadow_log.append(action)
            return self._simulate_result(task, tool)
        else:
            # Real execution
            return self._real_execute(task, tool)
```

### 3. Knowledge Graph Interface

LADDER interfaces with the knowledge graph for persistent memory:

```python
class KGInterface:
    def query_fact(self, query: str) -> Optional[Any]:
        """Query existing knowledge before task execution."""
        pass

    def store_result(self, task: Task, result: Any) -> None:
        """Store task results as facts."""
        pass

    def check_consistency(self, new_fact: Any) -> bool:
        """Validate new information against existing knowledge."""
        pass
```

## Environment Configuration

### Core Settings

```bash
# Planner Selection
CORTEX_PLANNER=ladder              # Enable LADDER
LADDER_MODE=shadow                 # shadow|active

# Bandit Configuration
LADDER_BANDIT_ALGORITHM=ucb1       # ucb1|epsilon_greedy|thompson
LADDER_BANDIT_EPSILON=0.1          # Exploration rate for ε-greedy
LADDER_BANDIT_CONFIDENCE=1.414     # UCB confidence multiplier

# Planning Constraints
LADDER_MAX_DEPTH=5                 # Maximum decomposition depth
LADDER_MAX_TASKS=50                # Maximum tasks per plan
LADDER_ENERGY_BUDGET=100.0         # Maximum energy per plan

# Knowledge Graph
LADDER_KG_ENABLED=true             # Enable KG integration
LADDER_KG_URL=neo4j://localhost    # KG connection string
LADDER_KG_WRITEBACK=true           # Allow storing new facts

# Debugging & Monitoring
LADDER_DEBUG=true                  # Verbose logging
LADDER_LOG_SHADOW=true             # Log shadow mode actions
LADDER_METRICS_ENABLED=true        # Enable performance tracking
LADDER_TIMEOUT=300                 # Execution timeout (seconds)
```

### Advanced Configuration

```python
# Configuration object
LADDER_CONFIG = {
    "decomposition": {
        "llm_model": "gpt-4",
        "max_subtasks": 10,
        "min_task_complexity": 0.3
    },
    "scheduling": {
        "priority_strategy": "energy_first",  # energy_first|deadline_first|hybrid
        "parallel_execution": False,
        "retry_failed_tasks": True
    },
    "learning": {
        "reward_discount": 0.95,
        "exploration_decay": 0.99,
        "save_stats_interval": 100
    }
}
```

## Deployment Considerations

### 1. Scaling Architecture

For production deployment, consider:

```text
┌─────────────────┐
│   Load Balancer │
└─────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌─────────┐
│Cortex-1 │ │Cortex-2 │  ← Multiple instances
└─────────┘ └─────────┘
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│   Shared KG     │  ← Centralized knowledge
│   & Bandit      │    and learning state
│   State         │
└─────────────────┘
```

### 2. State Persistence

```python
# Bandit state persistence
class PersistentBanditPolicy(BanditPolicy):
    def __init__(self, state_file: str = "bandit_state.json"):
        super().__init__()
        self.state_file = state_file
        self.load_state()

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.stats, f)

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                self.stats = json.load(f)
```

### 3. Monitoring & Observability

Key metrics to track:

- **Success Rate**: Task completion percentage
- **Plan Efficiency**: Average tasks per successful completion
- **Learning Progress**: Bandit convergence metrics
- **Latency**: Planning and execution time
- **Resource Usage**: Memory and compute utilization

```python
# Metrics collection
class LADDERMetrics:
    def __init__(self):
        self.task_counter = Counter()
        self.success_rate = MovingAverage(window=100)
        self.latency_tracker = LatencyTracker()

    def record_task_completion(self, task: Task, success: bool, duration: float):
        self.task_counter[task.type] += 1
        self.success_rate.update(1.0 if success else 0.0)
        self.latency_tracker.record(duration)
```

### 4. Security Considerations

```python
# Tool access controls
class SecureToolExecutor:
    def __init__(self, allowed_tools: Set[str], user_permissions: Dict[str, Set[str]]):
        self.allowed_tools = allowed_tools
        self.user_permissions = user_permissions

    def execute_tool(self, tool_name: str, user_id: str, input_data: Any) -> Any:
        if tool_name not in self.allowed_tools:
            raise SecurityError(f"Tool {tool_name} not allowed")

        if tool_name not in self.user_permissions.get(user_id, set()):
            raise SecurityError(f"User {user_id} cannot access {tool_name}")

        # Additional input validation
        if not self._validate_input(tool_name, input_data):
            raise SecurityError("Invalid input data")

        return self._execute(tool_name, input_data)
```

## Performance Optimization

### 1. Caching Strategy

```python
# Task result caching
class TaskCache:
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl

    def get_cached_result(self, task_hash: str) -> Optional[Any]:
        if task_hash in self.cache:
            result, timestamp = self.cache[task_hash]
            if time.time() - timestamp < self.ttl:
                return result
        return None

    def cache_result(self, task_hash: str, result: Any):
        self.cache[task_hash] = (result, time.time())
```

### 2. Parallel Execution

```python
# Concurrent task execution
async def execute_ready_tasks(self, ready_tasks: List[Task]) -> Dict[str, Any]:
    """Execute independent tasks concurrently."""
    if not self.config.get("parallel_execution", False):
        return await self._sequential_execution(ready_tasks)

    # Group tasks by resource requirements
    task_groups = self._group_by_resources(ready_tasks)
    results = {}

    for group in task_groups:
        group_results = await asyncio.gather(
            *[self._execute_single_task(task) for task in group],
            return_exceptions=True
        )
        results.update(dict(zip([t.id for t in group], group_results)))

    return results
```

## Integration Testing

### 1. Shadow Mode Validation

```python
async def test_shadow_mode_comparison():
    """Compare LADDER shadow mode with baseline system."""
    test_queries = load_test_queries()

    for query in test_queries:
        # Run baseline
        baseline_result = await baseline_system.process(query)

        # Run LADDER in shadow
        ladder_result = await ladder_system.process(query, mode="shadow")

        # Compare results
        comparison = compare_results(baseline_result, ladder_result)
        assert comparison.accuracy >= 0.8
        assert comparison.safety_score == 1.0
```

### 2. End-to-End Testing

```python
async def test_complex_workflow():
    """Test multi-step task execution."""
    task = "Create a REST API with database integration and tests"

    # Execute with LADDER
    result = await ladder_planner.execute(task)

    # Verify plan structure
    assert len(result.plan.tasks) >= 4  # Decomposed appropriately
    assert result.plan.has_dependencies()  # Proper ordering

    # Verify execution
    assert result.success
    assert all(task.status == TaskStatus.COMPLETED for task in result.plan.tasks)
```

## Future Enhancements

1. **Dynamic Re-planning**: Ability to modify plans during execution
2. **Multi-Agent Coordination**: LADDER instances working together
3. **Advanced Learning**: Deep RL for decomposition strategies
4. **Real-time Adaptation**: Online learning from user feedback

See [LADDER_DEVELOPER_GUIDE.md](LADDER_DEVELOPER_GUIDE.md) for implementation details.
