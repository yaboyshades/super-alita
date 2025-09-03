# LADDER Developer Implementation Guide

## Overview

This guide provides implementation details for the LADDER planner system, including core modules, code structure, and extensibility patterns.

## Core Architecture

### Class Hierarchy

```text
LadderPlanner (orchestrator)
├── TaskGraph (DAG management)
│   └── Task (individual task model)
├── BanditPolicy (tool selection learning)
├── LadderDecomposer (task breakdown)
│   ├── DefaultDecomposer
│   ├── CodeDecomposer
│   └── CustomDecomposer
└── LadderAdapter (orchestrator integration)
```

## Core Modules

### 1. Task Model (`src/ladder/models/task.py`)

```python
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Task:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = None  # Task IDs this depends on
    result: Any = None
    energy: float = 1.0  # Energy/cost metric
    tool_options: List[str] = None  # Available tools for this task
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tool_options is None:
            self.tool_options = []
        if self.metadata is None:
            self.metadata = {}
        if not self.id:
            self.id = str(uuid4())
```

### 2. TaskGraph (`src/ladder/models/task_graph.py`)

```python
from typing import List, Dict, Set, Optional
import networkx as nx
from .task import Task, TaskStatus

class TaskGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.tasks: Dict[str, Task] = {}
        self.root_task_id: Optional[str] = None

    def add_task(self, task: Task, parent_id: Optional[str] = None) -> None:
        """Add task to graph with optional parent dependency."""
        self.tasks[task.id] = task
        self.graph.add_node(task.id)

        if parent_id:
            self.add_dependency(parent_id, task.id)
        elif self.root_task_id is None:
            self.root_task_id = task.id

    def add_dependency(self, parent_id: str, child_id: str) -> None:
        """Add dependency: child depends on parent."""
        self.graph.add_edge(parent_id, child_id)
        if child_id in self.tasks:
            self.tasks[child_id].dependencies.append(parent_id)

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks ready for execution (dependencies satisfied)."""
        ready = []
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                if self._are_dependencies_satisfied(task_id):
                    ready.append(task)
        return ready

    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies are completed."""
        for dep_id in self.tasks[task_id].dependencies:
            if self.tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True

    def mark_completed(self, task_id: str, result: Any = None) -> None:
        """Mark task as completed and update result."""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.COMPLETED
            self.tasks[task_id].result = result

    def get_dependents(self, task_id: str) -> List[Task]:
        """Get tasks that depend on this task."""
        dependents = []
        for dependent_id in self.graph.successors(task_id):
            dependents.append(self.tasks[dependent_id])
        return dependents

    def has_pending_tasks(self) -> bool:
        """Check if any tasks are still pending."""
        return any(task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]
                  for task in self.tasks.values())
```

### 3. Multi-Armed Bandit (`src/ladder/learning/bandit.py`)

```python
import math
import random
from typing import Dict, List
from dataclasses import dataclass, field

@dataclass
class BanditStats:
    wins: int = 0
    plays: int = 0
    total_reward: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.plays if self.plays > 0 else 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.plays if self.plays > 0 else 0.0

class BanditPolicy:
    def __init__(self, algorithm: str = "ucb1", epsilon: float = 0.1, confidence: float = 1.414):
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.confidence = confidence
        self.stats: Dict[str, BanditStats] = {}
        self.total_plays = 0

    def select_tool(self, available_tools: List[str]) -> str:
        """Select best tool using bandit algorithm."""
        if not available_tools:
            raise ValueError("No tools available")

        # Initialize stats for new tools
        for tool in available_tools:
            if tool not in self.stats:
                self.stats[tool] = BanditStats()

        if self.algorithm == "epsilon_greedy":
            return self._epsilon_greedy_select(available_tools)
        elif self.algorithm == "ucb1":
            return self._ucb1_select(available_tools)
        else:
            return random.choice(available_tools)

    def _epsilon_greedy_select(self, tools: List[str]) -> str:
        """ε-greedy selection: exploit best tool or explore randomly."""
        if random.random() < self.epsilon:
            return random.choice(tools)
        else:
            return max(tools, key=lambda t: self.stats[t].avg_reward)

    def _ucb1_select(self, tools: List[str]) -> str:
        """UCB1 selection: balance exploitation and exploration."""
        best_tool = None
        best_ucb = -float('inf')

        for tool in tools:
            stat = self.stats[tool]
            if stat.plays == 0:
                ucb = float('inf')  # Try unplayed tools first
            else:
                exploitation = stat.avg_reward
                exploration = self.confidence * math.sqrt(
                    math.log(self.total_plays) / stat.plays
                )
                ucb = exploitation + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_tool = tool

        return best_tool

    def update_reward(self, tool: str, reward: float) -> None:
        """Update tool statistics with reward feedback."""
        if tool not in self.stats:
            self.stats[tool] = BanditStats()

        stat = self.stats[tool]
        stat.plays += 1
        stat.total_reward += reward
        if reward > 0:
            stat.wins += 1

        self.total_plays += 1

    def get_tool_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current statistics for all tools."""
        return {
            tool: {
                "win_rate": stat.win_rate,
                "avg_reward": stat.avg_reward,
                "plays": stat.plays
            }
            for tool, stat in self.stats.items()
        }
```

### 4. Decomposer Framework (`src/ladder/decomposition/decomposer.py`)

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from ..models.task import Task

class LadderDecomposer(ABC):
    """Base class for task decomposers."""

    @abstractmethod
    def can_decompose(self, task: Task) -> bool:
        """Check if this decomposer can handle the task."""
        pass

    @abstractmethod
    def decompose(self, task: Task) -> List[Task]:
        """Break down task into subtasks."""
        pass

class DefaultDecomposer(LadderDecomposer):
    """Default LLM-based decomposer for general tasks."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def can_decompose(self, task: Task) -> bool:
        """Can handle any task as fallback."""
        return True

    def decompose(self, task: Task) -> List[Task]:
        """Use LLM to break down task into steps."""
        if not self.llm_client:
            # Fallback: simple heuristic decomposition
            return self._heuristic_decompose(task)

        prompt = f"""
        Break down this task into smaller, actionable subtasks:
        Task: {task.description}

        Return a list of specific subtasks that can be executed independently.
        """

        # Call LLM and parse response into subtasks
        response = self.llm_client.complete(prompt)
        return self._parse_llm_response(response, task)

    def _heuristic_decompose(self, task: Task) -> List[Task]:
        """Simple fallback decomposition."""
        # Basic keyword-based decomposition
        if any(word in task.description.lower() for word in ['build', 'create', 'develop']):
            return [
                Task(description=f"Plan approach for: {task.description}"),
                Task(description=f"Implement: {task.description}"),
                Task(description=f"Test and validate: {task.description}")
            ]
        else:
            # Single-step task
            return [task]

    def _parse_llm_response(self, response: str, parent_task: Task) -> List[Task]:
        """Parse LLM response into Task objects."""
        subtasks = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove numbering and bullet points
                clean_line = line.lstrip('0123456789.-• ')
                if clean_line:
                    subtasks.append(Task(description=clean_line))

        return subtasks if subtasks else [parent_task]

class CodeDecomposer(LadderDecomposer):
    """Specialized decomposer for coding tasks."""

    def can_decompose(self, task: Task) -> bool:
        """Check if task is code-related."""
        code_keywords = ['code', 'program', 'function', 'class', 'script', 'api', 'web app']
        return any(keyword in task.description.lower() for keyword in code_keywords)

    def decompose(self, task: Task) -> List[Task]:
        """Break down coding task into development phases."""
        base_desc = task.description

        subtasks = [
            Task(description=f"Design architecture for: {base_desc}"),
            Task(description=f"Implement core functionality: {base_desc}"),
            Task(description=f"Add error handling: {base_desc}"),
            Task(description=f"Write tests: {base_desc}"),
            Task(description=f"Document code: {base_desc}")
        ]

        return subtasks
```

## Implementation Steps

### Phase 1: Core Foundation

1. **Task Models** - Implement Task and TaskGraph classes
2. **Basic Planner** - Create LadderPlanner with simple execution
3. **Tool Integration** - Connect to existing tool registry

### Phase 2: Learning System

1. **Bandit Implementation** - Add multi-armed bandit for tool selection
2. **Reward System** - Implement reward calculation and feedback
3. **Decomposer Framework** - Create extensible decomposition system

### Phase 3: Advanced Features

1. **Shadow Mode** - Add safe execution simulation
2. **Energy Prioritization** - Implement task scheduling by energy
3. **Knowledge Graph** - Integrate with memory system

### Phase 4: Integration & Testing

1. **Orchestrator Adapter** - Build integration layer
2. **Configuration** - Add environment variables and settings
3. **Test Suite** - Comprehensive testing framework

## Usage Examples

### Basic Usage

```python
from ladder import LadderPlanner, DefaultDecomposer, BanditPolicy

# Initialize components
planner = LadderPlanner(
    tool_registry=tool_registry,
    decomposers=[DefaultDecomposer()],
    bandit_policy=BanditPolicy(algorithm="ucb1"),
    mode="shadow"
)

# Execute task
result = planner.handle_request("Create a simple web API")
```

### Custom Decomposer

```python
class DatabaseDecomposer(LadderDecomposer):
    def can_decompose(self, task: Task) -> bool:
        return 'database' in task.description.lower()

    def decompose(self, task: Task) -> List[Task]:
        return [
            Task(description="Design database schema"),
            Task(description="Create database tables"),
            Task(description="Implement data access layer"),
            Task(description="Add data validation")
        ]

# Register custom decomposer
planner.register_decomposer(DatabaseDecomposer())
```

## Testing Strategy

### Unit Tests

- Task and TaskGraph operations
- Bandit selection algorithms
- Decomposer logic
- Tool integration

### Integration Tests

- End-to-end task execution
- Shadow vs active mode validation
- Multi-step plan execution
- Error handling and recovery

### Performance Tests

- Plan creation latency
- Execution overhead
- Memory usage
- Bandit convergence rates

## Configuration

Key environment variables:

- `LADDER_MODE`: "shadow" or "active"
- `LADDER_BANDIT_ALGORITHM`: "epsilon_greedy" or "ucb1"
- `LADDER_MAX_DEPTH`: Maximum decomposition depth
- `LADDER_ENERGY_BUDGET`: Maximum energy per plan

## Next Steps

1. Implement core Task and TaskGraph models
2. Create basic LadderPlanner orchestrator
3. Add multi-armed bandit learning
4. Build decomposer framework
5. Integrate with existing Cortex system

See [LADDER_PLANNER_GUIDE.md](LADDER_PLANNER_GUIDE.md) for high-level overview.
