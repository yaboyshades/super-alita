# Planning Algorithms - Agent Instructions

## Overview
The `src/planning/` directory contains planning and decision-making algorithms:
- **Synchronization** - One-time synchronization planning utilities
- **Decision Trees** - Decision-making algorithm implementations
- **Goal Planning** - Goal-oriented planning and path finding
- **Strategy Selection** - Strategy selection and optimization

## Key Files & Responsibilities

### Planning Components
- `sync_once.py` - Single synchronization planning utility
- Additional planning modules (to be implemented)

## Development Guidelines

### Planning Algorithm Structure
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class PlanningStrategy(Enum):
    GREEDY = "greedy"
    OPTIMAL = "optimal"
    HEURISTIC = "heuristic"
    REINFORCEMENT = "reinforcement"

@dataclass
class PlanningGoal:
    """Goal definition for planning algorithms"""
    goal_id: str
    description: str
    priority: int = 1
    constraints: Dict[str, Any] = None
    success_criteria: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}
        if self.success_criteria is None:
            self.success_criteria = {}

@dataclass
class PlanningAction:
    """Action that can be taken in planning"""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    preconditions: List[str] = None
    effects: List[str] = None
    cost: float = 1.0
    duration: float = 1.0
    
    def __post_init__(self):
        if self.preconditions is None:
            self.preconditions = []
        if self.effects is None:
            self.effects = []

@dataclass
class Plan:
    """Planning result containing sequence of actions"""
    plan_id: str
    goal: PlanningGoal
    actions: List[PlanningAction]
    total_cost: float
    estimated_duration: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class PlanningAlgorithm(ABC):
    """Base class for planning algorithms"""
    
    def __init__(self, strategy: PlanningStrategy = PlanningStrategy.HEURISTIC):
        self.strategy = strategy
        self.state_space: Dict[str, Any] = {}
        self.action_library: List[PlanningAction] = []
        
    @abstractmethod
    async def generate_plan(self, goal: PlanningGoal, context: Dict[str, Any]) -> Plan:
        """Generate plan to achieve goal"""
        pass
        
    @abstractmethod
    def evaluate_plan(self, plan: Plan, context: Dict[str, Any]) -> float:
        """Evaluate plan quality/feasibility"""
        pass
        
    def add_action(self, action: PlanningAction):
        """Add action to action library"""
        self.action_library.append(action)
        
    def update_state(self, state_updates: Dict[str, Any]):
        """Update planning state space"""
        self.state_space.update(state_updates)
```

### Goal-Oriented Planning
```python
import heapq
from typing import Set, Tuple

class GoalOrientedPlanner(PlanningAlgorithm):
    """Goal-oriented planning using A* algorithm"""
    
    def __init__(self, strategy: PlanningStrategy = PlanningStrategy.OPTIMAL):
        super().__init__(strategy)
        self.heuristic_weights = {
            'cost': 0.4,
            'duration': 0.3,
            'success_probability': 0.3
        }
        
    async def generate_plan(self, goal: PlanningGoal, context: Dict[str, Any]) -> Plan:
        """Generate plan using A* search"""
        start_state = self._get_current_state(context)
        goal_state = self._define_goal_state(goal)
        
        # A* search implementation
        open_set = [(0, start_state, [])]  # (f_score, state, actions)
        closed_set: Set[str] = set()
        
        while open_set:
            f_score, current_state, actions_taken = heapq.heappop(open_set)
            
            if self._is_goal_achieved(current_state, goal_state):
                # Goal reached, construct plan
                total_cost = sum(action.cost for action in actions_taken)
                total_duration = sum(action.duration for action in actions_taken)
                
                return Plan(
                    plan_id=f"plan_{goal.goal_id}_{int(time.time())}",
                    goal=goal,
                    actions=actions_taken,
                    total_cost=total_cost,
                    estimated_duration=total_duration,
                    confidence=self._calculate_confidence(actions_taken, context)
                )
            
            state_key = self._state_to_key(current_state)
            if state_key in closed_set:
                continue
                
            closed_set.add(state_key)
            
            # Explore possible actions
            for action in self._get_applicable_actions(current_state):
                new_state = self._apply_action(current_state, action)
                new_actions = actions_taken + [action]
                
                g_score = sum(a.cost for a in new_actions)
                h_score = self._heuristic(new_state, goal_state)
                f_score = g_score + h_score
                
                heapq.heappush(open_set, (f_score, new_state, new_actions))
        
        # No plan found
        raise PlanningError(f"No plan found for goal: {goal.goal_id}")
        
    def _heuristic(self, state: Dict[str, Any], goal_state: Dict[str, Any]) -> float:
        """Heuristic function for A* search"""
        # Simple Manhattan distance for numeric values
        distance = 0.0
        
        for key, target_value in goal_state.items():
            current_value = state.get(key, 0)
            
            if isinstance(target_value, (int, float)) and isinstance(current_value, (int, float)):
                distance += abs(target_value - current_value)
            elif target_value != current_value:
                distance += 1.0
                
        return distance
        
    def _get_applicable_actions(self, state: Dict[str, Any]) -> List[PlanningAction]:
        """Get actions applicable in current state"""
        applicable = []
        
        for action in self.action_library:
            if self._are_preconditions_met(action.preconditions, state):
                applicable.append(action)
                
        return applicable
        
    def _are_preconditions_met(self, preconditions: List[str], state: Dict[str, Any]) -> bool:
        """Check if action preconditions are met"""
        for condition in preconditions:
            if not self._evaluate_condition(condition, state):
                return False
        return True
        
    def _evaluate_condition(self, condition: str, state: Dict[str, Any]) -> bool:
        """Evaluate a single condition against state"""
        # Simple condition evaluation (can be extended)
        if condition.startswith("has_"):
            key = condition[4:]
            return state.get(key, False)
        elif "=" in condition:
            key, value = condition.split("=", 1)
            return str(state.get(key.strip(), "")) == value.strip()
        else:
            return state.get(condition, False)
```

### Reinforcement Learning Planner
```python
import numpy as np
from typing import Dict, Any, List, Tuple

class RLPlanner(PlanningAlgorithm):
    """Planning using reinforcement learning"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        super().__init__(PlanningStrategy.REINFORCEMENT)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table: Dict[Tuple[str, str], float] = {}  # (state, action) -> value
        self.episode_history: List[Dict[str, Any]] = []
        
    async def generate_plan(self, goal: PlanningGoal, context: Dict[str, Any]) -> Plan:
        """Generate plan using Q-learning"""
        current_state = self._get_current_state(context)
        goal_state = self._define_goal_state(goal)
        
        actions_taken = []
        max_steps = 100  # Prevent infinite loops
        
        for step in range(max_steps):
            if self._is_goal_achieved(current_state, goal_state):
                break
                
            # Select action using epsilon-greedy
            action = self._select_action(current_state, epsilon=0.1)
            
            if action is None:
                break
                
            actions_taken.append(action)
            
            # Simulate action execution
            next_state = self._apply_action(current_state, action)
            reward = self._calculate_reward(current_state, action, next_state, goal_state)
            
            # Update Q-value
            self._update_q_value(current_state, action, reward, next_state)
            
            current_state = next_state
            
        if not self._is_goal_achieved(current_state, goal_state):
            raise PlanningError("RL planner failed to reach goal")
            
        total_cost = sum(action.cost for action in actions_taken)
        total_duration = sum(action.duration for action in actions_taken)
        
        return Plan(
            plan_id=f"rl_plan_{goal.goal_id}_{int(time.time())}",
            goal=goal,
            actions=actions_taken,
            total_cost=total_cost,
            estimated_duration=total_duration,
            confidence=self._calculate_rl_confidence(actions_taken)
        )
        
    def _select_action(self, state: Dict[str, Any], epsilon: float = 0.1) -> Optional[PlanningAction]:
        """Select action using epsilon-greedy strategy"""
        applicable_actions = self._get_applicable_actions(state)
        
        if not applicable_actions:
            return None
            
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.choice(applicable_actions)
        else:
            # Exploit: best known action
            best_action = None
            best_value = float('-inf')
            
            state_key = self._state_to_key(state)
            
            for action in applicable_actions:
                q_value = self.q_table.get((state_key, action.action_id), 0.0)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
                    
            return best_action or applicable_actions[0]
            
    def _update_q_value(self, state: Dict[str, Any], action: PlanningAction, 
                       reward: float, next_state: Dict[str, Any]):
        """Update Q-value using Q-learning update rule"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Current Q-value
        current_q = self.q_table.get((state_key, action.action_id), 0.0)
        
        # Maximum Q-value for next state
        next_actions = self._get_applicable_actions(next_state)
        max_next_q = 0.0
        
        if next_actions:
            max_next_q = max(
                self.q_table.get((next_state_key, a.action_id), 0.0)
                for a in next_actions
            )
            
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[(state_key, action.action_id)] = new_q
        
    def _calculate_reward(self, state: Dict[str, Any], action: PlanningAction,
                         next_state: Dict[str, Any], goal_state: Dict[str, Any]) -> float:
        """Calculate reward for state transition"""
        # Goal achievement bonus
        if self._is_goal_achieved(next_state, goal_state):
            return 100.0
            
        # Progress toward goal
        current_distance = self._heuristic(state, goal_state)
        next_distance = self._heuristic(next_state, goal_state)
        progress_reward = current_distance - next_distance
        
        # Action cost penalty
        cost_penalty = -action.cost
        
        return progress_reward + cost_penalty
```

### Synchronization Planning
```python
from datetime import datetime, timedelta
from typing import List, Dict, Any

class SynchronizationPlanner:
    """Planning for synchronization tasks"""
    
    def __init__(self):
        self.sync_tasks: List[Dict[str, Any]] = []
        self.dependencies: Dict[str, List[str]] = {}
        
    def add_sync_task(self, task_id: str, task_config: Dict[str, Any]):
        """Add synchronization task"""
        self.sync_tasks.append({
            'id': task_id,
            'config': task_config,
            'status': 'pending',
            'scheduled_time': None,
            'duration': task_config.get('duration', 60)  # seconds
        })
        
    def add_dependency(self, task_id: str, depends_on: List[str]):
        """Add task dependencies"""
        self.dependencies[task_id] = depends_on
        
    async def plan_synchronization(self) -> List[Dict[str, Any]]:
        """Plan synchronization task execution order"""
        # Topological sort for dependency resolution
        execution_plan = []
        completed_tasks = set()
        
        while len(completed_tasks) < len(self.sync_tasks):
            ready_tasks = []
            
            for task in self.sync_tasks:
                if task['id'] in completed_tasks:
                    continue
                    
                dependencies = self.dependencies.get(task['id'], [])
                if all(dep in completed_tasks for dep in dependencies):
                    ready_tasks.append(task)
                    
            if not ready_tasks:
                raise PlanningError("Circular dependency detected in sync tasks")
                
            # Schedule ready tasks
            current_time = datetime.now()
            for task in ready_tasks:
                task['scheduled_time'] = current_time
                current_time += timedelta(seconds=task['duration'])
                execution_plan.append(task)
                completed_tasks.add(task['id'])
                
        return execution_plan
        
    async def execute_sync_once(self, task_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single synchronization task"""
        start_time = datetime.now()
        
        try:
            # Simulate sync execution
            result = await self._perform_sync_operation(task_id, config)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                'task_id': task_id,
                'success': True,
                'duration': duration,
                'result': result,
                'timestamp': end_time.isoformat()
            }
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                'task_id': task_id,
                'success': False,
                'duration': duration,
                'error': str(e),
                'timestamp': end_time.isoformat()
            }
            
    async def _perform_sync_operation(self, task_id: str, config: Dict[str, Any]) -> Any:
        """Perform actual synchronization operation"""
        # Implementation specific to sync type
        sync_type = config.get('type', 'default')
        
        if sync_type == 'file_sync':
            return await self._sync_files(config)
        elif sync_type == 'data_sync':
            return await self._sync_data(config)
        elif sync_type == 'state_sync':
            return await self._sync_state(config)
        else:
            raise ValueError(f"Unknown sync type: {sync_type}")
```

## Testing Guidelines

### Planning Algorithm Testing
```python
import pytest
from unittest.mock import AsyncMock
from src.planning.goal_planner import GoalOrientedPlanner, PlanningGoal, PlanningAction

@pytest.mark.asyncio
async def test_goal_oriented_planning():
    """Test goal-oriented planning"""
    planner = GoalOrientedPlanner()
    
    # Setup action library
    planner.add_action(PlanningAction(
        action_id="move_to_a",
        action_type="move",
        parameters={"target": "A"},
        effects=["at_A"],
        cost=1.0
    ))
    
    planner.add_action(PlanningAction(
        action_id="move_to_b",
        action_type="move", 
        parameters={"target": "B"},
        preconditions=["at_A"],
        effects=["at_B"],
        cost=2.0
    ))
    
    # Define goal
    goal = PlanningGoal(
        goal_id="reach_b",
        description="Reach location B",
        success_criteria={"location": "B"}
    )
    
    # Generate plan
    context = {"current_location": "start"}
    plan = await planner.generate_plan(goal, context)
    
    # Verify plan
    assert len(plan.actions) >= 2
    assert plan.total_cost > 0
    assert plan.goal.goal_id == "reach_b"

@pytest.mark.asyncio
async def test_rl_planner_learning():
    """Test reinforcement learning planner"""
    planner = RLPlanner(learning_rate=0.5)
    
    # Train with multiple episodes
    for episode in range(10):
        goal = PlanningGoal(
            goal_id=f"episode_{episode}",
            description="Test goal",
            success_criteria={"value": 10}
        )
        
        context = {"current_value": 0}
        
        try:
            plan = await planner.generate_plan(goal, context)
            # Q-table should be updated
            assert len(planner.q_table) > 0
        except PlanningError:
            # Early episodes may fail
            pass

def test_synchronization_planning():
    """Test synchronization planning"""
    planner = SynchronizationPlanner()
    
    # Add tasks with dependencies
    planner.add_sync_task("task_a", {"type": "file_sync", "duration": 30})
    planner.add_sync_task("task_b", {"type": "data_sync", "duration": 60})
    planner.add_sync_task("task_c", {"type": "state_sync", "duration": 45})
    
    # task_b depends on task_a, task_c depends on task_b
    planner.add_dependency("task_b", ["task_a"])
    planner.add_dependency("task_c", ["task_b"])
    
    # Plan execution
    execution_plan = asyncio.run(planner.plan_synchronization())
    
    # Verify order
    task_order = [task['id'] for task in execution_plan]
    assert task_order == ["task_a", "task_b", "task_c"]
```

### Performance Testing
```python
@pytest.mark.performance
async def test_planning_performance():
    """Test planning algorithm performance"""
    planner = GoalOrientedPlanner()
    
    # Add many actions
    for i in range(100):
        planner.add_action(PlanningAction(
            action_id=f"action_{i}",
            action_type="test",
            parameters={"value": i},
            cost=1.0
        ))
    
    goal = PlanningGoal(
        goal_id="performance_test",
        description="Performance test goal"
    )
    
    start_time = time.time()
    
    try:
        plan = await planner.generate_plan(goal, {})
        planning_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert planning_time < 5.0  # 5 seconds max
        
    except PlanningError:
        # No valid plan is acceptable for performance test
        pass
```

## Performance Guidelines

### Optimization Strategies
```python
from functools import lru_cache
import asyncio

class OptimizedPlanner(PlanningAlgorithm):
    """Planning algorithm with performance optimizations"""
    
    def __init__(self):
        super().__init__()
        self.state_cache: Dict[str, Any] = {}
        self.plan_cache: Dict[str, Plan] = {}
        
    @lru_cache(maxsize=1000)
    def _cached_heuristic(self, state_key: str, goal_key: str) -> float:
        """Cached heuristic calculation"""
        # Convert keys back to states for calculation
        state = self._key_to_state(state_key)
        goal = self._key_to_state(goal_key)
        return self._heuristic(state, goal)
        
    async def _parallel_action_evaluation(self, state: Dict[str, Any]) -> List[PlanningAction]:
        """Evaluate actions in parallel"""
        tasks = [
            self._evaluate_action_async(action, state)
            for action in self.action_library
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful evaluations
        valid_actions = []
        for action, result in zip(self.action_library, results):
            if isinstance(result, bool) and result:
                valid_actions.append(action)
                
        return valid_actions
        
    async def _evaluate_action_async(self, action: PlanningAction, state: Dict[str, Any]) -> bool:
        """Asynchronously evaluate if action is applicable"""
        return self._are_preconditions_met(action.preconditions, state)
```

## Common Patterns

### Planning with Uncertainty
```python
from typing import Tuple
import random

class UncertaintyAwarePlanner(PlanningAlgorithm):
    """Planning that accounts for uncertainty"""
    
    def __init__(self):
        super().__init__()
        self.uncertainty_model: Dict[str, float] = {}
        
    async def generate_robust_plan(self, goal: PlanningGoal, context: Dict[str, Any]) -> Plan:
        """Generate plan robust to uncertainty"""
        # Generate multiple candidate plans
        candidate_plans = []
        
        for _ in range(5):  # Generate 5 candidates
            try:
                plan = await self._generate_single_plan(goal, context)
                robustness = self._evaluate_robustness(plan)
                candidate_plans.append((plan, robustness))
            except PlanningError:
                continue
                
        if not candidate_plans:
            raise PlanningError("No robust plans found")
            
        # Select most robust plan
        best_plan, best_robustness = max(candidate_plans, key=lambda x: x[1])
        best_plan.confidence = best_robustness
        
        return best_plan
        
    def _evaluate_robustness(self, plan: Plan) -> float:
        """Evaluate plan robustness to uncertainty"""
        robustness_score = 1.0
        
        for action in plan.actions:
            # Factor in action uncertainty
            uncertainty = self.uncertainty_model.get(action.action_type, 0.1)
            robustness_score *= (1.0 - uncertainty)
            
        return robustness_score
        
    def add_uncertainty(self, action_type: str, uncertainty: float):
        """Add uncertainty model for action type"""
        self.uncertainty_model[action_type] = uncertainty
```

### Hierarchical Planning
```python
class HierarchicalPlanner(PlanningAlgorithm):
    """Hierarchical task network planning"""
    
    def __init__(self):
        super().__init__()
        self.task_hierarchies: Dict[str, List[str]] = {}
        self.abstract_actions: Dict[str, List[PlanningAction]] = {}
        
    def define_task_hierarchy(self, abstract_task: str, subtasks: List[str]):
        """Define task decomposition hierarchy"""
        self.task_hierarchies[abstract_task] = subtasks
        
    async def generate_hierarchical_plan(self, goal: PlanningGoal, context: Dict[str, Any]) -> Plan:
        """Generate plan using hierarchical decomposition"""
        # Start with abstract plan
        abstract_plan = await self._plan_abstract_level(goal, context)
        
        # Decompose abstract actions
        concrete_actions = []
        
        for abstract_action in abstract_plan:
            if abstract_action.action_id in self.task_hierarchies:
                # Decompose this abstract action
                subtasks = self.task_hierarchies[abstract_action.action_id]
                subactions = await self._plan_subtasks(subtasks, context)
                concrete_actions.extend(subactions)
            else:
                # Already concrete action
                concrete_actions.append(abstract_action)
                
        total_cost = sum(action.cost for action in concrete_actions)
        total_duration = sum(action.duration for action in concrete_actions)
        
        return Plan(
            plan_id=f"hierarchical_{goal.goal_id}_{int(time.time())}",
            goal=goal,
            actions=concrete_actions,
            total_cost=total_cost,
            estimated_duration=total_duration
        )
```

## Debugging Tips
- **Plan visualization** - Visualize generated plans and decision trees
- **State space exploration** - Monitor state space exploration during planning
- **Action evaluation** - Debug action precondition and effect evaluation
- **Performance profiling** - Profile planning algorithm performance
- **Convergence monitoring** - Monitor RL algorithm convergence