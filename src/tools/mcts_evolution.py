"""
MCTS-based evolutionary engine for Super Alita.
Implements creative evolution and skill discovery through Monte Carlo Tree Search.
"""

import json
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional


@dataclass
class EvolutionNode:
    """Node in the MCTS evolution tree."""

    id: str
    state: dict[str, Any]
    parent: Optional["EvolutionNode"] = None
    children: list["EvolutionNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def average_reward(self) -> float:
        """Average reward for this node."""
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    @property
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return self.metadata.get("is_terminal", False)

    def ucb1_score(self, exploration_weight: float = 1.4) -> float:
        """Calculate UCB1 score for selection."""
        if self.visits == 0:
            return float("inf")

        exploitation = self.average_reward
        exploration = (
            exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
            if self.parent
            else 0
        )

        return exploitation + exploration


class EvolutionEnvironment(ABC):
    """Abstract environment for evolution experiments."""

    @abstractmethod
    async def get_initial_state(self) -> dict[str, Any]:
        """Get initial state for evolution."""
        pass

    @abstractmethod
    async def get_available_actions(self, state: dict[str, Any]) -> list[str]:
        """Get available actions for a given state."""
        pass

    @abstractmethod
    async def apply_action(self, state: dict[str, Any], action: str) -> dict[str, Any]:
        """Apply action to state and return new state."""
        pass

    @abstractmethod
    async def evaluate_state(self, state: dict[str, Any]) -> float:
        """Evaluate fitness of a state."""
        pass

    @abstractmethod
    async def is_terminal(self, state: dict[str, Any]) -> bool:
        """Check if state is terminal."""
        pass


class SkillEvolutionEnvironment(EvolutionEnvironment):
    """Environment for evolving agent skills."""

    def __init__(self, base_skills: list[str], max_depth: int = 5):
        self.base_skills = base_skills
        self.max_depth = max_depth
        self.mutation_types = [
            "combine",
            "modify",
            "specialize",
            "generalize",
            "optimize",
        ]

    async def get_initial_state(self) -> dict[str, Any]:
        """Start with a random base skill."""
        return {
            "skills": [random.choice(self.base_skills)],
            "depth": 0,
            "fitness": 0.0,
            "metadata": {"evolution_history": []},
        }

    async def get_available_actions(self, state: dict[str, Any]) -> list[str]:
        """Get available mutation actions."""
        if state["depth"] >= self.max_depth:
            return []

        actions = []
        current_skills = state["skills"]

        # Combination actions
        if len(current_skills) >= 2:
            for i in range(len(current_skills)):
                for j in range(i + 1, len(current_skills)):
                    actions.append(f"combine_{i}_{j}")

        # Modification actions
        for i, skill in enumerate(current_skills):
            for mutation_type in self.mutation_types[1:]:  # Skip 'combine'
                actions.append(f"{mutation_type}_{i}")

        # Add new skill from base
        if len(current_skills) < 3:
            for skill in self.base_skills:
                if skill not in current_skills:
                    actions.append(f"add_{skill}")

        return actions

    async def apply_action(self, state: dict[str, Any], action: str) -> dict[str, Any]:
        """Apply evolution action."""
        new_state = {
            "skills": state["skills"].copy(),
            "depth": state["depth"] + 1,
            "fitness": 0.0,
            "metadata": {
                "evolution_history": state["metadata"]["evolution_history"] + [action]
            },
        }

        parts = action.split("_")
        action_type = parts[0]

        if action_type == "combine" and len(parts) == 3:
            idx1, idx2 = int(parts[1]), int(parts[2])
            skill1, skill2 = new_state["skills"][idx1], new_state["skills"][idx2]
            combined_skill = f"combined_{skill1}_{skill2}"
            new_state["skills"] = [
                s for i, s in enumerate(new_state["skills"]) if i not in [idx1, idx2]
            ] + [combined_skill]

        elif action_type in ["modify", "specialize", "generalize", "optimize"]:
            idx = int(parts[1])
            original_skill = new_state["skills"][idx]
            modified_skill = f"{action_type}_{original_skill}"
            new_state["skills"][idx] = modified_skill

        elif action_type == "add":
            skill_to_add = "_".join(parts[1:])
            new_state["skills"].append(skill_to_add)

        return new_state

    async def evaluate_state(self, state: dict[str, Any]) -> float:
        """Evaluate skill combination fitness."""
        skills = state["skills"]
        depth = state["depth"]

        # Base fitness from skill diversity and complexity
        diversity_score = len(set(skills)) / max(len(skills), 1)
        complexity_score = min(depth / self.max_depth, 1.0)

        # Bonus for interesting combinations
        combination_bonus = 0.0
        for skill in skills:
            if "combined_" in skill:
                combination_bonus += 0.2
            if any(modifier in skill for modifier in self.mutation_types[1:]):
                combination_bonus += 0.1

        # Penalty for excessive depth without progress
        depth_penalty = max(0, (depth - 3) * 0.1) if depth > 3 else 0

        fitness = diversity_score + complexity_score + combination_bonus - depth_penalty

        # Add some randomness to simulate real-world uncertainty
        fitness += random.gauss(0, 0.05)

        return max(0, fitness)

    async def is_terminal(self, state: dict[str, Any]) -> bool:
        """Check if evolution should stop."""
        return state["depth"] >= self.max_depth or len(state["skills"]) == 0


class MCTSEvolutionArena:
    """
    Monte Carlo Tree Search-based evolution arena.

    Uses MCTS to explore the space of possible evolutions and find
    high-fitness solutions through intelligent exploration.
    """

    def __init__(
        self,
        environment: EvolutionEnvironment,
        exploration_weight: float = 1.4,
        max_iterations: int = 1000,
        max_depth: int = 50,
    ):
        self.environment = environment
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.root: EvolutionNode | None = None
        self.best_solutions: list[tuple[float, dict[str, Any]]] = []
        self.iteration_count = 0
        self.start_time: datetime | None = None

    async def evolve(self, num_generations: int = 10) -> list[dict[str, Any]]:
        """
        Run evolution for specified number of generations.

        Args:
            num_generations: Number of evolution generations to run

        Returns:
            List of best evolved solutions
        """

        self.start_time = datetime.now(UTC)
        all_solutions = []

        for generation in range(num_generations):
            print(f"Evolution Generation {generation + 1}/{num_generations}")

            # Initialize root node for this generation
            initial_state = await self.environment.get_initial_state()
            self.root = EvolutionNode(
                id=f"gen_{generation}_root",
                state=initial_state,
                untried_actions=await self.environment.get_available_actions(
                    initial_state
                ),
            )

            # Run MCTS iterations
            for iteration in range(self.max_iterations):
                await self._mcts_iteration()
                self.iteration_count += 1

                # Periodically collect good solutions
                if iteration % 100 == 0:
                    await self._collect_solutions()

            # Collect final solutions for this generation
            generation_solutions = await self._collect_solutions()
            all_solutions.extend(generation_solutions)

            print(
                f"Generation {generation + 1} complete: {len(generation_solutions)} solutions found"
            )

        # Sort and return best solutions
        all_solutions.sort(key=lambda x: x[0], reverse=True)
        return [solution for fitness, solution in all_solutions[:20]]

    async def _mcts_iteration(self) -> None:
        """Single MCTS iteration: Select, Expand, Simulate, Backpropagate."""

        # Selection
        node = await self._select(self.root)

        # Expansion
        if (
            not await self.environment.is_terminal(node.state)
            and not node.is_fully_expanded
        ):
            node = await self._expand(node)

        # Simulation
        reward = await self._simulate(node)

        # Backpropagation
        await self._backpropagate(node, reward)

    async def _select(self, node: EvolutionNode) -> EvolutionNode:
        """Select most promising node using UCB1."""

        current = node
        depth = 0

        while (
            not await self.environment.is_terminal(current.state)
            and current.is_fully_expanded
            and current.children
            and depth < self.max_depth
        ):
            # Select child with highest UCB1 score
            current = max(
                current.children, key=lambda c: c.ucb1_score(self.exploration_weight)
            )
            depth += 1

        return current

    async def _expand(self, node: EvolutionNode) -> EvolutionNode:
        """Expand node by trying an untried action."""

        if not node.untried_actions:
            return node

        # Choose random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # Apply action to get new state
        new_state = await self.environment.apply_action(node.state, action)

        # Create child node
        child = EvolutionNode(
            id=f"{node.id}_child_{len(node.children)}",
            state=new_state,
            parent=node,
            untried_actions=await self.environment.get_available_actions(new_state),
            metadata={"action_taken": action},
        )

        node.children.append(child)
        return child

    async def _simulate(self, node: EvolutionNode) -> float:
        """Simulate random rollout from node to estimate value."""

        current_state = node.state.copy()
        depth = 0
        total_reward = 0.0

        while (
            not await self.environment.is_terminal(current_state)
            and depth < self.max_depth
        ):
            actions = await self.environment.get_available_actions(current_state)
            if not actions:
                break

            # Choose random action
            action = random.choice(actions)
            current_state = await self.environment.apply_action(current_state, action)

            # Accumulate rewards
            reward = await self.environment.evaluate_state(current_state)
            total_reward += reward * (0.9**depth)  # Discount future rewards

            depth += 1

        # Final evaluation
        final_reward = await self.environment.evaluate_state(current_state)
        total_reward += final_reward * (0.9**depth)

        return total_reward / (depth + 1)  # Average reward

    async def _backpropagate(self, node: EvolutionNode, reward: float) -> None:
        """Backpropagate reward up the tree."""

        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    async def _collect_solutions(self) -> list[tuple[float, dict[str, Any]]]:
        """Collect high-quality solutions from the tree."""

        solutions = []

        def collect_from_node(node: EvolutionNode):
            # Evaluate current node
            if node.visits > 5:  # Only consider well-explored nodes
                fitness = node.average_reward
                if fitness > 0.5:  # Threshold for good solutions
                    solutions.append((fitness, node.state))

            # Recursively collect from children
            for child in node.children:
                collect_from_node(child)

        if self.root:
            collect_from_node(self.root)

        # Sort by fitness and return top solutions
        solutions.sort(key=lambda x: x[0], reverse=True)
        return solutions[:10]

    async def get_evolution_stats(self) -> dict[str, Any]:
        """Get statistics about the evolution process."""

        total_nodes = 0
        max_depth = 0
        avg_reward = 0.0

        def analyze_node(node: EvolutionNode, depth: int = 0):
            nonlocal total_nodes, max_depth, avg_reward
            total_nodes += 1
            max_depth = max(max_depth, depth)
            avg_reward += node.average_reward

            for child in node.children:
                analyze_node(child, depth + 1)

        if self.root:
            analyze_node(self.root)

        runtime = (
            (datetime.now(UTC) - self.start_time).total_seconds()
            if self.start_time
            else 0
        )

        return {
            "total_iterations": self.iteration_count,
            "total_nodes": total_nodes,
            "max_tree_depth": max_depth,
            "average_reward": avg_reward / max(total_nodes, 1),
            "best_solutions_found": len(self.best_solutions),
            "runtime_seconds": runtime,
            "iterations_per_second": self.iteration_count / max(runtime, 1),
        }

    async def export_tree(self, filepath: str) -> None:
        """Export evolution tree to JSON for analysis."""

        def node_to_dict(node: EvolutionNode) -> dict[str, Any]:
            return {
                "id": node.id,
                "state": node.state,
                "visits": node.visits,
                "total_reward": node.total_reward,
                "average_reward": node.average_reward,
                "untried_actions": node.untried_actions,
                "created_at": node.created_at.isoformat(),
                "metadata": node.metadata,
                "children": [node_to_dict(child) for child in node.children],
            }

        export_data = {
            "root": node_to_dict(self.root) if self.root else None,
            "best_solutions": self.best_solutions,
            "stats": await self.get_evolution_stats(),
            "export_timestamp": datetime.now(UTC).isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)


# Predefined evolution environments


class CodeEvolutionEnvironment(EvolutionEnvironment):
    """Environment for evolving code structures."""

    def __init__(self, base_functions: list[str]):
        self.base_functions = base_functions
        self.operations = ["compose", "refactor", "optimize", "generalize"]

    async def get_initial_state(self) -> dict[str, Any]:
        return {
            "functions": [random.choice(self.base_functions)],
            "complexity": 1,
            "performance": 0.5,
        }

    async def get_available_actions(self, state: dict[str, Any]) -> list[str]:
        actions = []
        for op in self.operations:
            actions.append(op)
        return actions

    async def apply_action(self, state: dict[str, Any], action: str) -> dict[str, Any]:
        new_state = state.copy()

        if action == "compose":
            new_func = f"composed_{len(new_state['functions'])}"
            new_state["functions"].append(new_func)
            new_state["complexity"] += 0.5
        elif action == "optimize":
            new_state["performance"] *= 1.2
            new_state["complexity"] *= 0.9
        elif action == "refactor":
            new_state["complexity"] *= 0.8
            new_state["performance"] *= 1.1
        elif action == "generalize":
            new_state["complexity"] += 0.3
            new_state["performance"] *= 0.95

        return new_state

    async def evaluate_state(self, state: dict[str, Any]) -> float:
        # Balance between functionality, performance, and simplicity
        func_score = len(state["functions"]) * 0.3
        perf_score = min(state["performance"], 2.0) * 0.4
        complexity_penalty = max(0, state["complexity"] - 2) * 0.2

        return func_score + perf_score - complexity_penalty

    async def is_terminal(self, state: dict[str, Any]) -> bool:
        return len(state["functions"]) > 5 or state["complexity"] > 10


# Factory function for creating evolution arenas


async def create_skill_evolution_arena(
    base_skills: list[str], max_iterations: int = 1000
) -> MCTSEvolutionArena:
    """Create an arena for evolving agent skills."""

    environment = SkillEvolutionEnvironment(base_skills)
    return MCTSEvolutionArena(environment=environment, max_iterations=max_iterations)


async def create_code_evolution_arena(
    base_functions: list[str], max_iterations: int = 1000
) -> MCTSEvolutionArena:
    """Create an arena for evolving code structures."""

    environment = CodeEvolutionEnvironment(base_functions)
    return MCTSEvolutionArena(environment=environment, max_iterations=max_iterations)
