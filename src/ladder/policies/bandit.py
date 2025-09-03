"""Multi-Armed Bandit policies for adaptive tool selection in LADDER."""

import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BanditAlgorithm(Enum):
    """Available bandit algorithms."""

    UCB1 = "ucb1"
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON_SAMPLING = "thompson_sampling"
    EXP3 = "exp3"


@dataclass
class ToolMetrics:
    """Metrics for a single tool's performance."""

    tool_name: str
    total_uses: int = 0
    total_reward: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    last_used: float = field(default_factory=time.time)

    @property
    def average_reward(self) -> float:
        """Calculate average reward per use."""
        return self.total_reward / max(1, self.total_uses)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        total_attempts = self.success_count + self.failure_count
        return self.success_count / max(1, total_attempts)

    @property
    def confidence_interval(self) -> float:
        """Calculate confidence interval for UCB1."""
        if self.total_uses == 0:
            return float("inf")
        return math.sqrt(
            2 * math.log(time.time() - self.last_used + 1) / self.total_uses
        )

    def update_metrics(
        self, reward: float, success: bool, execution_time: float
    ) -> None:
        """Update metrics with new observation."""
        self.total_uses += 1
        self.total_reward += reward

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # Update average execution time (exponential moving average)
        alpha = 0.1  # Learning rate
        self.avg_execution_time = (
            alpha * execution_time + (1 - alpha) * self.avg_execution_time
        )

        self.last_used = time.time()


class BanditPolicy(ABC):
    """Abstract base class for multi-armed bandit policies."""

    def __init__(self, tools: list[str]):
        """Initialize policy with available tools."""
        self.tools = tools
        self.metrics: dict[str, ToolMetrics] = {
            tool: ToolMetrics(tool_name=tool) for tool in tools
        }
        self.total_selections = 0

    @abstractmethod
    def select_tool(self, context: dict[str, Any] | None = None) -> str:
        """Select a tool based on the bandit algorithm."""
        pass

    def update_reward(
        self,
        tool: str,
        reward: float,
        success: bool = True,
        execution_time: float = 0.0,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Update tool metrics with observed reward."""
        if tool in self.metrics:
            self.metrics[tool].update_metrics(reward, success, execution_time)

    def get_tool_stats(self) -> dict[str, dict[str, float]]:
        """Get current statistics for all tools."""
        return {
            tool: {
                "avg_reward": metrics.average_reward,
                "success_rate": metrics.success_rate,
                "total_uses": metrics.total_uses,
                "avg_execution_time": metrics.avg_execution_time,
            }
            for tool, metrics in self.metrics.items()
        }

    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        for tool in self.tools:
            self.metrics[tool] = ToolMetrics(tool_name=tool)
        self.total_selections = 0


class UCB1Policy(BanditPolicy):
    """
    Upper Confidence Bound (UCB1) bandit policy.

    Balances exploitation of high-reward tools with exploration
    of under-tried tools using confidence intervals.
    """

    def __init__(self, tools: list[str], exploration_factor: float = 2.0):
        """
        Initialize UCB1 policy.

        Args:
            tools: List of available tool names
            exploration_factor: Controls exploration vs exploitation balance
        """
        super().__init__(tools)
        self.exploration_factor = exploration_factor

    def select_tool(self, context: dict[str, Any] | None = None) -> str:
        """Select tool using UCB1 algorithm."""
        self.total_selections += 1

        # First, try each tool at least once
        for tool in self.tools:
            if self.metrics[tool].total_uses == 0:
                return tool

        # Calculate UCB1 scores for all tools
        ucb_scores = {}
        total_time = sum(m.total_uses for m in self.metrics.values())

        for tool, metrics in self.metrics.items():
            if metrics.total_uses == 0:
                ucb_scores[tool] = float("inf")
            else:
                confidence = math.sqrt(
                    self.exploration_factor * math.log(total_time) / metrics.total_uses
                )
                ucb_scores[tool] = metrics.average_reward + confidence

        # Select tool with highest UCB1 score
        return max(ucb_scores.items(), key=lambda x: x[1])[0]


class EpsilonGreedyPolicy(BanditPolicy):
    """
    Epsilon-greedy bandit policy.

    With probability epsilon, explores by choosing randomly.
    Otherwise, exploits by choosing the tool with highest average reward.
    """

    def __init__(
        self, tools: list[str], epsilon: float = 0.1, decay_rate: float = 0.995
    ):
        """
        Initialize epsilon-greedy policy.

        Args:
            tools: List of available tool names
            epsilon: Exploration probability (0.0 to 1.0)
            decay_rate: Rate at which epsilon decays over time
        """
        super().__init__(tools)
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def select_tool(self, context: dict[str, Any] | None = None) -> str:
        """Select tool using epsilon-greedy algorithm."""
        self.total_selections += 1

        # Decay epsilon over time
        self.epsilon *= self.decay_rate

        # Explore with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(self.tools)

        # Exploit: choose tool with highest average reward
        if all(m.total_uses == 0 for m in self.metrics.values()):
            return random.choice(self.tools)

        best_tool = max(
            self.metrics.items(),
            key=lambda x: x[1].average_reward if x[1].total_uses > 0 else -1,
        )
        return best_tool[0]


class ThompsonSamplingPolicy(BanditPolicy):
    """
    Thompson Sampling bandit policy.

    Uses Bayesian inference to sample from posterior distributions
    of tool rewards.
    """

    def __init__(self, tools: list[str]):
        """Initialize Thompson Sampling policy."""
        super().__init__(tools)
        # Beta distribution parameters for each tool
        self.alpha: dict[str, float] = dict.fromkeys(tools, 1.0)
        self.beta: dict[str, float] = dict.fromkeys(tools, 1.0)

    def select_tool(self, context: dict[str, Any] | None = None) -> str:
        """Select tool using Thompson Sampling."""
        self.total_selections += 1

        # Sample from Beta distribution for each tool
        samples = {}
        for tool in self.tools:
            samples[tool] = random.betavariate(self.alpha[tool], self.beta[tool])

        # Select tool with highest sample
        return max(samples.items(), key=lambda x: x[1])[0]

    def update_reward(
        self,
        tool: str,
        reward: float,
        success: bool = True,
        execution_time: float = 0.0,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Update Beta distribution parameters."""
        super().update_reward(tool, reward, success, execution_time, context)

        if tool in self.alpha:
            if success:
                self.alpha[tool] += reward
            else:
                self.beta[tool] += 1.0 - reward


class ContextualBanditPolicy(BanditPolicy):
    """
    Contextual bandit policy that considers task context.

    Uses simple feature-based heuristics to adjust tool selection
    based on task characteristics.
    """

    def __init__(
        self, tools: list[str], base_policy: BanditPolicy, context_weight: float = 0.3
    ):
        """
        Initialize contextual bandit policy.

        Args:
            tools: List of available tool names
            base_policy: Underlying bandit policy
            context_weight: Weight given to context vs base policy
        """
        super().__init__(tools)
        self.base_policy = base_policy
        self.context_weight = context_weight

        # Simple context-tool affinity scores
        self.context_affinities: dict[str, dict[str, float]] = {
            "coding": {
                "code_executor": 0.8,
                "file_writer": 0.6,
                "test_runner": 0.7,
                "llm": 0.4,
            },
            "research": {
                "web_search": 0.9,
                "knowledge_graph": 0.8,
                "llm": 0.7,
                "file_reader": 0.5,
            },
            "analysis": {
                "data_analyzer": 0.9,
                "llm": 0.8,
                "knowledge_graph": 0.6,
                "file_reader": 0.7,
            },
        }

    def select_tool(self, context: dict[str, Any] | None = None) -> str:
        """Select tool using context-aware scoring."""
        # Get base policy recommendation
        base_tool = self.base_policy.select_tool(context)

        if not context:
            return base_tool

        # Extract task type from context
        task_type = context.get("task_type", "general")

        if task_type not in self.context_affinities:
            return base_tool

        # Calculate context-adjusted scores
        context_scores = {}
        affinities = self.context_affinities[task_type]

        for tool in self.tools:
            base_score = (
                self.base_policy.metrics[tool].average_reward
                if self.base_policy.metrics[tool].total_uses > 0
                else 0.5
            )
            context_score = affinities.get(tool, 0.3)

            # Combine base score with context affinity
            combined_score = (
                1 - self.context_weight
            ) * base_score + self.context_weight * context_score
            context_scores[tool] = combined_score

        # Select tool with highest combined score
        return max(context_scores.items(), key=lambda x: x[1])[0]

    def update_reward(
        self,
        tool: str,
        reward: float,
        success: bool = True,
        execution_time: float = 0.0,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Update both local and base policy metrics."""
        super().update_reward(tool, reward, success, execution_time, context)
        self.base_policy.update_reward(tool, reward, success, execution_time, context)


def create_bandit_policy(
    algorithm: BanditAlgorithm, tools: list[str], **kwargs
) -> BanditPolicy:
    """Factory function to create bandit policies."""
    if algorithm == BanditAlgorithm.UCB1:
        return UCB1Policy(tools, **kwargs)
    elif algorithm == BanditAlgorithm.EPSILON_GREEDY:
        return EpsilonGreedyPolicy(tools, **kwargs)
    elif algorithm == BanditAlgorithm.THOMPSON_SAMPLING:
        return ThompsonSamplingPolicy(tools, **kwargs)
    else:
        raise ValueError(f"Unknown bandit algorithm: {algorithm}")


def calculate_reward(
    task_result: Any,
    execution_time: float,
    success: bool = True,
    context: dict[str, Any] | None = None,
) -> float:
    """
    Calculate reward for tool execution.

    This is a heuristic function that can be customized based on
    specific requirements and success metrics.
    """
    base_reward = 1.0 if success else 0.0

    # Time penalty (faster execution gets higher reward)
    time_factor = max(0.1, 1.0 / (1.0 + execution_time / 10.0))

    # Quality bonus based on result characteristics
    quality_bonus = 0.0
    if success and task_result:
        if isinstance(task_result, dict):
            # Reward detailed results
            quality_bonus = min(0.3, len(task_result) * 0.05)
        elif isinstance(task_result, str) and len(task_result) > 50:
            # Reward substantial text outputs
            quality_bonus = 0.2

    # Context-specific adjustments
    context_bonus = 0.0
    if context and success:
        priority = context.get("priority", "normal")
        if priority == "high":
            context_bonus = 0.1
        elif priority == "critical":
            context_bonus = 0.2

    total_reward = (
        base_reward * 0.6
        + time_factor * 0.2
        + quality_bonus * 0.1
        + context_bonus * 0.1
    )

    return max(0.0, min(1.0, total_reward))
