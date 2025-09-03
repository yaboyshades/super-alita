"""
Non-stationary Multi-Armed Bandit optimization for adaptive tool selection
Implements advanced bandit algorithms with drift detection and adaptation
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np


@dataclass
class BanditArm:
    """Individual arm in the multi-armed bandit"""

    arm_id: int
    reward_estimates: list[float] = field(default_factory=list)
    action_count: int = 0
    total_reward: float = 0.0
    confidence_bound: float = 0.0
    last_selected: datetime | None = None

    @property
    def average_reward(self) -> float:
        """Calculate average reward for this arm"""
        return self.total_reward / max(1, self.action_count)

    def update_reward(self, reward: float):
        """Update arm with new reward"""
        self.action_count += 1
        self.total_reward += reward
        self.reward_estimates.append(reward)
        self.last_selected = datetime.now(UTC)

        # Limit history size
        if len(self.reward_estimates) > 1000:
            self.reward_estimates = self.reward_estimates[-500:]

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive arm statistics"""
        return {
            "arm_id": self.arm_id,
            "action_count": self.action_count,
            "average_reward": self.average_reward,
            "total_reward": self.total_reward,
            "confidence_bound": self.confidence_bound,
            "last_selected": (
                self.last_selected.isoformat() if self.last_selected else None
            ),
            "recent_performance": {
                "last_10_avg": (
                    np.mean(self.reward_estimates[-10:])
                    if len(self.reward_estimates) >= 10
                    else self.average_reward
                ),
                "variance": (
                    np.var(self.reward_estimates)
                    if len(self.reward_estimates) > 1
                    else 0.0
                ),
                "trend": self._calculate_trend(),
            },
        }

    def _calculate_trend(self) -> float:
        """Calculate trend in recent rewards"""
        if len(self.reward_estimates) < 5:
            return 0.0

        recent = self.reward_estimates[-10:]
        if len(recent) < 2:
            return 0.0

        # Simple linear trend
        x = np.arange(len(recent))
        y = np.array(recent)

        try:
            # Calculate slope of linear regression
            slope = (
                np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
                if np.std(x) > 0
                else 0.0
            )
            return slope
        except:
            return 0.0


class ChangePointDetector:
    """
    Detect change points in reward distributions for non-stationary environments
    """

    def __init__(self, window_size: int = 50, threshold: float = 2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.change_points: list[int] = []

    def detect_change(self, rewards: list[float]) -> bool:
        """
        Detect if there's been a significant change in reward distribution
        """
        if len(rewards) < self.window_size * 2:
            return False

        # Compare recent window to previous window
        recent_window = rewards[-self.window_size :]
        previous_window = rewards[-2 * self.window_size : -self.window_size]

        # Statistical test for distribution change
        recent_mean = np.mean(recent_window)
        previous_mean = np.mean(previous_window)
        recent_std = np.std(recent_window)
        previous_std = np.std(previous_window)

        # Welch's t-test approximation
        pooled_std = np.sqrt((recent_std**2 + previous_std**2) / 2)
        if pooled_std == 0:
            return False

        t_statistic = abs(recent_mean - previous_mean) / (
            pooled_std * np.sqrt(2 / self.window_size)
        )

        # If t-statistic exceeds threshold, change detected
        if t_statistic > self.threshold:
            self.change_points.append(len(rewards))
            return True

        return False

    def get_latest_change_distance(self, total_samples: int) -> int:
        """Get distance to most recent change point"""
        if not self.change_points:
            return total_samples

        return total_samples - max(self.change_points)


class NonStationaryBandit:
    """
    Non-stationary multi-armed bandit with adaptive exploration
    """

    def __init__(
        self,
        n_arms: int,
        learning_rate: float = 0.1,
        exploration_param: float = 2.0,
        algorithm: str = "ucb_sliding",  # ucb_sliding, exp3, thompson_sampling
        window_size: int = 100,
    ):
        self.n_arms = n_arms
        self.learning_rate = learning_rate
        self.exploration_param = exploration_param
        self.algorithm = algorithm
        self.window_size = window_size

        # Initialize arms
        self.arms = [BanditArm(i) for i in range(n_arms)]

        # Algorithm-specific parameters
        self.total_actions = 0
        self.gamma = 0.95  # Discount factor for sliding window

        # Change detection
        self.change_detector = ChangePointDetector()

        # Performance tracking
        self.regret_history: list[float] = []
        self.optimal_arm_history: list[int] = []

        # Thompson Sampling parameters
        self.alpha_params = np.ones(n_arms)  # Beta distribution alpha
        self.beta_params = np.ones(n_arms)  # Beta distribution beta

        # EXP3 parameters
        self.weights = np.ones(n_arms)
        self.probabilities = np.ones(n_arms) / n_arms

        self.logger = logging.getLogger("nonstationary_bandit")

    def select_arm(self) -> int:
        """
        Select arm based on configured algorithm
        """
        self.total_actions += 1

        if self.algorithm == "ucb_sliding":
            return self._ucb_sliding_window()
        elif self.algorithm == "exp3":
            return self._exp3_select()
        elif self.algorithm == "thompson_sampling":
            return self._thompson_sampling()
        else:
            # Fallback to epsilon-greedy
            return self._epsilon_greedy()

    def update(self, arm_id: int, reward: float):
        """
        Update bandit with observed reward
        """
        if arm_id < 0 or arm_id >= self.n_arms:
            raise ValueError(f"Invalid arm_id: {arm_id}")

        # Update arm
        self.arms[arm_id].update_reward(reward)

        # Update algorithm-specific parameters
        if self.algorithm == "exp3":
            self._exp3_update(arm_id, reward)
        elif self.algorithm == "thompson_sampling":
            self._thompson_update(arm_id, reward)

        # Check for change points
        all_rewards = []
        for arm in self.arms:
            all_rewards.extend(arm.reward_estimates)

        if self.change_detector.detect_change(all_rewards):
            self._handle_change_point()

        # Update regret tracking
        self._update_regret(arm_id, reward)

    def _ucb_sliding_window(self) -> int:
        """
        Upper Confidence Bound with sliding window for non-stationarity
        """
        if self.total_actions <= self.n_arms:
            # Explore each arm at least once
            for i, arm in enumerate(self.arms):
                if arm.action_count == 0:
                    return i

        # Calculate UCB values with sliding window
        ucb_values = []

        for arm in self.arms:
            if arm.action_count == 0:
                ucb_values.append(float("inf"))
                continue

            # Use only recent rewards for non-stationarity
            recent_rewards = arm.reward_estimates[-self.window_size :]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0

            # Confidence interval based on recent actions
            recent_count = len(recent_rewards)
            if recent_count == 0:
                confidence = float("inf")
            else:
                confidence = self.exploration_param * np.sqrt(
                    np.log(self.total_actions) / recent_count
                )

            ucb_value = avg_reward + confidence
            ucb_values.append(ucb_value)

            # Store confidence bound for monitoring
            arm.confidence_bound = confidence

        return int(np.argmax(ucb_values))

    def _exp3_select(self) -> int:
        """
        EXP3 algorithm for adversarial bandits
        """
        # Update probabilities
        gamma = min(
            1.0,
            np.sqrt((np.log(self.n_arms)) / (self.n_arms * self.total_actions)),
        )

        self.probabilities = (1 - gamma) * (
            self.weights / np.sum(self.weights)
        ) + gamma / self.n_arms

        # Sample from probability distribution
        return np.random.choice(self.n_arms, p=self.probabilities)

    def _exp3_update(self, arm_id: int, reward: float):
        """Update EXP3 weights"""
        # Normalize reward to [0, 1]
        normalized_reward = max(
            0, min(1, (reward + 100) / 200)
        )  # Assuming rewards in [-100, 100]

        # Estimated reward
        estimated_reward = normalized_reward / self.probabilities[arm_id]

        # Update weight
        self.weights[arm_id] *= np.exp(
            self.learning_rate * estimated_reward / self.n_arms
        )

    def _thompson_sampling(self) -> int:
        """
        Thompson Sampling with Beta-Bernoulli model
        """
        # Sample from posterior distributions
        samples = []
        for i in range(self.n_arms):
            sample = np.random.beta(self.alpha_params[i], self.beta_params[i])
            samples.append(sample)

        return int(np.argmax(samples))

    def _thompson_update(self, arm_id: int, reward: float):
        """Update Thompson Sampling parameters"""
        # Convert reward to success/failure
        # Assuming rewards are in range that can be normalized to [0, 1]
        normalized_reward = max(0, min(1, (reward + 100) / 200))

        if normalized_reward > 0.5:  # Success
            self.alpha_params[arm_id] += 1
        else:  # Failure
            self.beta_params[arm_id] += 1

        # Apply discount for non-stationarity
        decay_factor = 0.99
        self.alpha_params[arm_id] *= decay_factor
        self.beta_params[arm_id] *= decay_factor

        # Ensure minimum values
        self.alpha_params[arm_id] = max(1.0, self.alpha_params[arm_id])
        self.beta_params[arm_id] = max(1.0, self.beta_params[arm_id])

    def _epsilon_greedy(self, epsilon: float = 0.1) -> int:
        """
        Epsilon-greedy with sliding window
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.n_arms)
        else:
            # Select arm with highest recent average
            recent_averages = []
            for arm in self.arms:
                recent_rewards = arm.reward_estimates[-self.window_size :]
                avg = np.mean(recent_rewards) if recent_rewards else 0.0
                recent_averages.append(avg)

            return int(np.argmax(recent_averages))

    def _handle_change_point(self):
        """
        Handle detected change point by adapting algorithm parameters
        """
        self.logger.info(f"Change point detected at action {self.total_actions}")

        # Reset or decay historical information
        if self.algorithm == "ucb_sliding":
            # Reduce effective window size temporarily
            self.window_size = max(10, self.window_size // 2)
        elif self.algorithm == "exp3":
            # Reset weights
            self.weights = np.ones(self.n_arms)
        elif self.algorithm == "thompson_sampling":
            # Decay prior parameters
            self.alpha_params *= 0.5
            self.beta_params *= 0.5
            self.alpha_params = np.maximum(1.0, self.alpha_params)
            self.beta_params = np.maximum(1.0, self.beta_params)

        # Increase exploration temporarily
        self.exploration_param *= 1.5

        # Gradually restore parameters
        if self.total_actions % 100 == 0:
            self.window_size = min(100, self.window_size * 1.1)
            self.exploration_param = max(2.0, self.exploration_param * 0.95)

    def _update_regret(self, selected_arm: int, reward: float):
        """
        Update regret calculation for performance monitoring
        """
        # Find optimal arm (highest recent average)
        recent_averages = []
        for arm in self.arms:
            recent_rewards = arm.reward_estimates[-50:]  # Last 50 for optimal
            avg = np.mean(recent_rewards) if recent_rewards else 0.0
            recent_averages.append(avg)

        optimal_arm = int(np.argmax(recent_averages))
        optimal_reward = recent_averages[optimal_arm]

        # Regret is difference between optimal and obtained reward
        instantaneous_regret = optimal_reward - reward
        self.regret_history.append(instantaneous_regret)
        self.optimal_arm_history.append(optimal_arm)

        # Limit history size
        if len(self.regret_history) > 1000:
            self.regret_history = self.regret_history[-500:]
            self.optimal_arm_history = self.optimal_arm_history[-500:]

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get comprehensive performance statistics
        """
        total_regret = sum(self.regret_history) if self.regret_history else 0.0

        stats = {
            "algorithm": self.algorithm,
            "total_actions": self.total_actions,
            "n_arms": self.n_arms,
            "performance": {
                "total_regret": total_regret,
                "average_regret": total_regret / max(1, len(self.regret_history)),
                "recent_regret": (
                    np.mean(self.regret_history[-100:])
                    if len(self.regret_history) >= 100
                    else 0.0
                ),
            },
            "exploration": {
                "current_exploration_param": self.exploration_param,
                "current_window_size": self.window_size,
            },
            "change_detection": {
                "change_points_detected": len(self.change_detector.change_points),
                "distance_to_last_change": self.change_detector.get_latest_change_distance(
                    self.total_actions
                ),
                "change_points": self.change_detector.change_points[-10:],  # Last 10
            },
            "arms": [arm.get_stats() for arm in self.arms],
        }

        # Add algorithm-specific stats
        if self.algorithm == "exp3":
            stats["exp3_params"] = {
                "weights": self.weights.tolist(),
                "probabilities": self.probabilities.tolist(),
            }
        elif self.algorithm == "thompson_sampling":
            stats["thompson_params"] = {
                "alpha_params": self.alpha_params.tolist(),
                "beta_params": self.beta_params.tolist(),
            }

        return stats

    def get_best_arm(self) -> int:
        """
        Get currently estimated best arm
        """
        recent_averages = []
        for arm in self.arms:
            recent_rewards = arm.reward_estimates[-self.window_size :]
            avg = np.mean(recent_rewards) if recent_rewards else 0.0
            recent_averages.append(avg)

        return int(np.argmax(recent_averages))

    def reset(self):
        """
        Reset bandit to initial state
        """
        self.arms = [BanditArm(i) for i in range(self.n_arms)]
        self.total_actions = 0
        self.regret_history.clear()
        self.optimal_arm_history.clear()

        # Reset algorithm-specific parameters
        self.alpha_params = np.ones(self.n_arms)
        self.beta_params = np.ones(self.n_arms)
        self.weights = np.ones(self.n_arms)
        self.probabilities = np.ones(self.n_arms) / self.n_arms

        # Reset change detection
        self.change_detector = ChangePointDetector()

        self.logger.info("Bandit reset to initial state")


# Export main classes
__all__ = [
    "NonStationaryBandit",
    "BanditArm",
    "ChangePointDetector",
]
