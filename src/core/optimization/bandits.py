"""
Multi-Armed Bandit Algorithms

Implements various bandit algorithms for decision optimization:
- Thompson Sampling: Bayesian approach using Beta distributions
- UCB1: Upper Confidence Bound with square root exploration bonus
- Epsilon-Greedy: Simple exploration with epsilon probability
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from scipy.stats import beta


@dataclass
class BanditArm:
    """Represents a single arm in a multi-armed bandit."""
    
    arm_id: str
    name: str
    metadata: Dict[str, Any]
    successes: int = 0
    failures: int = 0
    total_pulls: int = 0
    last_reward: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of this arm."""
        if self.total_pulls == 0:
            return 0.0
        return self.successes / self.total_pulls
    
    def update(self, reward: float) -> None:
        """Update arm statistics with a new reward."""
        self.total_pulls += 1
        self.last_reward = reward
        
        if reward > 0.5:  # Consider > 0.5 as success
            self.successes += 1
        else:
            self.failures += 1


@dataclass
class BanditDecision:
    """Represents a decision made by a bandit algorithm."""
    
    decision_id: str
    arm_id: str
    arm_name: str
    algorithm: str
    confidence: float
    context: Dict[str, Any]
    timestamp: float
    
    @classmethod
    def create(
        cls,
        arm: BanditArm,
        algorithm: str,
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> "BanditDecision":
        """Create a new bandit decision."""
        import time
        
        return cls(
            decision_id=str(uuid4()),
            arm_id=arm.arm_id,
            arm_name=arm.name,
            algorithm=algorithm,
            confidence=confidence,
            context=context or {},
            timestamp=time.time()
        )


class BanditAlgorithm(ABC):
    """Abstract base class for bandit algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.arms: Dict[str, BanditArm] = {}
        self.decision_history: List[BanditDecision] = []
    
    def add_arm(self, arm_id: str, name: str, metadata: Optional[Dict[str, Any]] = None) -> BanditArm:
        """Add a new arm to the bandit."""
        arm = BanditArm(
            arm_id=arm_id,
            name=name,
            metadata=metadata or {}
        )
        self.arms[arm_id] = arm
        return arm
    
    def update_reward(self, decision_id: str, reward: float) -> bool:
        """Update the reward for a previous decision."""
        # Find the decision
        decision = None
        for d in self.decision_history:
            if d.decision_id == decision_id:
                decision = d
                break
        
        if decision is None:
            return False
        
        # Update the arm
        arm = self.arms.get(decision.arm_id)
        if arm is None:
            return False
        
        arm.update(reward)
        return True
    
    @abstractmethod
    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> BanditDecision:
        """Select an arm based on the algorithm's strategy."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the bandit's performance."""
        total_pulls = sum(arm.total_pulls for arm in self.arms.values())
        
        return {
            "algorithm": self.name,
            "total_arms": len(self.arms),
            "total_pulls": total_pulls,
            "total_decisions": len(self.decision_history),
            "arms": {
                arm_id: {
                    "name": arm.name,
                    "pulls": arm.total_pulls,
                    "successes": arm.successes,
                    "failures": arm.failures,
                    "success_rate": arm.success_rate,
                    "last_reward": arm.last_reward
                }
                for arm_id, arm in self.arms.items()
            }
        }


class ThompsonSamplingBandit(BanditAlgorithm):
    """
    Thompson Sampling bandit using Beta distributions.
    
    Each arm is modeled as a Beta distribution with parameters (successes + 1, failures + 1).
    At each step, we sample from each arm's distribution and select the highest sample.
    """
    
    def __init__(self):
        super().__init__("Thompson Sampling")
    
    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> BanditDecision:
        """Select arm using Thompson Sampling."""
        if not self.arms:
            raise ValueError("No arms available for selection")
        
        best_arm = None
        best_sample = -1.0
        arm_samples = {}
        
        # Sample from each arm's Beta distribution
        for arm in self.arms.values():
            # Beta distribution parameters (add 1 to avoid zeros)
            alpha = arm.successes + 1
            beta_param = arm.failures + 1
            
            # Sample from Beta distribution
            sample = beta.rvs(alpha, beta_param)
            arm_samples[arm.arm_id] = sample
            
            if sample > best_sample:
                best_sample = sample
                best_arm = arm
        
        if best_arm is None:
            # Fallback to random selection
            best_arm = random.choice(list(self.arms.values()))
            best_sample = 0.5
        
        # Calculate confidence based on the margin between best and second-best
        sorted_samples = sorted(arm_samples.values(), reverse=True)
        confidence = best_sample
        if len(sorted_samples) > 1:
            margin = sorted_samples[0] - sorted_samples[1]
            confidence = min(1.0, 0.5 + margin)
        
        decision = BanditDecision.create(
            arm=best_arm,
            algorithm=self.name,
            confidence=confidence,
            context=context
        )
        
        self.decision_history.append(decision)
        return decision


class UCB1Bandit(BanditAlgorithm):
    """
    Upper Confidence Bound (UCB1) bandit algorithm.
    
    Selects the arm with the highest upper confidence bound:
    UCB1(i) = average_reward(i) + sqrt(2 * ln(total_pulls) / pulls(i))
    """
    
    def __init__(self):
        super().__init__("UCB1")
    
    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> BanditDecision:
        """Select arm using UCB1 algorithm."""
        if not self.arms:
            raise ValueError("No arms available for selection")
        
        total_pulls = sum(arm.total_pulls for arm in self.arms.values())
        
        # If any arm hasn't been pulled, select it first
        for arm in self.arms.values():
            if arm.total_pulls == 0:
                decision = BanditDecision.create(
                    arm=arm,
                    algorithm=self.name,
                    confidence=1.0,  # High confidence for exploration
                    context=context
                )
                self.decision_history.append(decision)
                return decision
        
        best_arm = None
        best_ucb = -float('inf')
        arm_ucbs = {}
        
        # Calculate UCB1 value for each arm
        for arm in self.arms.values():
            if arm.total_pulls == 0:
                ucb_value = float('inf')
            else:
                confidence_radius = math.sqrt(2 * math.log(total_pulls) / arm.total_pulls)
                ucb_value = arm.success_rate + confidence_radius
            
            arm_ucbs[arm.arm_id] = ucb_value
            
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_arm = arm
        
        if best_arm is None:
            best_arm = random.choice(list(self.arms.values()))
            best_ucb = 0.5
        
        # Calculate confidence based on UCB margin
        sorted_ucbs = sorted(arm_ucbs.values(), reverse=True)
        confidence = min(1.0, best_ucb)
        if len(sorted_ucbs) > 1 and sorted_ucbs[0] != float('inf'):
            margin = sorted_ucbs[0] - sorted_ucbs[1]
            confidence = min(1.0, 0.5 + margin / 2)
        
        decision = BanditDecision.create(
            arm=best_arm,
            algorithm=self.name,
            confidence=confidence,
            context=context
        )
        
        self.decision_history.append(decision)
        return decision


class EpsilonGreedyBandit(BanditAlgorithm):
    """
    Epsilon-Greedy bandit algorithm.
    
    With probability epsilon, explores by selecting a random arm.
    With probability (1-epsilon), exploits by selecting the arm with highest success rate.
    """
    
    def __init__(self, epsilon: float = 0.1):
        super().__init__("Epsilon-Greedy")
        self.epsilon = epsilon
    
    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> BanditDecision:
        """Select arm using Epsilon-Greedy strategy."""
        if not self.arms:
            raise ValueError("No arms available for selection")
        
        # Epsilon-greedy decision
        if random.random() < self.epsilon:
            # Explore: random selection
            selected_arm = random.choice(list(self.arms.values()))
            confidence = 1.0 - self.epsilon  # Lower confidence for exploration
            exploration = True
        else:
            # Exploit: select best arm
            best_arm = None
            best_rate = -1.0
            
            for arm in self.arms.values():
                if arm.success_rate > best_rate:
                    best_rate = arm.success_rate
                    best_arm = arm
            
            selected_arm = best_arm or random.choice(list(self.arms.values()))
            confidence = 1.0 - self.epsilon + (self.epsilon * best_rate)  # Higher confidence for exploitation
            exploration = False
        
        decision = BanditDecision.create(
            arm=selected_arm,
            algorithm=self.name,
            confidence=confidence,
            context={**(context or {}), "exploration": exploration, "epsilon": self.epsilon}
        )
        
        self.decision_history.append(decision)
        return decision