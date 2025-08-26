"""
Decision Policy Engine

Manages decision-making policies using multi-armed bandit algorithms.
Provides a high-level interface for intelligent decision optimization.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from .bandits import BanditAlgorithm, BanditDecision, ThompsonSamplingBandit, UCB1Bandit, EpsilonGreedyBandit


@dataclass
class DecisionContext:
    """Context information for making decisions."""
    
    session_id: str
    user_id: Optional[str] = None
    workspace: Optional[str] = None
    task_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PolicyDefinition:
    """Defines a decision policy with its arms and algorithm."""
    
    policy_id: str
    name: str
    description: str
    algorithm_type: str  # "thompson", "ucb1", "epsilon_greedy"
    arms: List[Dict[str, Any]]  # List of arm definitions
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass 
class PolicyDecision:
    """Represents a decision made by a policy."""
    
    decision_id: str
    policy_id: str
    bandit_decision: BanditDecision
    context: DecisionContext
    reward: Optional[float] = None
    feedback_received: bool = False
    created_at: float = field(default_factory=time.time)


class DecisionPolicyEngine:
    """
    Engine for managing and executing decision policies using bandit algorithms.
    
    Supports multiple policies, each with their own bandit algorithm and arms.
    Provides learning through reward feedback and performance tracking.
    """
    
    def __init__(self):
        self.policies: Dict[str, PolicyDefinition] = {}
        self.bandits: Dict[str, BanditAlgorithm] = {}
        self.decisions: Dict[str, PolicyDecision] = {}
        self.active_sessions: Set[str] = set()
    
    def create_policy(
        self,
        name: str,
        description: str,
        algorithm_type: str,
        arms: List[Dict[str, Any]],
        policy_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Create a new decision policy.
        
        Args:
            name: Human-readable name for the policy
            description: Description of what this policy decides
            algorithm_type: Type of bandit algorithm ("thompson", "ucb1", "epsilon_greedy")
            arms: List of arm definitions, each with "id", "name", and optional "metadata"
            policy_id: Optional custom policy ID
            **kwargs: Additional metadata
        
        Returns:
            The policy ID
        """
        if policy_id is None:
            policy_id = str(uuid4())
        
        # Validate algorithm type
        valid_algorithms = {"thompson", "ucb1", "epsilon_greedy"}
        if algorithm_type not in valid_algorithms:
            raise ValueError(f"Invalid algorithm type: {algorithm_type}. Must be one of {valid_algorithms}")
        
        # Validate arms
        if not arms:
            raise ValueError("At least one arm must be provided")
        
        for arm in arms:
            if "id" not in arm or "name" not in arm:
                raise ValueError("Each arm must have 'id' and 'name' fields")
        
        # Create policy definition
        policy = PolicyDefinition(
            policy_id=policy_id,
            name=name,
            description=description,
            algorithm_type=algorithm_type,
            arms=arms,
            metadata=kwargs
        )
        
        # Create bandit algorithm
        bandit = self._create_bandit(algorithm_type, **kwargs)
        
        # Add arms to bandit
        for arm_def in arms:
            bandit.add_arm(
                arm_id=arm_def["id"],
                name=arm_def["name"],
                metadata=arm_def.get("metadata", {})
            )
        
        # Store policy and bandit
        self.policies[policy_id] = policy
        self.bandits[policy_id] = bandit
        
        return policy_id
    
    def _create_bandit(self, algorithm_type: str, **kwargs) -> BanditAlgorithm:
        """Create a bandit algorithm instance."""
        if algorithm_type == "thompson":
            return ThompsonSamplingBandit()
        elif algorithm_type == "ucb1":
            return UCB1Bandit()
        elif algorithm_type == "epsilon_greedy":
            epsilon = kwargs.get("epsilon", 0.1)
            return EpsilonGreedyBandit(epsilon=epsilon)
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    async def make_decision(
        self,
        policy_id: str,
        context: DecisionContext
    ) -> PolicyDecision:
        """
        Make a decision using the specified policy.
        
        Args:
            policy_id: ID of the policy to use
            context: Decision context information
        
        Returns:
            The policy decision
        """
        if policy_id not in self.policies:
            raise ValueError(f"Policy not found: {policy_id}")
        
        bandit = self.bandits[policy_id]
        
        # Make bandit decision
        bandit_decision = bandit.select_arm(context=context.metadata)
        
        # Create policy decision
        decision = PolicyDecision(
            decision_id=str(uuid4()),
            policy_id=policy_id,
            bandit_decision=bandit_decision,
            context=context
        )
        
        # Store decision
        self.decisions[decision.decision_id] = decision
        self.active_sessions.add(context.session_id)
        
        return decision
    
    async def provide_feedback(
        self,
        decision_id: str,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Provide reward feedback for a previous decision.
        
        Args:
            decision_id: ID of the decision to provide feedback for
            reward: Reward value (typically 0.0 to 1.0)
            metadata: Optional additional feedback metadata
        
        Returns:
            True if feedback was successfully applied
        """
        if decision_id not in self.decisions:
            return False
        
        decision = self.decisions[decision_id]
        policy_id = decision.policy_id
        
        if policy_id not in self.bandits:
            return False
        
        bandit = self.bandits[policy_id]
        
        # Update bandit with reward
        success = bandit.update_reward(decision.bandit_decision.decision_id, reward)
        
        if success:
            decision.reward = reward
            decision.feedback_received = True
            
            # Add metadata to decision context if provided
            if metadata:
                decision.context.metadata.update(metadata)
        
        return success
    
    def get_policy_statistics(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific policy."""
        if policy_id not in self.policies or policy_id not in self.bandits:
            return None
        
        policy = self.policies[policy_id]
        bandit = self.bandits[policy_id]
        
        bandit_stats = bandit.get_statistics()
        
        # Count decisions and feedback
        policy_decisions = [d for d in self.decisions.values() if d.policy_id == policy_id]
        decisions_with_feedback = [d for d in policy_decisions if d.feedback_received]
        
        return {
            "policy": {
                "id": policy.policy_id,
                "name": policy.name,
                "description": policy.description,
                "algorithm_type": policy.algorithm_type,
                "created_at": policy.created_at,
                "updated_at": policy.updated_at
            },
            "bandit": bandit_stats,
            "decisions": {
                "total": len(policy_decisions),
                "with_feedback": len(decisions_with_feedback),
                "feedback_rate": len(decisions_with_feedback) / len(policy_decisions) if policy_decisions else 0.0
            }
        }
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all policies."""
        return {
            "engine": {
                "total_policies": len(self.policies),
                "total_decisions": len(self.decisions),
                "active_sessions": len(self.active_sessions)
            },
            "policies": {
                policy_id: self.get_policy_statistics(policy_id)
                for policy_id in self.policies.keys()
            }
        }
    
    def list_policies(self) -> List[Dict[str, Any]]:
        """List all policies with basic information."""
        return [
            {
                "id": policy.policy_id,
                "name": policy.name,
                "description": policy.description,
                "algorithm_type": policy.algorithm_type,
                "arms_count": len(policy.arms),
                "created_at": policy.created_at
            }
            for policy in self.policies.values()
        ]
    
    def get_policy(self, policy_id: str) -> Optional[PolicyDefinition]:
        """Get a policy definition by ID."""
        return self.policies.get(policy_id)
    
    def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy and all its associated data."""
        if policy_id not in self.policies:
            return False
        
        # Remove policy and bandit
        del self.policies[policy_id]
        del self.bandits[policy_id]
        
        # Remove associated decisions
        decisions_to_remove = [
            decision_id for decision_id, decision in self.decisions.items()
            if decision.policy_id == policy_id
        ]
        
        for decision_id in decisions_to_remove:
            del self.decisions[decision_id]
        
        return True
    
    async def optimize_all(self) -> Dict[str, Any]:
        """
        Run optimization across all policies.
        
        This can be used for periodic optimization tasks like:
        - Adjusting epsilon values for epsilon-greedy algorithms
        - Detecting and handling concept drift
        - Rebalancing arm priors
        
        Returns:
            Optimization results and recommendations
        """
        results = {
            "optimized_policies": [],
            "recommendations": [],
            "timestamp": time.time()
        }
        
        for policy_id, policy in self.policies.items():
            stats = self.get_policy_statistics(policy_id)
            if stats is None:
                continue
            
            # Example optimization: suggest epsilon adjustment for epsilon-greedy
            if policy.algorithm_type == "epsilon_greedy":
                feedback_rate = stats["decisions"]["feedback_rate"]
                total_decisions = stats["decisions"]["total"]
                
                if total_decisions > 100 and feedback_rate > 0.8:
                    # High feedback rate suggests we can reduce exploration
                    current_epsilon = self.bandits[policy_id].epsilon
                    if current_epsilon > 0.05:
                        results["recommendations"].append({
                            "policy_id": policy_id,
                            "type": "reduce_epsilon",
                            "current": current_epsilon,
                            "suggested": max(0.05, current_epsilon * 0.9),
                            "reason": "High feedback rate suggests effective exploitation"
                        })
            
            results["optimized_policies"].append(policy_id)
        
        return results