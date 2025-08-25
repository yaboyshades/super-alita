"""
Reward Tracker

Manages reward collection and feedback for multi-armed bandit optimization.
Provides mechanisms for tracking decision outcomes and learning from them.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from ..events import create_event


@dataclass
class RewardEvent:
    """Represents a reward event for a decision."""
    
    event_id: str
    decision_id: str
    reward_value: float
    reward_type: str  # "immediate", "delayed", "cumulative"
    source: str  # Where the reward came from
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class RewardRule:
    """Defines a rule for automatically calculating rewards."""
    
    rule_id: str
    name: str
    description: str
    condition: Callable[[Dict[str, Any]], bool]  # Function to check if rule applies
    calculator: Callable[[Dict[str, Any]], float]  # Function to calculate reward
    priority: int = 0  # Higher priority rules are evaluated first
    active: bool = True


class RewardTracker:
    """
    Tracks and manages rewards for bandit optimization decisions.
    
    Supports both immediate and delayed rewards, automatic reward calculation,
    and integration with the event system for real-time feedback.
    """
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.rewards: Dict[str, List[RewardEvent]] = {}  # decision_id -> rewards
        self.rules: Dict[str, RewardRule] = {}
        self.pending_rewards: Dict[str, List[RewardEvent]] = {}  # For delayed rewards
        self.callbacks: List[Callable[[RewardEvent], None]] = []
    
    def add_reward_rule(
        self,
        name: str,
        description: str,
        condition: Callable[[Dict[str, Any]], bool],
        calculator: Callable[[Dict[str, Any]], float],
        priority: int = 0,
        rule_id: Optional[str] = None
    ) -> str:
        """
        Add a rule for automatic reward calculation.
        
        Args:
            name: Human-readable name for the rule
            description: Description of what this rule measures
            condition: Function that takes context and returns True if rule applies
            calculator: Function that takes context and returns reward value (0.0-1.0)
            priority: Priority for rule evaluation (higher = evaluated first)
            rule_id: Optional custom rule ID
        
        Returns:
            The rule ID
        """
        if rule_id is None:
            rule_id = str(uuid4())
        
        rule = RewardRule(
            rule_id=rule_id,
            name=name,
            description=description,
            condition=condition,
            calculator=calculator,
            priority=priority
        )
        
        self.rules[rule_id] = rule
        return rule_id
    
    async def record_reward(
        self,
        decision_id: str,
        reward_value: float,
        reward_type: str = "immediate",
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a reward for a decision.
        
        Args:
            decision_id: ID of the decision being rewarded
            reward_value: Reward value (typically 0.0 to 1.0)
            reward_type: Type of reward ("immediate", "delayed", "cumulative")
            source: Source of the reward (e.g., "user", "system", "rule:rule_name")
            metadata: Additional reward metadata
        
        Returns:
            The reward event ID
        """
        event = RewardEvent(
            event_id=str(uuid4()),
            decision_id=decision_id,
            reward_value=reward_value,
            reward_type=reward_type,
            source=source,
            metadata=metadata or {}
        )
        
        # Store reward
        if decision_id not in self.rewards:
            self.rewards[decision_id] = []
        self.rewards[decision_id].append(event)
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                # Log error but don't fail the reward recording
                print(f"Error in reward callback: {e}")
        
        # Emit event if event bus is available
        if self.event_bus:
            try:
                reward_event = create_event(
                    "reward_recorded",
                    source_plugin="RewardTracker",
                    decision_id=decision_id,
                    reward_value=reward_value,
                    reward_type=reward_type,
                    source=source,
                    metadata=metadata
                )
                await self.event_bus.emit_event(reward_event)
            except Exception as e:
                print(f"Error emitting reward event: {e}")
        
        return event.event_id
    
    async def calculate_automatic_rewards(
        self,
        decision_id: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Calculate rewards automatically using registered rules.
        
        Args:
            decision_id: ID of the decision to evaluate
            context: Context information for rule evaluation
        
        Returns:
            List of reward event IDs that were created
        """
        reward_ids = []
        
        # Sort rules by priority (highest first)
        sorted_rules = sorted(
            self.rules.values(),
            key=lambda r: r.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            if not rule.active:
                continue
            
            try:
                # Check if rule condition is met
                if rule.condition(context):
                    # Calculate reward
                    reward_value = rule.calculator(context)
                    
                    # Clamp reward to valid range
                    reward_value = max(0.0, min(1.0, reward_value))
                    
                    # Record reward
                    reward_id = await self.record_reward(
                        decision_id=decision_id,
                        reward_value=reward_value,
                        reward_type="immediate",
                        source=f"rule:{rule.name}",
                        metadata={
                            "rule_id": rule.rule_id,
                            "rule_name": rule.name,
                            "auto_calculated": True
                        }
                    )
                    
                    reward_ids.append(reward_id)
                    
            except Exception as e:
                print(f"Error in reward rule '{rule.name}': {e}")
                continue
        
        return reward_ids
    
    def get_decision_rewards(self, decision_id: str) -> List[RewardEvent]:
        """Get all rewards for a specific decision."""
        return self.rewards.get(decision_id, [])
    
    def get_total_reward(self, decision_id: str) -> float:
        """Get the total reward value for a decision."""
        rewards = self.get_decision_rewards(decision_id)
        return sum(r.reward_value for r in rewards)
    
    def get_latest_reward(self, decision_id: str) -> Optional[RewardEvent]:
        """Get the most recent reward for a decision."""
        rewards = self.get_decision_rewards(decision_id)
        if not rewards:
            return None
        return max(rewards, key=lambda r: r.timestamp)
    
    def add_callback(self, callback: Callable[[RewardEvent], None]) -> None:
        """Add a callback to be notified of new rewards."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[RewardEvent], None]) -> bool:
        """Remove a reward callback."""
        try:
            self.callbacks.remove(callback)
            return True
        except ValueError:
            return False
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about reward rules."""
        active_rules = [r for r in self.rules.values() if r.active]
        inactive_rules = [r for r in self.rules.values() if not r.active]
        
        # Count rewards generated by each rule
        rule_rewards = {}
        for rewards_list in self.rewards.values():
            for reward in rewards_list:
                if reward.source.startswith("rule:"):
                    rule_name = reward.source[5:]  # Remove "rule:" prefix
                    rule_rewards[rule_name] = rule_rewards.get(rule_name, 0) + 1
        
        return {
            "total_rules": len(self.rules),
            "active_rules": len(active_rules),
            "inactive_rules": len(inactive_rules),
            "rule_usage": rule_rewards,
            "rules": [
                {
                    "id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "priority": rule.priority,
                    "active": rule.active,
                    "rewards_generated": rule_rewards.get(rule.name, 0)
                }
                for rule in self.rules.values()
            ]
        }
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global reward statistics."""
        total_decisions = len(self.rewards)
        total_rewards = sum(len(rewards) for rewards in self.rewards.values())
        
        # Calculate average rewards per decision
        avg_rewards_per_decision = total_rewards / total_decisions if total_decisions > 0 else 0
        
        # Count rewards by type and source
        reward_types = {}
        reward_sources = {}
        total_value = 0.0
        
        for rewards_list in self.rewards.values():
            for reward in rewards_list:
                reward_types[reward.reward_type] = reward_types.get(reward.reward_type, 0) + 1
                reward_sources[reward.source] = reward_sources.get(reward.source, 0) + 1
                total_value += reward.reward_value
        
        avg_reward_value = total_value / total_rewards if total_rewards > 0 else 0
        
        return {
            "total_decisions_with_rewards": total_decisions,
            "total_rewards": total_rewards,
            "average_rewards_per_decision": avg_rewards_per_decision,
            "average_reward_value": avg_reward_value,
            "total_reward_value": total_value,
            "reward_types": reward_types,
            "reward_sources": reward_sources
        }


# Example reward rules for common scenarios
def create_success_rate_rule() -> RewardRule:
    """Create a rule that rewards based on success indicators in context."""
    
    def condition(context: Dict[str, Any]) -> bool:
        return "success" in context or "error" in context
    
    def calculator(context: Dict[str, Any]) -> float:
        if context.get("success", False):
            return 1.0
        elif context.get("error", False):
            return 0.0
        else:
            return 0.5
    
    return RewardRule(
        rule_id="success_rate",
        name="Success Rate",
        description="Rewards decisions based on success/error indicators",
        condition=condition,
        calculator=calculator,
        priority=100
    )


def create_performance_rule() -> RewardRule:
    """Create a rule that rewards based on performance metrics."""
    
    def condition(context: Dict[str, Any]) -> bool:
        return "execution_time" in context or "performance_score" in context
    
    def calculator(context: Dict[str, Any]) -> float:
        # Reward faster execution times
        execution_time = context.get("execution_time", 1.0)
        if execution_time < 0.1:
            time_reward = 1.0
        elif execution_time < 0.5:
            time_reward = 0.8
        elif execution_time < 1.0:
            time_reward = 0.6
        else:
            time_reward = 0.3
        
        # Reward higher performance scores
        performance_score = context.get("performance_score", 0.5)
        
        # Combine both factors
        return (time_reward + performance_score) / 2.0
    
    return RewardRule(
        rule_id="performance",
        name="Performance",
        description="Rewards decisions based on execution time and performance metrics",
        condition=condition,
        calculator=calculator,
        priority=80
    )