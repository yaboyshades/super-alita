"""
Optimization Plugin

Provides multi-armed bandit optimization capabilities as a plugin for Super Alita.
Integrates decision policies, reward tracking, and learning with the event system.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from ..plugin_interface import PluginInterface
from ..events import create_event
from .policy_engine import DecisionPolicyEngine, DecisionContext, PolicyDecision
from .reward_tracker import RewardTracker, create_success_rate_rule, create_performance_rule


class OptimizationPlugin(PluginInterface):
    """
    Plugin that provides intelligent decision optimization using multi-armed bandit algorithms.
    
    Features:
    - Multiple bandit algorithms (Thompson Sampling, UCB1, Epsilon-Greedy)
    - Decision policy management
    - Automatic and manual reward tracking
    - Performance analytics and optimization
    - Event-driven integration
    """
    
    def __init__(self):
        super().__init__()
        self.policy_engine = DecisionPolicyEngine()
        self.reward_tracker = RewardTracker()
        self.event_bus = None
        self._optimization_task: Optional[asyncio.Task] = None
        self._metrics = {
            "decisions_made": 0,
            "rewards_collected": 0,
            "policies_created": 0,
            "average_reward": 0.0
        }
    
    @property
    def name(self) -> str:
        return "OptimizationPlugin"
    
    async def setup(self, event_bus=None, **kwargs) -> None:
        """Setup the optimization plugin."""
        self.event_bus = event_bus
        self.reward_tracker.event_bus = event_bus
        
        # Add default reward rules
        self.reward_tracker.rules["success_rate"] = create_success_rate_rule()
        self.reward_tracker.rules["performance"] = create_performance_rule()
        
        # Add reward callback to update metrics
        self.reward_tracker.add_callback(self._on_reward_received)
        
        print(f"🎯 {self.name} initialized with {len(self.reward_tracker.rules)} reward rules")
    
    async def start(self) -> None:
        """Start the optimization plugin."""
        self._is_running = True
        
        # Subscribe to relevant events if event bus is available
        if self.event_bus:
            await self.event_bus.subscribe("cortex_cycle_complete", self._handle_cortex_cycle)
            await self.event_bus.subscribe("decision_needed", self._handle_decision_request)
            await self.event_bus.subscribe("task_completed", self._handle_task_completion)
        
        # Start periodic optimization task
        self._optimization_task = self.add_task(self._periodic_optimization())
    
    async def shutdown(self) -> None:
        """Shutdown the optimization plugin."""
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        if self.event_bus:
            await self.event_bus.unsubscribe("cortex_cycle_complete", self._handle_cortex_cycle)
            await self.event_bus.unsubscribe("decision_needed", self._handle_decision_request)
            await self.event_bus.unsubscribe("task_completed", self._handle_task_completion)
        
        print(f"🎯 {self.name} shutdown complete")
    
    async def _handle_cortex_cycle(self, event) -> None:
        """Handle Cortex cycle completion events for learning."""
        try:
            cycle_data = event.data
            
            # Extract decision context if available
            if "decisions" in cycle_data:
                for decision_data in cycle_data["decisions"]:
                    decision_id = decision_data.get("decision_id")
                    if decision_id:
                        # Calculate automatic rewards based on cycle performance
                        context = {
                            "success": cycle_data.get("success", False),
                            "execution_time": cycle_data.get("execution_time", 1.0),
                            "performance_score": cycle_data.get("performance_score", 0.5),
                            "cycle_id": cycle_data.get("cycle_id"),
                            "cycle_type": "cortex"
                        }
                        
                        await self.reward_tracker.calculate_automatic_rewards(decision_id, context)
            
        except Exception as e:
            print(f"Error handling Cortex cycle event: {e}")
    
    async def _handle_decision_request(self, event) -> None:
        """Handle decision request events."""
        try:
            request_data = event.data
            policy_id = request_data.get("policy_id")
            
            if policy_id and policy_id in self.policy_engine.policies:
                # Create decision context
                context = DecisionContext(
                    session_id=request_data.get("session_id", "unknown"),
                    user_id=request_data.get("user_id"),
                    workspace=request_data.get("workspace"),
                    task_type=request_data.get("task_type"),
                    metadata=request_data.get("metadata", {})
                )
                
                # Make decision
                decision = await self.policy_engine.make_decision(policy_id, context)
                self._metrics["decisions_made"] += 1
                
                # Emit decision made event
                if self.event_bus:
                    decision_event = create_event(
                        "decision_made",
                        source_plugin=self.name,
                        decision_id=decision.decision_id,
                        policy_id=policy_id,
                        arm_id=decision.bandit_decision.arm_id,
                        arm_name=decision.bandit_decision.arm_name,
                        algorithm=decision.bandit_decision.algorithm,
                        confidence=decision.bandit_decision.confidence,
                        context=decision.context.__dict__
                    )
                    await self.event_bus.emit_event(decision_event)
        
        except Exception as e:
            print(f"Error handling decision request: {e}")
    
    async def _handle_task_completion(self, event) -> None:
        """Handle task completion events for reward calculation."""
        try:
            task_data = event.data
            decision_id = task_data.get("decision_id")
            
            if decision_id:
                # Calculate reward based on task outcome
                reward_value = 0.5  # Default neutral reward
                
                if task_data.get("success", False):
                    reward_value = 0.8
                elif task_data.get("error", False):
                    reward_value = 0.2
                
                # Adjust based on performance metrics
                if "performance_score" in task_data:
                    performance = task_data["performance_score"]
                    reward_value = (reward_value + performance) / 2.0
                
                # Record reward
                await self.reward_tracker.record_reward(
                    decision_id=decision_id,
                    reward_value=reward_value,
                    reward_type="immediate",
                    source="task_completion",
                    metadata=task_data
                )
                
                # Provide feedback to policy engine
                await self.policy_engine.provide_feedback(decision_id, reward_value)
        
        except Exception as e:
            print(f"Error handling task completion: {e}")
    
    def _on_reward_received(self, reward_event) -> None:
        """Callback for when rewards are received."""
        self._metrics["rewards_collected"] += 1
        
        # Update average reward (simple moving average)
        if self._metrics["rewards_collected"] > 0:
            current_avg = self._metrics["average_reward"]
            new_avg = (current_avg * (self._metrics["rewards_collected"] - 1) + reward_event.reward_value) / self._metrics["rewards_collected"]
            self._metrics["average_reward"] = new_avg
    
    async def _periodic_optimization(self) -> None:
        """Periodic optimization task to improve policy performance."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Run optimization
                results = await self.policy_engine.optimize_all()
                
                # Apply recommendations if any
                if results["recommendations"]:
                    print(f"🎯 Optimization recommendations: {len(results['recommendations'])}")
                    
                    # Could automatically apply some recommendations here
                    for rec in results["recommendations"]:
                        if rec["type"] == "reduce_epsilon" and self.event_bus:
                            # Emit recommendation event for manual review
                            rec_event = create_event(
                                "optimization_recommendation",
                                source_plugin=self.name,
                                recommendation_type=rec["type"],
                                policy_id=rec["policy_id"],
                                current_value=rec["current"],
                                suggested_value=rec["suggested"],
                                reason=rec["reason"]
                            )
                            await self.event_bus.emit_event(rec_event)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in periodic optimization: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    # Public API methods
    
    async def create_policy(
        self,
        name: str,
        description: str,
        algorithm_type: str,
        arms: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Create a new decision policy."""
        policy_id = self.policy_engine.create_policy(
            name=name,
            description=description,
            algorithm_type=algorithm_type,
            arms=arms,
            **kwargs
        )
        
        self._metrics["policies_created"] += 1
        
        # Emit policy created event
        if self.event_bus:
            policy_event = create_event(
                "policy_created",
                source_plugin=self.name,
                policy_id=policy_id,
                name=name,
                algorithm_type=algorithm_type,
                arms_count=len(arms)
            )
            await self.event_bus.emit_event(policy_event)
        
        return policy_id
    
    async def make_decision(
        self,
        policy_id: str,
        session_id: str,
        user_id: Optional[str] = None,
        workspace: Optional[str] = None,
        task_type: Optional[str] = None,
        **metadata
    ) -> PolicyDecision:
        """Make a decision using a specific policy."""
        context = DecisionContext(
            session_id=session_id,
            user_id=user_id,
            workspace=workspace,
            task_type=task_type,
            metadata=metadata
        )
        
        decision = await self.policy_engine.make_decision(policy_id, context)
        self._metrics["decisions_made"] += 1
        
        return decision
    
    async def provide_feedback(
        self,
        decision_id: str,
        reward: float,
        source: str = "manual",
        **metadata
    ) -> bool:
        """Provide feedback for a decision."""
        # Record in reward tracker
        await self.reward_tracker.record_reward(
            decision_id=decision_id,
            reward_value=reward,
            source=source,
            metadata=metadata
        )
        
        # Provide to policy engine
        return await self.policy_engine.provide_feedback(decision_id, reward, metadata)
    
    def get_policies(self) -> List[Dict[str, Any]]:
        """Get all policies."""
        return self.policy_engine.list_policies()
    
    def get_policy_statistics(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific policy."""
        return self.policy_engine.get_policy_statistics(policy_id)
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global optimization statistics."""
        policy_stats = self.policy_engine.get_all_statistics()
        reward_stats = self.reward_tracker.get_global_statistics()
        rule_stats = self.reward_tracker.get_rule_statistics()
        
        return {
            "plugin_metrics": self._metrics,
            "policies": policy_stats,
            "rewards": reward_stats,
            "rules": rule_stats,
            "timestamp": time.time()
        }
    
    async def export_data(self) -> Dict[str, Any]:
        """Export all optimization data for analysis."""
        return {
            "policies": {
                policy_id: {
                    "definition": policy.__dict__,
                    "statistics": self.get_policy_statistics(policy_id)
                }
                for policy_id, policy in self.policy_engine.policies.items()
            },
            "decisions": {
                decision_id: decision.__dict__
                for decision_id, decision in self.policy_engine.decisions.items()
            },
            "rewards": {
                decision_id: [reward.__dict__ for reward in rewards]
                for decision_id, rewards in self.reward_tracker.rewards.items()
            },
            "rules": {
                rule_id: {
                    "id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "priority": rule.priority,
                    "active": rule.active
                }
                for rule_id, rule in self.reward_tracker.rules.items()
            },
            "metrics": self._metrics,
            "exported_at": time.time()
        }