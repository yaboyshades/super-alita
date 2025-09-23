"""
MoE Routing and Expert Gating System

Implements Mixture-of-Experts routing with attention-based gating,
budget constraints, and exploration policies for E-UPUSF orchestration.
"""

import math
import random
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExplorationPolicy(Enum):
    """Exploration policies for expert selection"""
    UCB1 = "ucb1"
    THOMPSON = "thompson"
    EPSILON_GREEDY = "epsilon_greedy"


@dataclass
class ExpertScore:
    """Scoring result for an expert"""
    expert_id: str
    fit_score: float
    utility_score: float
    cost_score: float
    risk_score: float
    total_score: float
    confidence: float = 1.0


@dataclass
class ExpertSelection:
    """Selected expert for execution"""
    expert_id: str
    score: ExpertScore
    inputs: List[str]
    outputs: List[str]
    estimated_cost: Dict[str, float]
    estimated_risk: Dict[str, float]


@dataclass
class RoutingDecision:
    """Complete routing decision with selected experts"""
    primary_expert: ExpertSelection
    backup_experts: List[ExpertSelection] = field(default_factory=list)
    routing_strategy: str = "single"
    total_estimated_cost: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    reasoning: List[str] = field(default_factory=list)


class ExpertGating:
    """Gating mechanism for expert selection with budget and
    risk constraints"""
    
    def __init__(self, routing_config: Dict[str, Any]):
        self.config = routing_config
        self.scoring_config = routing_config.get("scoring", {})
        self.weights = self.scoring_config.get("weights", {
            "fit": 0.4,
            "expected_utility": 0.3,
            "cost": 0.2,
            "risk": 0.1
        })
        self.top_k = self.scoring_config.get("top_k", 3)
        
        # Exploration policy
        exploration_config = self.scoring_config.get("exploration_policy", {})
        self.exploration_policy = ExplorationPolicy(
            exploration_config.get("type", "ucb1")
        )
        self.exploration_c = exploration_config.get("c", 1.2)
        self.exploration_epsilon = exploration_config.get("epsilon", 0.05)
        
        # Budget constraints
        budget_config = routing_config.get("budgets", {})
        self.reserve_fraction = budget_config.get("reserve_fraction", 0.2)
        
        # Gating rules
        self.gating_rules = routing_config.get("gating_rules", [])
        
        # Expert usage history for exploration
        self.expert_usage_history: Dict[str, Dict[str, Any]] = {}
    
    def score_expert(self, expert: Dict[str, Any],
                     current_state: str,
                     current_method: str,
                     context: Dict[str, Any]) -> ExpertScore:
        """Score an expert for current context"""
        
        expert_id = expert["id"]
        
        # Calculate fit score
        fit_score = self._calculate_fit_score(
            expert, current_state, current_method
        )
        
        # Calculate expected utility based on quality prior and history
        utility_score = self._calculate_utility_score(expert_id, expert, context)
        
        # Calculate cost score (lower cost = higher score)
        cost_score = self._calculate_cost_score(expert, context)
        
        # Calculate risk score (lower risk = higher score)
        risk_score = self._calculate_risk_score(expert, context)
        
        # Weighted total score
        total_score = (
            self.weights["fit"] * fit_score +
            self.weights["expected_utility"] * utility_score +
            self.weights["cost"] * cost_score +
            self.weights["risk"] * risk_score
        )
        
        # Apply exploration bonus if using UCB1
        if self.exploration_policy == ExplorationPolicy.UCB1:
            exploration_bonus = self._calculate_ucb1_bonus(expert_id)
            total_score += exploration_bonus
        
        return ExpertScore(
            expert_id=expert_id,
            fit_score=fit_score,
            utility_score=utility_score,
            cost_score=cost_score,
            risk_score=risk_score,
            total_score=min(1.0, max(0.0, total_score)),
            confidence=self._calculate_confidence(expert, context)
        )
    
    def _calculate_fit_score(self, expert: Dict[str, Any], 
                            current_state: str,
                            current_method: str) -> float:
        """Calculate how well expert fits current state and method"""
        
        fit_hints = expert.get("fit_hints", [])
        
        # Check if current state is in fit hints
        state_match = 1.0 if current_state in fit_hints else 0.0
        
        # Check if current method is in fit hints
        method_match = 1.0 if current_method in fit_hints else 0.0
        
        # Calculate semantic similarity (simplified)
        semantic_score = 0.5  # Default baseline
        
        # Boost for exact matches
        if state_match and method_match:
            return 1.0
        elif state_match or method_match:
            return 0.8
        else:
            return semantic_score
    
    def _calculate_utility_score(self, expert_id: str, 
                                expert: Dict[str, Any],
                                context: Dict[str, Any]) -> float:
        """Calculate expected utility based on prior and history"""
        
        # Base quality prior
        quality_prior = expert.get("quality_prior", 0.5)
        
        # Adjust based on usage history
        if expert_id in self.expert_usage_history:
            history = self.expert_usage_history[expert_id]
            success_rate = history.get("success_rate", quality_prior)
            # Weighted average of prior and historical performance
            return 0.3 * quality_prior + 0.7 * success_rate
        
        return quality_prior
    
    def _calculate_cost_score(self, expert: Dict[str, Any], 
                             context: Dict[str, Any]) -> float:
        """Calculate cost score (inverted - lower cost = higher score)"""
        
        expert_cost = expert.get("cost", {})
        
        # Normalize costs to 0-1 scale
        cpu_cost = expert_cost.get("cpu", 1.0) / 10.0  # Assume max 10 CPU
        gpu_cost = expert_cost.get("gpu", 0.0) / 4.0   # Assume max 4 GPU
        time_cost = expert_cost.get("time_s", 60.0) / 300.0  # Assume max 5 min
        
        # Combined cost (weighted average)
        total_cost = 0.3 * cpu_cost + 0.5 * gpu_cost + 0.2 * time_cost
        
        # Invert so lower cost = higher score
        return max(0.0, 1.0 - total_cost)
    
    def _calculate_risk_score(self, expert: Dict[str, Any],
                             context: Dict[str, Any]) -> float:
        """Calculate risk score (inverted - lower risk = higher score)"""
        
        expert_risk = expert.get("risk", {})
        
        safety_risk = expert_risk.get("safety", 0.0)
        privacy_risk = expert_risk.get("privacy", 0.0)
        
        # Combined risk (max of individual risks for conservative approach)
        total_risk = max(safety_risk, privacy_risk)
        
        # Invert so lower risk = higher score
        return 1.0 - total_risk
    
    def _calculate_ucb1_bonus(self, expert_id: str) -> float:
        """Calculate UCB1 exploration bonus"""
        
        if expert_id not in self.expert_usage_history:
            return self.exploration_c  # High bonus for unexplored experts
        
        history = self.expert_usage_history[expert_id]
        n_expert = history.get("usage_count", 1)
        n_total = sum(h.get("usage_count", 0) 
                     for h in self.expert_usage_history.values())
        
        if n_total <= 1:
            return self.exploration_c
        
        # UCB1 formula: c * sqrt(ln(n_total) / n_expert)
        bonus = self.exploration_c * math.sqrt(math.log(n_total) / n_expert)
        return min(bonus, 0.5)  # Cap bonus to prevent runaway exploration
    
    def _calculate_confidence(self, expert: Dict[str, Any], 
                             context: Dict[str, Any]) -> float:
        """Calculate confidence in expert selection"""
        
        # Base confidence from quality prior
        base_confidence = expert.get("quality_prior", 0.5)
        
        # Boost confidence if we have usage history
        expert_id = expert["id"]
        if expert_id in self.expert_usage_history:
            history = self.expert_usage_history[expert_id]
            usage_count = history.get("usage_count", 0)
            # More usage = higher confidence (with diminishing returns)
            usage_boost = min(0.3, 0.1 * math.sqrt(usage_count))
            base_confidence += usage_boost
        
        return min(1.0, base_confidence)
    
    def apply_gating_rules(self, expert_selections: List[ExpertSelection],
                          context: Dict[str, Any]) -> List[ExpertSelection]:
        """Apply gating rules to filter expert selections"""
        
        filtered_selections = []
        
        for selection in expert_selections:
            blocked = False
            
            for rule in self.gating_rules:
                condition = rule.get("if", "")
                action = rule.get("then", "")
                
                # Simplified rule evaluation
                if self._evaluate_gating_condition(condition, selection, context):
                    if action == "require_hil_approval":
                        # Would trigger human-in-loop approval
                        logger.warning(
                            f"Expert {selection.expert_id} requires HIL approval "
                            f"due to: {condition}"
                        )
                        # For now, skip this expert
                        blocked = True
                        break
                    elif action == "force_low_cost_only":
                        # Only allow if low cost
                        total_cost = sum(selection.estimated_cost.values())
                        if total_cost > 2.0:  # Arbitrary threshold
                            blocked = True
                            break
            
            if not blocked:
                filtered_selections.append(selection)
        
        return filtered_selections
    
    def _evaluate_gating_condition(self, condition: str, 
                                  selection: ExpertSelection,
                                  context: Dict[str, Any]) -> bool:
        """Evaluate a gating rule condition"""
        
        # Simplified condition evaluation
        if "risk.safety >" in condition:
            threshold = float(condition.split(">")[1].strip())
            safety_risk = selection.estimated_risk.get("safety", 0.0)
            return safety_risk > threshold
        
        elif "time_remaining <" in condition:
            threshold_str = condition.split("<")[1].strip()
            if "m" in threshold_str:
                threshold_minutes = float(threshold_str.replace("m", ""))
                remaining_time = context.get("time_remaining", {}).get(
                    "total", float('inf')
                )
                return remaining_time < threshold_minutes
        
        return False
    
    def update_expert_history(self, expert_id: str, 
                             success: bool,
                             actual_cost: Dict[str, float],
                             quality_score: float) -> None:
        """Update expert usage history"""
        
        if expert_id not in self.expert_usage_history:
            self.expert_usage_history[expert_id] = {
                "usage_count": 0,
                "success_count": 0,
                "total_cost": {"cpu": 0, "gpu": 0, "time_s": 0},
                "success_rate": 0.5,
                "avg_quality": 0.5
            }
        
        history = self.expert_usage_history[expert_id]
        history["usage_count"] += 1
        
        if success:
            history["success_count"] += 1
        
        # Update costs
        for resource, cost in actual_cost.items():
            history["total_cost"][resource] = (
                history["total_cost"].get(resource, 0) + cost
            )
        
        # Update success rate
        history["success_rate"] = (
            history["success_count"] / history["usage_count"]
        )
        
        # Update average quality
        current_avg = history["avg_quality"]
        count = history["usage_count"]
        history["avg_quality"] = (
            (current_avg * (count - 1) + quality_score) / count
        )


class MoERouter:
    """Mixture-of-Experts router for E-UPUSF orchestration"""
    
    def __init__(self, experts: List[Dict[str, Any]], 
                 routing_config: Dict[str, Any]):
        self.experts = {expert["id"]: expert for expert in experts}
        self.routing_config = routing_config
        self.gating = ExpertGating(routing_config)
    
    async def route_to_experts(self, current_state: str,
                              current_method: str,
                              required_inputs: List[str],
                              context: Dict[str, Any]) -> RoutingDecision:
        """Route task to best expert(s) based on current context"""
        
        # Score all compatible experts
        expert_scores = []
        reasoning = []
        
        for expert_id, expert in self.experts.items():
            # Check input compatibility
            expert_inputs = expert.get("inputs", [])
            if not self._check_input_compatibility(required_inputs, expert_inputs):
                reasoning.append(
                    f"Skipped {expert_id}: incompatible inputs "
                    f"(required: {required_inputs}, provides: {expert_inputs})"
                )
                continue
            
            # Score expert
            score = self.gating.score_expert(
                expert, current_state, current_method, context
            )
            expert_scores.append((expert, score))
            
            reasoning.append(
                f"Scored {expert_id}: fit={score.fit_score:.2f}, "
                f"utility={score.utility_score:.2f}, "
                f"cost={score.cost_score:.2f}, "
                f"risk={score.risk_score:.2f}, "
                f"total={score.total_score:.2f}"
            )
        
        if not expert_scores:
            raise RuntimeError("No compatible experts found")
        
        # Sort by total score
        expert_scores.sort(key=lambda x: x[1].total_score, reverse=True)
        
        # Apply exploration policy
        if self.gating.exploration_policy == ExplorationPolicy.EPSILON_GREEDY:
            if random.random() < self.gating.exploration_epsilon:
                # Explore: randomly select from top-k
                top_k_experts = expert_scores[:self.gating.top_k]
                expert, score = random.choice(top_k_experts)
                expert_scores = [(expert, score)] + [
                    x for x in expert_scores if x[0]["id"] != expert["id"]
                ]
                reasoning.append(f"Applied epsilon-greedy exploration")
        
        # Create expert selections
        selections = []
        total_cost = {"cpu": 0, "gpu": 0, "time_s": 0}
        
        for expert, score in expert_scores[:self.gating.top_k]:
            selection = ExpertSelection(
                expert_id=expert["id"],
                score=score,
                inputs=expert.get("inputs", []),
                outputs=expert.get("outputs", []),
                estimated_cost=expert.get("cost", {}),
                estimated_risk=expert.get("risk", {})
            )
            selections.append(selection)
            
            # Accumulate costs
            for resource, cost in selection.estimated_cost.items():
                total_cost[resource] += cost
        
        # Apply gating rules
        filtered_selections = self.gating.apply_gating_rules(selections, context)
        
        if not filtered_selections:
            raise RuntimeError("All experts blocked by gating rules")
        
        # Select primary and backup experts
        primary_expert = filtered_selections[0]
        backup_experts = filtered_selections[1:3]  # Up to 2 backups
        
        # Calculate overall confidence
        confidence = primary_expert.score.confidence
        if backup_experts:
            # Boost confidence if we have good backups
            backup_scores = [expert.score.total_score for expert in backup_experts]
            avg_backup_score = sum(backup_scores) / len(backup_scores)
            confidence = min(1.0, confidence + 0.1 * avg_backup_score)
        
        return RoutingDecision(
            primary_expert=primary_expert,
            backup_experts=backup_experts,
            routing_strategy="primary_with_backup",
            total_estimated_cost=total_cost,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _check_input_compatibility(self, required: List[str], 
                                  provided: List[str]) -> bool:
        """Check if expert inputs are compatible with requirements"""
        
        # Simple compatibility check - expert must support all required inputs
        return all(req in provided for req in required)
    
    async def execute_expert(self, selection: ExpertSelection,
                            inputs: Dict[str, Any],
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute selected expert with given inputs"""
        
        expert_id = selection.expert_id
        expert = self.experts[expert_id]
        
        logger.info(f"Executing expert: {expert_id}")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Simulate expert execution
            if expert.get("kind") == "tool":
                result = await self._execute_tool(expert, inputs, context)
            elif expert.get("kind") == "agent":
                result = await self._execute_agent(expert, inputs, context)
            else:
                result = await self._execute_human(expert, inputs, context)
            
            end_time = asyncio.get_event_loop().time()
            actual_duration = end_time - start_time
            
            # Update expert history
            actual_cost = {
                "cpu": selection.estimated_cost.get("cpu", 0),
                "gpu": selection.estimated_cost.get("gpu", 0),
                "time_s": actual_duration
            }
            
            quality_score = result.get("quality_score", 0.8)
            success = result.get("success", True)
            
            self.gating.update_expert_history(
                expert_id, success, actual_cost, quality_score
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Expert {expert_id} execution failed: {e}")
            
            # Update history with failure
            self.gating.update_expert_history(
                expert_id, False, selection.estimated_cost, 0.0
            )
            
            raise
    
    async def _execute_tool(self, expert: Dict[str, Any], 
                           inputs: Dict[str, Any],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool-based expert"""
        
        # Simulate tool execution
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "success": True,
            "outputs": {
                "result": f"Tool {expert['id']} processed inputs successfully",
                "timestamp": asyncio.get_event_loop().time()
            },
            "quality_score": 0.8,
            "metadata": {
                "expert_type": "tool",
                "expert_id": expert["id"]
            }
        }
    
    async def _execute_agent(self, expert: Dict[str, Any],
                            inputs: Dict[str, Any],
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent-based expert"""
        
        # Simulate agent execution
        await asyncio.sleep(0.2)  # Simulate longer processing time
        
        return {
            "success": True,
            "outputs": {
                "result": f"Agent {expert['id']} completed analysis",
                "reasoning": "Applied domain-specific reasoning",
                "confidence": 0.85,
                "timestamp": asyncio.get_event_loop().time()
            },
            "quality_score": 0.85,
            "metadata": {
                "expert_type": "agent",
                "expert_id": expert["id"]
            }
        }
    
    async def _execute_human(self, expert: Dict[str, Any],
                            inputs: Dict[str, Any],
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute human-in-loop expert"""
        
        # Simulate human consultation (would be actual HIL in practice)
        await asyncio.sleep(1.0)  # Simulate human response time
        
        return {
            "success": True,
            "outputs": {
                "result": f"Human expert {expert['id']} provided consultation",
                "recommendations": ["Consider alternative approach", 
                                 "Validate assumptions"],
                "timestamp": asyncio.get_event_loop().time()
            },
            "quality_score": 0.9,
            "metadata": {
                "expert_type": "human",
                "expert_id": expert["id"]
            }
        }
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing and expert usage statistics"""
        
        return {
            "total_experts": len(self.experts),
            "expert_usage_history": self.gating.expert_usage_history,
            "routing_config": self.routing_config
        }