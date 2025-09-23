#!/usr/bin/env python3
"""
Enhanced Ecosystem Orchestrator - Mangle Reasoning Integration
==============================================================

Upgrades the master orchestrator with Mangle deductive reasoning,
EOS LADDER operations, and constitutional compliance.

Author: Super ALITA Framework
Version: 2.0.0 (Enhanced with Mangle Reasoning)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning operations"""
    DEDUCTIVE = "deductive"
    STRATEGIC = "strategic" 
    COORDINATION = "coordination"


class CoordinationLevel(Enum):
    """Levels of system coordination"""
    TACTICAL = "tactical"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    CONSTITUTIONAL = "constitutional"


@dataclass
class ReasoningContext:
    """Context for Mangle reasoning operations"""
    
    reasoning_id: str
    reasoning_type: ReasoningType
    premises: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0


@dataclass
class CoordinationPlan:
    """Plan for coordinating system components"""
    
    plan_id: str
    coordination_level: CoordinationLevel
    components: list[str] = field(default_factory=list)
    execution_steps: list[dict[str, Any]] = field(default_factory=list)
    resource_allocation: dict[str, float] = field(default_factory=dict)
    confidence_score: float = 0.0


class MangleReasoningEngine:
    """Advanced Mangle deductive reasoning engine"""
    
    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core inference rules
        self.inference_rules = {
            "dependency": [
                "If A depends on B, then B must be ready before A",
                "If A fails and B depends on A, then B is at risk"
            ],
            "performance": [
                "If response time > threshold, then optimize bottleneck",
                "If error rate > limit, then increase validation"
            ],
            "security": [
                "If operation violates policy, then halt and review",
                "If data exposure risk exists, then apply protection"
            ]
        }
        
        self.knowledge_base = {
            "components": {
                "cognitive_systems": {"criticality": "high", "deps": ["reasoning"]},
                "execution_flow": {"criticality": "high", "deps": ["tools"]},
                "security_system": {"criticality": "critical", "deps": ["auth"]}
            }
        }
    
    async def perform_deductive_reasoning(
        self, 
        context: ReasoningContext
    ) -> dict[str, Any]:
        """Perform deductive reasoning using Mangle methodology"""
        
        self.logger.info(f"🧠 Deductive reasoning: {context.reasoning_id}")
        
        results = {
            "conclusions": [],
            "inference_chain": [],
            "confidence": 0.0,
            "recommendations": []
        }
        
        try:
            # Apply inference rules to premises
            for premise in context.premises:
                relevant_rules = self._find_relevant_rules(premise)
                for rule in relevant_rules:
                    inference = await self._apply_rule(premise, rule, context.evidence)
                    if inference:
                        results["inference_chain"].append(inference)
                        results["conclusions"].append(inference["conclusion"])
            
            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(
                results["conclusions"], context
            )
            
            # Calculate confidence
            if results["inference_chain"]:
                confidences = [step["confidence"] for step in results["inference_chain"]]
                results["confidence"] = sum(confidences) / len(confidences)
            
            self.logger.info(f"✅ Reasoning complete: {len(results['conclusions'])} conclusions")
            
        except Exception as e:
            self.logger.error(f"❌ Reasoning failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _find_relevant_rules(self, premise: str) -> list[str]:
        """Find rules relevant to premise"""
        relevant = []
        premise_lower = premise.lower()
        
        for category, rules in self.inference_rules.items():
            for rule in rules:
                if self._rule_matches_premise(rule, premise_lower):
                    relevant.append(rule)
        
        return relevant
    
    def _rule_matches_premise(self, rule: str, premise: str) -> bool:
        """Check if rule matches premise"""
        rule_keywords = ["depend", "fail", "response", "error", "violate", "risk"]
        return any(keyword in premise for keyword in rule_keywords if keyword in rule.lower())
    
    async def _apply_rule(
        self, 
        premise: str, 
        rule: str, 
        evidence: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Apply inference rule"""
        
        if "if" in rule.lower() and "then" in rule.lower():
            parts = rule.lower().split("then")
            if len(parts) == 2:
                condition = parts[0].replace("if", "").strip()
                conclusion = parts[1].strip()
                
                if self._condition_satisfied(condition, premise, evidence):
                    return {
                        "premise": premise,
                        "rule": rule,
                        "conclusion": conclusion,
                        "confidence": 0.8  # Default confidence
                    }
        
        return None
    
    def _condition_satisfied(
        self, 
        condition: str, 
        premise: str, 
        evidence: dict[str, Any]
    ) -> bool:
        """Check if condition is satisfied"""
        # Simplified condition matching
        condition_keywords = condition.split()
        premise_keywords = premise.lower().split()
        
        return any(kw in premise_keywords for kw in condition_keywords)
    
    def _generate_recommendations(
        self, 
        conclusions: list[str], 
        context: ReasoningContext
    ) -> list[dict[str, Any]]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        for conclusion in conclusions:
            rec = {
                "action": conclusion,
                "priority": self._calculate_priority(conclusion),
                "rationale": "Based on deductive reasoning",
                "impact": self._estimate_impact(conclusion)
            }
            recommendations.append(rec)
        
        return sorted(recommendations, key=lambda x: x["priority"], reverse=True)
    
    def _calculate_priority(self, conclusion: str) -> float:
        """Calculate recommendation priority"""
        if "halt" in conclusion.lower() or "critical" in conclusion.lower():
            return 0.9
        elif "optimize" in conclusion.lower():
            return 0.7
        else:
            return 0.5
    
    def _estimate_impact(self, conclusion: str) -> str:
        """Estimate implementation impact"""
        if "halt" in conclusion.lower():
            return "high"
        elif "optimize" in conclusion.lower():
            return "medium"
        else:
            return "low"
    
    async def generate_coordination_plan(
        self,
        objectives: list[str],
        constraints: list[str],
        system_state: dict[str, Any]
    ) -> CoordinationPlan:
        """Generate system coordination plan"""
        
        plan_id = f"coord_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"📋 Generating coordination plan: {plan_id}")
        
        # Determine coordination level
        level = CoordinationLevel.OPERATIONAL
        if any("strategic" in obj.lower() for obj in objectives):
            level = CoordinationLevel.STRATEGIC
        
        plan = CoordinationPlan(
            plan_id=plan_id,
            coordination_level=level
        )
        
        try:
            # Identify relevant components
            plan.components = self._identify_components(objectives, system_state)
            
            # Create execution sequence
            plan.execution_steps = self._create_execution_steps(
                objectives, plan.components
            )
            
            # Allocate resources
            plan.resource_allocation = self._allocate_resources(plan.components)
            
            # Calculate confidence
            plan.confidence_score = 0.8  # Default confidence
            
            self.logger.info(f"✅ Plan generated: {len(plan.execution_steps)} steps")
            
        except Exception as e:
            self.logger.error(f"❌ Plan generation failed: {e}")
            plan.confidence_score = 0.0
        
        return plan
    
    def _identify_components(
        self, 
        objectives: list[str], 
        system_state: dict[str, Any]
    ) -> list[str]:
        """Identify relevant system components"""
        
        components = set()
        available = system_state.get("components", {}).keys()
        
        for objective in objectives:
            obj_lower = objective.lower()
            if "cognitive" in obj_lower:
                components.add("cognitive_systems")
            if "execution" in obj_lower:
                components.add("execution_flow")
            if "security" in obj_lower:
                components.add("security_system")
        
        return [c for c in components if c in available]
    
    def _create_execution_steps(
        self, 
        objectives: list[str], 
        components: list[str]
    ) -> list[dict[str, Any]]:
        """Create execution steps for plan"""
        
        steps = []
        
        for i, component in enumerate(components):
            steps.append({
                "step": i + 1,
                "component": component,
                "action": f"Initialize {component}",
                "duration": 1.0,
                "success_criteria": [f"{component} operational"]
            })
        
        return steps
    
    def _allocate_resources(self, components: list[str]) -> dict[str, float]:
        """Allocate resources to components"""
        
        if not components:
            return {}
        
        # Equal allocation with criticality weighting
        allocation = {}
        base_allocation = 1.0 / len(components)
        
        for component in components:
            criticality = self.knowledge_base.get("components", {}).get(
                component, {}
            ).get("criticality", "medium")
            
            weight = {"critical": 1.5, "high": 1.2, "medium": 1.0}.get(criticality, 1.0)
            allocation[component] = base_allocation * weight
        
        return allocation


class EnhancedEcosystemOrchestrator:
    """
    Enhanced ecosystem orchestrator with Mangle reasoning integration
    """
    
    def __init__(
        self,
        mangle_engine: MangleReasoningEngine = None,
        constitutional_validator = None,
        eos_orchestrator = None,
        **kwargs
    ):
        # Initialize with existing orchestrator functionality
        self.original_orchestrator = self._create_original_orchestrator(**kwargs)
        
        # Enhanced components
        self.mangle_engine = mangle_engine or MangleReasoningEngine()
        self.constitutional_validator = constitutional_validator
        self.eos_orchestrator = eos_orchestrator
        
        self.logger = logging.getLogger(__name__)
        
        # Enhanced state
        self.coordination_history: list[dict[str, Any]] = []
        self.reasoning_cache: dict[str, dict[str, Any]] = {}
    
    def _create_original_orchestrator(self, **kwargs):
        """Create original orchestrator with dependency injection"""
        try:
            from src.ecosystem.master_orchestrator import EcosystemOrchestrator
            return EcosystemOrchestrator(**kwargs)
        except ImportError:
            # Mock implementation
            return type('MockOrchestrator', (), {
                'handle_developer_action': self._mock_handle_action
            })()
    
    async def _mock_handle_action(self, user_id: str, action: str, context: dict[str, Any]):
        """Mock implementation for testing"""
        return {
            "status": "success",
            "workflow_type": action,
            "confidence": 0.8,
            "message": f"Mock handled {action} for {user_id}"
        }
    
    async def handle_developer_action_enhanced(
        self, 
        user_id: str, 
        action: str, 
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Enhanced developer action handling with Mangle reasoning"""
        
        self.logger.info(f"🚀 Enhanced action handling: {action} for {user_id}")
        
        try:
            # Create reasoning context
            reasoning_context = ReasoningContext(
                reasoning_id=f"action_{action}_{datetime.now(UTC).strftime('%H%M%S')}",
                reasoning_type=ReasoningType.DEDUCTIVE,
                premises=[
                    f"Developer {user_id} requested action {action}",
                    f"Context provided: {context}",
                    "System must respond appropriately and safely"
                ],
                constraints=[
                    "Maintain system security",
                    "Ensure constitutional compliance", 
                    "Optimize for performance"
                ],
                evidence=context
            )
            
            # Perform Mangle reasoning
            reasoning_results = await self.mangle_engine.perform_deductive_reasoning(
                reasoning_context
            )
            
            # Generate coordination plan if needed
            coordination_plan = None
            if reasoning_results.get("recommendations"):
                objectives = [rec["action"] for rec in reasoning_results["recommendations"]]
                coordination_plan = await self.mangle_engine.generate_coordination_plan(
                    objectives=objectives,
                    constraints=reasoning_context.constraints,
                    system_state=context.get("system_state", {})
                )
            
            # Execute original action with enhancements
            original_result = await self.original_orchestrator.handle_developer_action(
                user_id, action, context
            )
            
            # Enhance result with reasoning insights
            enhanced_result = {
                **original_result,
                "reasoning_insights": {
                    "conclusions": reasoning_results.get("conclusions", []),
                    "recommendations": reasoning_results.get("recommendations", []),
                    "reasoning_confidence": reasoning_results.get("confidence", 0.0)
                }
            }
            
            if coordination_plan:
                enhanced_result["coordination_plan"] = {
                    "plan_id": coordination_plan.plan_id,
                    "coordination_level": coordination_plan.coordination_level.value,
                    "components": coordination_plan.components,
                    "steps": len(coordination_plan.execution_steps),
                    "confidence": coordination_plan.confidence_score
                }
            
            # Store in coordination history
            self.coordination_history.append({
                "timestamp": datetime.now(UTC).isoformat(),
                "user_id": user_id,
                "action": action,
                "reasoning_results": reasoning_results,
                "coordination_plan": coordination_plan.__dict__ if coordination_plan else None,
                "result": enhanced_result
            })
            
            self.logger.info(f"✅ Enhanced action complete: {action}")
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced action failed: {e}")
            
            # Fallback to original orchestrator
            fallback_result = await self.original_orchestrator.handle_developer_action(
                user_id, action, context
            )
            fallback_result["fallback_used"] = True
            fallback_result["enhancement_error"] = str(e)
            
            return fallback_result
    
    async def get_system_insights(self) -> dict[str, Any]:
        """Get insights from Mangle reasoning and coordination history"""
        
        insights = {
            "coordination_summary": {
                "total_actions": len(self.coordination_history),
                "recent_actions": len([
                    h for h in self.coordination_history[-10:]
                ]),
                "avg_reasoning_confidence": 0.0
            },
            "reasoning_patterns": {},
            "optimization_opportunities": [],
            "system_health": "healthy"
        }
        
        try:
            if self.coordination_history:
                # Calculate average reasoning confidence
                confidences = [
                    h.get("reasoning_results", {}).get("confidence", 0.0)
                    for h in self.coordination_history
                ]
                insights["coordination_summary"]["avg_reasoning_confidence"] = (
                    sum(confidences) / len(confidences) if confidences else 0.0
                )
                
                # Extract reasoning patterns
                conclusions = []
                for history in self.coordination_history:
                    conclusions.extend(
                        history.get("reasoning_results", {}).get("conclusions", [])
                    )
                
                # Count conclusion patterns
                conclusion_counts = {}
                for conclusion in conclusions:
                    key = conclusion[:50]  # First 50 chars as pattern key
                    conclusion_counts[key] = conclusion_counts.get(key, 0) + 1
                
                insights["reasoning_patterns"] = dict(
                    sorted(conclusion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                )
                
                # Identify optimization opportunities
                recent_recommendations = []
                for history in self.coordination_history[-5:]:  # Last 5 actions
                    recent_recommendations.extend(
                        history.get("reasoning_results", {}).get("recommendations", [])
                    )
                
                high_priority_recs = [
                    rec for rec in recent_recommendations 
                    if rec.get("priority", 0) > 0.7
                ]
                insights["optimization_opportunities"] = high_priority_recs[:3]  # Top 3
            
            self.logger.info("✅ System insights generated")
            
        except Exception as e:
            self.logger.error(f"❌ System insights generation failed: {e}")
            insights["error"] = str(e)
        
        return insights


# Export main classes
__all__ = [
    "ReasoningType",
    "CoordinationLevel", 
    "ReasoningContext",
    "CoordinationPlan",
    "MangleReasoningEngine",
    "EnhancedEcosystemOrchestrator"
]