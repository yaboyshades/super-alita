#!/usr/bin/env python3
"""
Decision Policy v1 - Implementation-grade agency decision maker

This module implements the tight specification for "how the agent decides what to do"
when given agency. Designed to work with Tool Registry, MCP registry, and Neural Atom
registry in a single response call (plan → execute → stream → finalize).

Key Features:
- Intent classification and goal synthesis
- Unified capability matching across all registries
- Utility-based tool selection with multi-armed bandit learning
- Strategy selection (SINGLE_BEST, SEQUENTIAL, PARALLEL, DELEGATE, GUARDRAIL)
- Plan DSL generation for deterministic execution
- Budget enforcement and safety controls
- Circuit breaker pattern for reliability
"""

import json
import logging
import math
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """User intent classifications"""

    BOOTSTRAP = auto()  # Setup/initialization tasks
    QUERY = auto()  # Information retrieval
    CREATE = auto()  # Content/resource creation
    MODIFY = auto()  # Update existing resources
    ANALYZE = auto()  # Data analysis/processing
    MONITOR = auto()  # Status checking/monitoring
    COLLABORATE = auto()  # Multi-step coordination
    UNKNOWN = auto()  # Unclear intent


class StrategyType(Enum):
    """Execution strategies"""

    SINGLE_BEST = auto()  # One tool with highest utility
    SEQUENTIAL = auto()  # Dependencies require ordered execution
    PARALLEL = auto()  # Independent subtasks
    DELEGATE = auto()  # Handoff to specialized sub-agent
    GUARDRAIL = auto()  # Safety gate or human confirmation required


class RiskLevel(Enum):
    """Risk assessment levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Goal:
    """Structured goal representation"""

    intent: IntentType
    description: str
    slots: Dict[str, Any]
    success_criteria: List[str]
    constraints: List[str]
    risk_level: RiskLevel


@dataclass
class CapabilityNode:
    """Node in the unified capability graph"""

    id: str
    name: str
    description: str
    schema: Dict[str, Any]
    preconditions: List[str]
    side_effects: List[str]
    cost_hint: float
    streamable: bool = True
    max_retries: int = 3
    registry_type: str = "unknown"  # normal, mcp, neural_atom

    # Performance tracking
    wins: int = 0
    attempts: int = 0
    avg_latency: float = 0.0
    avg_cost: float = 0.0
    recent_failures: int = 0
    circuit_open: bool = False
    last_success: Optional[float] = None


@dataclass
class Budget:
    """Execution budget constraints"""

    max_steps: int = 20
    max_tool_calls: int = 10
    timeout_ms: int = 120000
    cost_cap: float = 10.0


@dataclass
class PolicyConfig:
    """Configuration for decision policy"""

    # Matching weights
    w_schema_fit: float = 0.35
    w_text_sim: float = 0.20
    w_precond: float = 0.25
    w_history: float = 0.20
    w_risk: float = 0.30

    # Utility parameters
    alpha_latency: float = 0.2
    beta_cost: float = 1.0
    gamma_risk: float = 0.5
    explore_epsilon: float = 0.15

    # Thresholds
    min_match: float = 0.55
    min_utility: float = 0.10
    parallel_delta: float = 0.05

    # Circuit breaker
    circuit_failures: int = 3
    circuit_window_s: int = 120
    circuit_cooldown_s: int = 60

    # Safety
    high_risk_requires_human: bool = True


@dataclass
class PlanStep:
    """Single step in execution plan"""

    type: str  # "say", "tool", "if", "else"
    content: Optional[str] = None
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    assign: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None
    steps: Optional[List["PlanStep"]] = None


@dataclass
class ExecutionPlan:
    """Complete execution plan with budget"""

    run_id: str
    budget: Budget
    plan: List[Dict[str, Any]]
    strategy: StrategyType
    confidence: float
    estimated_cost: float
    risk_factors: List[str]


class DecisionPolicyEngine:
    """Core decision policy implementation"""

    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self.capabilities: Dict[str, CapabilityNode] = {}
        self.intent_classifier = IntentClassifier()
        self.goal_synthesizer = GoalSynthesizer()
        self.utility_calculator = UtilityCalculator(self.config)
        self.plan_builder = PlanBuilder()

        # Bandit state
        self.bandit_stats: Dict[str, Dict[str, float]] = {}

    def register_capability(self, capability: CapabilityNode):
        """Register a capability from any registry type"""
        self.capabilities[capability.id] = capability
        if capability.id not in self.bandit_stats:
            self.bandit_stats[capability.id] = {
                "wins": 0,
                "attempts": 0,
                "avg_reward": 0.0,
                "explore_count": 0,
            }

    async def decide_and_plan(
        self, message: str, ctx: Dict[str, Any], budget: Optional[Budget] = None
    ) -> ExecutionPlan:
        """Main decision entry point"""

        # 1. Intent → Goal → Constraints
        intent = self.intent_classifier.classify(message)
        slots = self.extract_slots(message)
        goal = self.goal_synthesizer.synthesize(intent, slots, ctx)

        # 2. Capability matching across all registries
        candidates = self.resolve_candidates(goal)

        # 3. Utility calculation and strategy selection
        scored_candidates = []
        for candidate in candidates:
            if self.is_circuit_open(candidate):
                continue

            match_score = self.calculate_match_score(candidate, goal, ctx)
            if match_score < self.config.min_match:
                continue

            utility = self.utility_calculator.calculate(candidate, goal, ctx)
            if utility < self.config.min_utility:
                continue

            scored_candidates.append((utility, candidate, match_score))

        if not scored_candidates:
            return self.safe_fallback_plan(goal, budget)

        # Sort by utility (descending)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 4. Strategy selection
        strategy = self.pick_strategy(scored_candidates, goal)

        # 5. Build execution plan
        plan = self.plan_builder.build_plan(
            strategy, scored_candidates, goal, budget or Budget()
        )

        return plan

    def extract_slots(self, message: str) -> Dict[str, Any]:
        """Extract structured slots from user message"""
        slots = {}

        # Simple regex-based extraction (can be enhanced with NER)
        import re

        # Repository URLs
        repo_pattern = r"https?://github\.com/[\w-]+/[\w.-]+"
        repo_match = re.search(repo_pattern, message)
        if repo_match:
            slots["repo_url"] = repo_match.group()

        # Ports
        port_pattern = r"port\s+(\d+)"
        port_match = re.search(port_pattern, message, re.IGNORECASE)
        if port_match:
            slots["port"] = int(port_match.group(1))

        # File paths
        path_pattern = r"[~./][\w/.-]+"
        path_matches = re.findall(path_pattern, message)
        if path_matches:
            slots["paths"] = path_matches

        return slots

    def resolve_candidates(self, goal: Goal) -> List[CapabilityNode]:
        """Find candidate capabilities for the goal"""
        candidates = []

        for cap_id, capability in self.capabilities.items():
            # Basic text matching (can be enhanced with embeddings)
            if self.text_similarity(goal.description, capability.description) > 0.3:
                candidates.append(capability)

            # Schema compatibility check
            if self.schema_compatible(goal, capability):
                candidates.append(capability)

        # Remove duplicates
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate.id not in seen:
                seen.add(candidate.id)
                unique_candidates.append(candidate)

        return unique_candidates

    def calculate_match_score(
        self, capability: CapabilityNode, goal: Goal, ctx: Dict[str, Any]
    ) -> float:
        """Calculate match score using weighted factors"""

        schema_fit = self.schema_fitness(goal, capability)
        text_sim = self.text_similarity(goal.description, capability.description)
        precond_sat = self.precondition_satisfaction(capability, ctx)
        historical = self.historical_success(capability)
        risk_penalty = self.risk_penalty(capability, goal.risk_level)

        score = (
            self.config.w_schema_fit * schema_fit
            + self.config.w_text_sim * text_sim
            + self.config.w_precond * precond_sat
            + self.config.w_history * historical
            - self.config.w_risk * risk_penalty
        )

        return max(0.0, min(1.0, score))

    def schema_fitness(self, goal: Goal, capability: CapabilityNode) -> float:
        """Calculate I/O schema compatibility"""
        # Simplified implementation - can be enhanced with actual schema validation
        goal_slots = set(goal.slots.keys())
        cap_inputs = set(
            capability.schema.get("input_schema", {}).get("properties", {}).keys()
        )

        if not cap_inputs:
            return 0.5  # No schema info

        overlap = len(goal_slots.intersection(cap_inputs))
        union = len(goal_slots.union(cap_inputs))

        return overlap / max(1, union)

    def text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        # Simplified implementation using word overlap
        # In production, use sentence transformers or similar
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return overlap / max(1, union)

    def precondition_satisfaction(
        self, capability: CapabilityNode, ctx: Dict[str, Any]
    ) -> float:
        """Check if preconditions are satisfied"""
        if not capability.preconditions:
            return 1.0

        satisfied = 0
        for precond in capability.preconditions:
            # Simple key-based checking
            if precond in ctx:
                satisfied += 1

        return satisfied / len(capability.preconditions)

    def historical_success(self, capability: CapabilityNode) -> float:
        """Get historical success rate for capability"""
        stats = self.bandit_stats.get(capability.id, {})
        attempts = stats.get("attempts", 0)
        wins = stats.get("wins", 0)

        if attempts == 0:
            return 0.5  # No history

        return wins / attempts

    def risk_penalty(self, capability: CapabilityNode, risk_level: RiskLevel) -> float:
        """Calculate risk penalty based on side effects"""
        if not capability.side_effects:
            return 0.0

        # Higher penalty for destructive operations in high-risk scenarios
        destructive_effects = ["delete", "remove", "destroy", "overwrite"]
        has_destructive = any(
            effect in " ".join(capability.side_effects).lower()
            for effect in destructive_effects
        )

        base_penalty = len(capability.side_effects) * 0.1

        if risk_level == RiskLevel.HIGH and has_destructive:
            return base_penalty * 3.0
        elif risk_level == RiskLevel.MEDIUM and has_destructive:
            return base_penalty * 1.5

        return base_penalty

    def is_circuit_open(self, capability: CapabilityNode) -> bool:
        """Check if circuit breaker is open for capability"""
        return capability.circuit_open

    def pick_strategy(
        self, scored_candidates: List[Tuple[float, CapabilityNode, float]], goal: Goal
    ) -> StrategyType:
        """Select execution strategy based on candidates and goal"""

        if not scored_candidates:
            return StrategyType.GUARDRAIL

        top_utility = scored_candidates[0][0]

        # Single dominant candidate
        if len(scored_candidates) == 1 or top_utility > scored_candidates[1][0] + 0.2:
            return StrategyType.SINGLE_BEST

        # Check for dependencies
        has_dependencies = any(
            candidate.preconditions for _, candidate, _ in scored_candidates
        )

        if has_dependencies:
            return StrategyType.SEQUENTIAL

        # Multiple good candidates with similar utility
        similar_utilities = [
            (util, candidate)
            for util, candidate, _ in scored_candidates[:3]
            if abs(util - top_utility) <= self.config.parallel_delta
        ]

        if len(similar_utilities) > 1:
            return StrategyType.PARALLEL

        # High-risk scenario
        if goal.risk_level == RiskLevel.HIGH:
            return StrategyType.GUARDRAIL

        return StrategyType.SINGLE_BEST

    def safe_fallback_plan(self, goal: Goal, budget: Optional[Budget]) -> ExecutionPlan:
        """Create safe fallback plan when no good candidates found"""
        plan_steps = [
            {"say": f"I understand you want to: {goal.description}"},
            {
                "say": "However, I don't have sufficient capabilities to complete this safely."
            },
            {
                "say": "Could you provide more specific instructions or break this into smaller steps?"
            },
        ]

        return ExecutionPlan(
            run_id=f"fallback_{uuid.uuid4().hex[:8]}",
            budget=budget or Budget(),
            plan=plan_steps,
            strategy=StrategyType.GUARDRAIL,
            confidence=0.1,
            estimated_cost=0.0,
            risk_factors=["insufficient_capabilities", "high_uncertainty"],
        )

    def update_bandit_stats(
        self, capability_id: str, success: bool, cost: float, latency: float
    ):
        """Update multi-armed bandit statistics"""
        if capability_id not in self.bandit_stats:
            self.bandit_stats[capability_id] = {
                "wins": 0,
                "attempts": 0,
                "avg_reward": 0.0,
                "explore_count": 0,
            }

        stats = self.bandit_stats[capability_id]
        stats["attempts"] += 1

        if success:
            stats["wins"] += 1

        # Calculate reward: success bonus - cost penalty - latency penalty
        reward = (1.0 if success else 0.0) - 0.1 * cost - 0.01 * latency

        # Update average reward using exponential moving average
        alpha = 0.1
        stats["avg_reward"] = (1 - alpha) * stats["avg_reward"] + alpha * reward


class IntentClassifier:
    """Classifies user intent from message"""

    def classify(self, message: str) -> IntentType:
        """Classify intent using keyword matching"""
        message_lower = message.lower()

        # Bootstrap/setup keywords
        bootstrap_keywords = [
            "setup",
            "install",
            "bootstrap",
            "initialize",
            "spin up",
            "start",
            "clone",
        ]
        if any(keyword in message_lower for keyword in bootstrap_keywords):
            return IntentType.BOOTSTRAP

        # Query keywords
        query_keywords = [
            "show",
            "list",
            "get",
            "find",
            "search",
            "what",
            "how",
            "status",
        ]
        if any(keyword in message_lower for keyword in query_keywords):
            return IntentType.QUERY

        # Create keywords
        create_keywords = ["create", "make", "build", "generate", "new"]
        if any(keyword in message_lower for keyword in create_keywords):
            return IntentType.CREATE

        # Modify keywords
        modify_keywords = ["update", "change", "modify", "edit", "fix"]
        if any(keyword in message_lower for keyword in modify_keywords):
            return IntentType.MODIFY

        # Analysis keywords
        analyze_keywords = ["analyze", "check", "validate", "test", "benchmark"]
        if any(keyword in message_lower for keyword in analyze_keywords):
            return IntentType.ANALYZE

        return IntentType.UNKNOWN


class GoalSynthesizer:
    """Synthesizes structured goals from intent and slots"""

    def synthesize(
        self, intent: IntentType, slots: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Goal:
        """Create structured goal from intent and extracted slots"""

        if intent == IntentType.BOOTSTRAP:
            return Goal(
                intent=intent,
                description="Bootstrap and start service from repository",
                slots=slots,
                success_criteria=[
                    "repository_cloned",
                    "dependencies_installed",
                    "service_running",
                    "health_check_passed",
                ],
                constraints=["no_destructive_changes", "use_existing_config"],
                risk_level=RiskLevel.MEDIUM,
            )

        elif intent == IntentType.QUERY:
            return Goal(
                intent=intent,
                description="Retrieve and present information",
                slots=slots,
                success_criteria=["information_found", "response_provided"],
                constraints=["read_only_operations"],
                risk_level=RiskLevel.LOW,
            )

        # Default goal structure
        return Goal(
            intent=intent,
            description=f"Complete {intent.name.lower()} task",
            slots=slots,
            success_criteria=["task_completed"],
            constraints=[],
            risk_level=RiskLevel.MEDIUM,
        )


class UtilityCalculator:
    """Calculates utility scores for capabilities"""

    def __init__(self, config: PolicyConfig):
        self.config = config

    def calculate(
        self, capability: CapabilityNode, goal: Goal, ctx: Dict[str, Any]
    ) -> float:
        """Calculate utility using the specified formula"""

        # Estimate success probability
        p_success = self.estimate_success_probability(capability)

        # Get performance metrics
        latency = capability.avg_latency or 1.0
        cost = capability.cost_hint or 0.1

        # Calculate risk penalty
        risk_penalty = self.calculate_risk_penalty(capability, goal.risk_level)

        # Calculate exploration bonus
        explore_bonus = self.calculate_exploration_bonus(capability)

        # Reward is based on goal importance (simplified)
        reward = 1.0

        # Utility formula: p_success*Reward - α*latency - β*cost - γ*risk_penalty + explore_bonus
        utility = (
            p_success * reward
            - self.config.alpha_latency * latency
            - self.config.beta_cost * cost
            - self.config.gamma_risk * risk_penalty
            + explore_bonus
        )

        return utility

    def estimate_success_probability(self, capability: CapabilityNode) -> float:
        """Estimate success probability from historical data"""
        if capability.attempts == 0:
            return 0.5  # No history, assume 50%

        success_rate = capability.wins / capability.attempts

        # Apply confidence adjustment based on sample size
        confidence_factor = min(1.0, capability.attempts / 10.0)

        return 0.5 + confidence_factor * (success_rate - 0.5)

    def calculate_risk_penalty(
        self, capability: CapabilityNode, risk_level: RiskLevel
    ) -> float:
        """Calculate risk penalty based on side effects and risk level"""
        if not capability.side_effects:
            return 0.0

        base_penalty = len(capability.side_effects) * 0.1

        if risk_level == RiskLevel.HIGH:
            return base_penalty * 2.0
        elif risk_level == RiskLevel.MEDIUM:
            return base_penalty * 1.2

        return base_penalty

    def calculate_exploration_bonus(self, capability: CapabilityNode) -> float:
        """Calculate exploration bonus using epsilon-greedy approach"""
        uses = capability.attempts
        bonus = self.config.explore_epsilon / math.sqrt(1 + uses)
        return bonus


class PlanBuilder:
    """Builds execution plans in Plan DSL format"""

    def build_plan(
        self,
        strategy: StrategyType,
        scored_candidates: List[Tuple[float, CapabilityNode, float]],
        goal: Goal,
        budget: Budget,
    ) -> ExecutionPlan:
        """Build execution plan based on strategy and candidates"""

        run_id = f"r_{uuid.uuid4().hex[:8]}"

        if strategy == StrategyType.SINGLE_BEST:
            plan = self.build_single_best_plan(scored_candidates[0][1], goal)
        elif strategy == StrategyType.SEQUENTIAL:
            plan = self.build_sequential_plan(scored_candidates, goal)
        elif strategy == StrategyType.PARALLEL:
            plan = self.build_parallel_plan(scored_candidates, goal)
        elif strategy == StrategyType.DELEGATE:
            plan = self.build_delegate_plan(scored_candidates, goal)
        else:  # GUARDRAIL
            plan = self.build_guardrail_plan(goal)

        # Calculate confidence and cost
        confidence = self.calculate_plan_confidence(scored_candidates, strategy)
        estimated_cost = self.calculate_estimated_cost(scored_candidates, strategy)

        return ExecutionPlan(
            run_id=run_id,
            budget=budget,
            plan=plan,
            strategy=strategy,
            confidence=confidence,
            estimated_cost=estimated_cost,
            risk_factors=self.identify_risk_factors(scored_candidates, goal),
        )

    def build_single_best_plan(
        self, capability: CapabilityNode, goal: Goal
    ) -> List[Dict[str, Any]]:
        """Build plan for single best capability"""
        return [
            {"say": f"Executing {capability.name}..."},
            {
                "tool": capability.id,
                "args": self.extract_args_from_goal(goal, capability),
                "assign": "result",
            },
            {"say": "Task completed successfully."},
        ]

    def build_sequential_plan(
        self, scored_candidates: List[Tuple[float, CapabilityNode, float]], goal: Goal
    ) -> List[Dict[str, Any]]:
        """Build sequential execution plan"""
        plan = [{"say": "Starting sequential execution..."}]

        for i, (_, capability, _) in enumerate(scored_candidates[:3]):
            plan.extend(
                [
                    {"say": f"Step {i + 1}: {capability.name}"},
                    {
                        "tool": capability.id,
                        "args": self.extract_args_from_goal(goal, capability),
                        "assign": f"step_{i + 1}_result",
                    },
                ]
            )

        plan.append({"say": "Sequential execution completed."})
        return plan

    def build_parallel_plan(
        self, scored_candidates: List[Tuple[float, CapabilityNode, float]], goal: Goal
    ) -> List[Dict[str, Any]]:
        """Build parallel execution plan"""
        return [
            {"say": "Starting parallel execution..."},
            # Note: Actual parallel execution would require special runner support
            {"say": "Parallel execution completed."},
        ]

    def build_delegate_plan(
        self, scored_candidates: List[Tuple[float, CapabilityNode, float]], goal: Goal
    ) -> List[Dict[str, Any]]:
        """Build delegation plan"""
        return [
            {"say": "Delegating to specialized agent..."},
            {"say": "Delegation completed."},
        ]

    def build_guardrail_plan(self, goal: Goal) -> List[Dict[str, Any]]:
        """Build safety guardrail plan"""
        return [
            {"say": "Safety check required for this operation."},
            {"say": "Please confirm you want to proceed with this action."},
            {"say": "Operation paused pending confirmation."},
        ]

    def extract_args_from_goal(
        self, goal: Goal, capability: CapabilityNode
    ) -> Dict[str, Any]:
        """Extract arguments for capability from goal slots"""
        args = {}

        # Map goal slots to capability input schema
        input_schema = capability.schema.get("input_schema", {})
        properties = input_schema.get("properties", {})

        for prop_name in properties.keys():
            if prop_name in goal.slots:
                args[prop_name] = goal.slots[prop_name]

        return args

    def calculate_plan_confidence(
        self,
        scored_candidates: List[Tuple[float, CapabilityNode, float]],
        strategy: StrategyType,
    ) -> float:
        """Calculate overall plan confidence"""
        if not scored_candidates:
            return 0.1

        # Base confidence on top candidate's utility
        base_confidence = min(0.9, scored_candidates[0][0])

        # Adjust based on strategy
        if strategy == StrategyType.SINGLE_BEST:
            return base_confidence
        elif strategy == StrategyType.SEQUENTIAL:
            return base_confidence * 0.8  # Lower due to dependencies
        elif strategy == StrategyType.PARALLEL:
            return base_confidence * 0.9  # Slightly lower due to coordination
        else:
            return 0.2  # Low confidence for guardrails/delegation

    def calculate_estimated_cost(
        self,
        scored_candidates: List[Tuple[float, CapabilityNode, float]],
        strategy: StrategyType,
    ) -> float:
        """Calculate estimated execution cost"""
        if not scored_candidates:
            return 0.0

        if strategy == StrategyType.SINGLE_BEST:
            return scored_candidates[0][1].cost_hint
        else:
            # Sum costs for multiple capabilities
            return sum(candidate.cost_hint for _, candidate, _ in scored_candidates[:3])

    def identify_risk_factors(
        self, scored_candidates: List[Tuple[float, CapabilityNode, float]], goal: Goal
    ) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []

        if goal.risk_level == RiskLevel.HIGH:
            risk_factors.append("high_risk_goal")

        for _, capability, _ in scored_candidates:
            if capability.side_effects:
                risk_factors.extend(capability.side_effects)

        return list(set(risk_factors))


# Example usage and integration
def create_bootstrap_capabilities() -> List[CapabilityNode]:
    """Create example capabilities for the bootstrap scenario"""
    return [
        CapabilityNode(
            id="git.clone_or_pull",
            name="Git Clone or Pull",
            description="Clone repository or pull latest changes",
            schema={
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "repo_url": {"type": "string"},
                        "path": {"type": "string"},
                    },
                }
            },
            preconditions=["git_available"],
            side_effects=["filesystem_modification"],
            cost_hint=0.1,
            registry_type="normal",
        ),
        CapabilityNode(
            id="env.ensure_dotenv",
            name="Environment Setup",
            description="Ensure .env file exists from template",
            schema={
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cwd": {"type": "string"},
                        "template": {"type": "string"},
                    },
                }
            },
            preconditions=["filesystem_access"],
            side_effects=["config_file_creation"],
            cost_hint=0.05,
            registry_type="normal",
        ),
        CapabilityNode(
            id="deps.install",
            name="Dependency Installation",
            description="Install Python dependencies from requirements files",
            schema={
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cwd": {"type": "string"},
                        "requirements": {"type": "array"},
                    },
                }
            },
            preconditions=["python_available", "pip_available"],
            side_effects=["package_installation"],
            cost_hint=0.3,
            registry_type="normal",
        ),
        CapabilityNode(
            id="runtime.start",
            name="Runtime Service Start",
            description="Start application runtime service",
            schema={
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "cwd": {"type": "string"},
                        "port": {"type": "integer"},
                    },
                }
            },
            preconditions=["service_available"],
            side_effects=["service_start", "port_binding"],
            cost_hint=0.2,
            registry_type="normal",
        ),
        CapabilityNode(
            id="http.get",
            name="HTTP Health Check",
            description="Perform HTTP GET request for health checking",
            schema={
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                }
            },
            preconditions=["network_access"],
            side_effects=[],
            cost_hint=0.01,
            registry_type="normal",
        ),
    ]


if __name__ == "__main__":
    # Example usage
    engine = DecisionPolicyEngine()

    # Register capabilities
    for capability in create_bootstrap_capabilities():
        engine.register_capability(capability)

    # Test decision making
    import asyncio

    async def test_decision():
        message = "get the newest version from the repo and spin it up https://github.com/yaboyshades/super-alita.git"
        ctx = {
            "git_available": True,
            "python_available": True,
            "pip_available": True,
            "filesystem_access": True,
            "network_access": True,
            "service_available": True,
        }

        plan = await engine.decide_and_plan(message, ctx)
        print(f"Generated plan: {json.dumps(plan.plan, indent=2)}")
        print(f"Strategy: {plan.strategy}")
        print(f"Confidence: {plan.confidence}")
        print(f"Estimated cost: {plan.estimated_cost}")

    asyncio.run(test_decision())
