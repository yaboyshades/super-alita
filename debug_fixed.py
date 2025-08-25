#!/usr/bin/env python3
"""Fixed debug script with realistic capability parameters"""

import logging
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.decision_policy_v1 import (
    CapabilityNode,
    DecisionPolicyEngine,
    Goal,
    IntentType,
    PolicyConfig,
    RiskLevel,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_realistic_capabilities() -> dict[str, CapabilityNode]:
    """Create realistic capabilities with proper cost/latency values"""

    capabilities = {}

    # Git tools - low cost, medium latency
    capabilities["git_tools"] = CapabilityNode(
        id="git_tools",
        name="Git Repository Tools",
        description="Clone, manage, and analyze git repositories",
        schema={
            "input_schema": {
                "properties": {
                    "repo_url": {"type": "string"},
                    "branch": {"type": "string"},
                }
            }
        },
        preconditions=["git_available"],
        side_effects=["file_system_changes"],
        circuit_open=False,
        cost_hint=0.1,  # Low cost
        avg_latency=0.3,  # Medium latency
        attempts=5,
        wins=4,  # Good success rate
    )

    # Service monitor - very low cost, low latency
    capabilities["service_monitor"] = CapabilityNode(
        id="service_monitor",
        name="Service Status Monitor",
        description="Monitor and check system service status",
        schema={
            "input_schema": {
                "properties": {
                    "service_name": {"type": "string"},
                    "timeout": {"type": "number"},
                }
            }
        },
        preconditions=["system_access"],
        side_effects=[],
        circuit_open=False,
        cost_hint=0.05,  # Very low cost
        avg_latency=0.1,  # Low latency
        attempts=10,
        wins=9,  # Excellent success rate
    )

    # Code analyzer - medium cost, high latency
    capabilities["code_analyzer"] = CapabilityNode(
        id="code_analyzer",
        name="Code Quality Analyzer",
        description="Perform static code analysis and review",
        schema={
            "input_schema": {
                "properties": {
                    "file_path": {"type": "string"},
                    "language": {"type": "string"},
                }
            }
        },
        preconditions=["file_access"],
        side_effects=["temp_files"],
        circuit_open=False,
        cost_hint=0.3,  # Medium cost
        avg_latency=0.8,  # High latency
        attempts=8,
        wins=6,  # Good success rate
    )

    return capabilities


def test_realistic_scenarios() -> None:
    """Test scenarios with realistic capabilities"""

    capabilities = create_realistic_capabilities()

    # Create engine with adjusted config
    config = PolicyConfig()
    config.min_utility = 0.05  # Lower threshold
    config.beta_cost = 0.5  # Reduce cost penalty
    config.alpha_latency = 0.1  # Reduce latency penalty

    engine = DecisionPolicyEngine(config=config)
    engine.capabilities = capabilities

    scenarios = [
        {
            "name": "git_clone",
            "goal": Goal(
                description="clone a git repository from github",
                intent=IntentType.CREATE,
                slots={"repo_url": "https://github.com/user/repo"},
                risk_level=RiskLevel.LOW,
                success_criteria=["repository_cloned"],
                constraints=["timeout_60s"],
            ),
        },
        {
            "name": "service_check",
            "goal": Goal(
                description="check if nginx service is running",
                intent=IntentType.MONITOR,
                slots={"service_name": "nginx"},
                risk_level=RiskLevel.LOW,
                success_criteria=["status_retrieved"],
                constraints=["quick_response"],
            ),
        },
        {
            "name": "code_analysis",
            "goal": Goal(
                description="analyze Python code for quality issues",
                intent=IntentType.ANALYZE,
                slots={"file_path": "/src/main.py", "language": "python"},
                risk_level=RiskLevel.MEDIUM,
                success_criteria=["analysis_complete"],
                constraints=["detailed_report"],
            ),
        },
    ]

    logger.info("🧪 REALISTIC SCENARIO TESTING")
    logger.info("=" * 50)

    for scenario in scenarios:
        goal = scenario["goal"]
        logger.info(f"🎯 Testing: {scenario['name']}")
        logger.info(f"   Goal: {goal.description}")

        # Run decision process
        plan = engine.decide(goal.description)

        logger.info(f"   ✅ Strategy: {plan.strategy}")
        logger.info(f"   ✅ Confidence: {plan.confidence:.3f}")
        logger.info(f"   ✅ Cost: {plan.estimated_cost:.3f}")
        logger.info(f"   ✅ Plan steps: {len(plan.plan)}")
        logger.info("")


def debug_utility_calculation() -> None:
    """Debug utility calculation in detail"""

    capability = CapabilityNode(
        id="test_tool",
        name="Test Tool",
        description="A test tool for debugging",
        schema={"input_schema": {"properties": {}}},
        preconditions=[],
        side_effects=[],
        circuit_open=False,
        cost_hint=0.1,  # Low cost
        avg_latency=0.2,  # Low latency
        attempts=5,
        wins=4,  # 80% success rate
    )

    goal = Goal(
        description="test goal",
        intent=IntentType.CREATE,
        slots={},
        risk_level=RiskLevel.LOW,
        success_criteria=["test_complete"],
        constraints=[],
    )

    config = PolicyConfig()
    calc = DecisionPolicyEngine(config=config).utility_calculator

    logger.info("🔍 UTILITY CALCULATION DEBUG")
    logger.info("=" * 40)

    # Calculate each component
    p_success = calc.estimate_success_probability(capability)
    latency = capability.avg_latency or 1.0
    cost = capability.cost_hint or 0.1
    risk_penalty = calc.calculate_risk_penalty(capability, goal.risk_level)
    explore_bonus = calc.calculate_exploration_bonus(capability)
    reward = 1.0

    logger.info(f"📊 Success probability: {p_success:.3f}")
    logger.info(f"⏱️  Latency: {latency:.3f}")
    logger.info(f"💰 Cost: {cost:.3f}")
    logger.info(f"⚠️  Risk penalty: {risk_penalty:.3f}")
    logger.info(f"🎲 Explore bonus: {explore_bonus:.3f}")
    logger.info(f"🏆 Reward: {reward:.3f}")

    # Calculate utility components
    success_component = p_success * reward
    latency_penalty = config.alpha_latency * latency
    cost_penalty = config.beta_cost * cost
    risk_component = config.gamma_risk * risk_penalty

    utility = (
        success_component
        - latency_penalty
        - cost_penalty
        - risk_component
        + explore_bonus
    )

    logger.info("")
    logger.info("💎 UTILITY BREAKDOWN:")
    logger.info(f"   Success component:  {success_component:.3f}")
    logger.info(f"   Latency penalty:   -{latency_penalty:.3f}")
    logger.info(f"   Cost penalty:      -{cost_penalty:.3f}")
    logger.info(f"   Risk component:    -{risk_component:.3f}")
    logger.info(f"   Explore bonus:     +{explore_bonus:.3f}")
    logger.info(f"   Total utility:      {utility:.3f}")


if __name__ == "__main__":
    logger.info("🚀 Fixed Decision Policy Debug")
    logger.info("=" * 50)

    debug_utility_calculation()
    print()
    test_realistic_scenarios()

    logger.info("\n🎉 Debug completed!")
