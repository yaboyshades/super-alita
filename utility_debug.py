#!/usr/bin/env python3
"""Simple debug to understand utility calculation"""

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


def test_utility_with_different_costs() -> None:
    """Test utility calculation with different cost values"""

    goal = Goal(
        description="test task",
        intent=IntentType.CREATE,
        slots={},
        risk_level=RiskLevel.LOW,
        success_criteria=["complete"],
        constraints=[],
    )

    config = PolicyConfig()
    engine = DecisionPolicyEngine(config=config)

    cost_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    logger.info("🧪 UTILITY vs COST ANALYSIS")
    logger.info("=" * 50)
    logger.info(
        f"Config: alpha_latency={config.alpha_latency}, beta_cost={config.beta_cost}, gamma_risk={config.gamma_risk}"
    )
    logger.info(f"Min utility threshold: {config.min_utility}")
    logger.info("")

    for cost in cost_values:
        capability = CapabilityNode(
            id=f"tool_cost_{cost}",
            name=f"Tool with cost {cost}",
            description="Test tool for cost analysis",
            schema={"input_schema": {"properties": {}}},
            preconditions=[],
            side_effects=[],
            circuit_open=False,
            cost_hint=cost,
            avg_latency=0.2,  # Fixed low latency
            attempts=5,
            wins=4,  # 80% success rate
        )

        utility = engine.utility_calculator.calculate(capability, goal, {})
        passes_threshold = utility >= config.min_utility

        logger.info(
            f"Cost {cost:4.2f}: Utility {utility:6.3f} {'✅' if passes_threshold else '❌'}"
        )

    logger.info("")
    logger.info("🔧 TESTING WITH ADJUSTED CONFIG")
    logger.info("=" * 40)

    # Test with reduced cost penalty
    config.beta_cost = 0.3  # Reduce from 1.0 to 0.3
    engine = DecisionPolicyEngine(config=config)

    logger.info(f"Adjusted beta_cost to {config.beta_cost}")
    logger.info("")

    for cost in cost_values:
        capability = CapabilityNode(
            id=f"tool_cost_{cost}",
            name=f"Tool with cost {cost}",
            description="Test tool for cost analysis",
            schema={"input_schema": {"properties": {}}},
            preconditions=[],
            side_effects=[],
            circuit_open=False,
            cost_hint=cost,
            avg_latency=0.2,
            attempts=5,
            wins=4,
        )

        utility = engine.utility_calculator.calculate(capability, goal, {})
        passes_threshold = utility >= config.min_utility

        logger.info(
            f"Cost {cost:4.2f}: Utility {utility:6.3f} {'✅' if passes_threshold else '❌'}"
        )


if __name__ == "__main__":
    logger.info("🚀 Utility vs Cost Analysis")
    logger.info("=" * 50)

    test_utility_with_different_costs()

    logger.info("\n🎉 Analysis completed!")
