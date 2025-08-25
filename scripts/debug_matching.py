#!/usr/bin/env python3
"""Debug script to understand matching and utility calculation issues"""

import logging
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).resolve().parents[1] / "src"
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


def debug_single_match() -> None:
    """Debug a single goal-capability match in detail"""

    # Create a simple capability
    capability = CapabilityNode(
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
        cost_hint=2.0,
    )

    # Create a matching goal
    goal = Goal(
        description="clone a git repository from github",
        intent=IntentType.CREATE,
        slots={"repo_url": "https://github.com/user/repo"},
        risk_level=RiskLevel.MEDIUM,
        success_criteria=["repository_cloned", "files_accessible"],
        constraints=["timeout_60s", "disk_space_required"],
    )

    # Create engine with verbose config
    config = PolicyConfig()
    config.min_utility = 0.05  # Lower threshold for debugging
    config.min_match = 0.1  # Lower threshold for debugging

    engine = DecisionPolicyEngine(config=config)
    engine.capabilities = {"git_tools": capability}

    # Debug each component
    logger.info("🔍 DEBUGGING MATCH CALCULATION")
    logger.info(f"Goal: {goal.description}")
    logger.info(f"Capability: {capability.description}")

    # 1. Text similarity
    text_sim = engine.text_similarity(goal.description, capability.description)
    logger.info(f"📝 Text similarity: {text_sim:.3f}")

    # 2. Schema fitness
    schema_fit = engine.schema_fitness(goal, capability)
    logger.info(f"🏗️  Schema fitness: {schema_fit:.3f}")

    # 3. Precondition satisfaction (mock context)
    ctx = {"git_available": True}
    precond_sat = engine.precondition_satisfaction(capability, ctx)
    logger.info(f"✅ Precondition satisfaction: {precond_sat:.3f}")

    # 4. Historical success
    historical = engine.historical_success(capability)
    logger.info(f"📊 Historical success: {historical:.3f}")

    # 5. Risk penalty
    risk_penalty = engine.risk_penalty(capability, goal.risk_level)
    logger.info(f"⚠️  Risk penalty: {risk_penalty:.3f}")

    # 6. Overall match score
    match_score = engine.calculate_match_score(capability, goal, ctx)
    logger.info(f"🎯 Overall match score: {match_score:.3f}")

    # 7. Utility calculation
    utility = engine.utility_calculator.calculate(capability, goal, ctx)
    logger.info(f"💎 Utility score: {utility:.3f}")

    # 8. Component breakdown
    logger.info("\n📋 COMPONENT BREAKDOWN:")
    logger.info(
        f"  Schema fit weight: {config.w_schema_fit} × {schema_fit:.3f} = {config.w_schema_fit * schema_fit:.3f}"
    )
    logger.info(
        f"  Text sim weight:   {config.w_text_sim} × {text_sim:.3f} = {config.w_text_sim * text_sim:.3f}"
    )
    logger.info(
        f"  Precond weight:    {config.w_precond} × {precond_sat:.3f} = {config.w_precond * precond_sat:.3f}"
    )
    logger.info(
        f"  History weight:    {config.w_history} × {historical:.3f} = {config.w_history * historical:.3f}"
    )
    logger.info(
        f"  Risk penalty:      {config.w_risk} × {risk_penalty:.3f} = {config.w_risk * risk_penalty:.3f}"
    )

    total_positive = (
        config.w_schema_fit * schema_fit
        + config.w_text_sim * text_sim
        + config.w_precond * precond_sat
        + config.w_history * historical
    )
    total_penalty = config.w_risk * risk_penalty
    raw_score = total_positive - total_penalty

    logger.info(f"  Total positive:    {total_positive:.3f}")
    logger.info(f"  Total penalty:     {total_penalty:.3f}")
    logger.info(f"  Raw score:         {raw_score:.3f}")
    logger.info(f"  Clamped score:     {max(0.0, min(1.0, raw_score)):.3f}")


def debug_text_similarity() -> None:
    """Debug text similarity function with various examples"""

    engine = DecisionPolicyEngine()

    test_cases = [
        ("clone a git repository", "Clone, manage, and analyze git repositories"),
        ("check service status", "Monitor and check system service status"),
        ("analyze code quality", "Perform static code analysis and review"),
        ("update configuration file", "Edit and update system configuration files"),
        ("search the web", "Search and retrieve web content"),
        ("deploy in parallel", "Deploy services across multiple environments"),
    ]

    logger.info("🔍 TEXT SIMILARITY DEBUG")
    for goal_desc, cap_desc in test_cases:
        sim = engine.text_similarity(goal_desc, cap_desc)
        logger.info(f"  '{goal_desc}' vs '{cap_desc}': {sim:.3f}")


if __name__ == "__main__":
    logger.info("🚀 Decision Policy Matching Debug")
    logger.info("=" * 50)

    debug_text_similarity()
    print()
    debug_single_match()

    logger.info("\n🎉 Debug completed!")
