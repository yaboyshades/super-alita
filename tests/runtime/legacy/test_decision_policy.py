#!/usr/bin/env python3
"""
Test the Decision Policy v1 implementation with various scenarios.
"""
import pytest
pytest.skip("legacy test", allow_module_level=True)

import asyncio
import json

from core.decision_policy_v1 import (
    DecisionPolicyEngine,
    IntentType,
    create_bootstrap_capabilities,
)

async def test_bootstrap_scenario():
    """Test the bootstrap scenario with enhanced text matching"""
    print("=== Testing Bootstrap Scenario ===")

    engine = DecisionPolicyEngine()

    # Register capabilities
    for capability in create_bootstrap_capabilities():
        engine.register_capability(capability)

    # Test message
    message = "get the newest version from the repo and spin it up https://github.com/yaboyshades/super-alita.git"

    # Context with all preconditions satisfied
    ctx = {
        "git_available": True,
        "python_available": True,
        "pip_available": True,
        "filesystem_access": True,
        "network_access": True,
        "service_available": True,
    }

    # Generate plan
    plan = await engine.decide_and_plan(message, ctx)

    print(f"Intent Classification: {plan.strategy}")
    print(f"Plan Confidence: {plan.confidence:.2f}")
    print(f"Estimated Cost: {plan.estimated_cost:.2f}")
    print(f"Risk Factors: {plan.risk_factors}")
    print("Generated Plan:")
    for i, step in enumerate(plan.plan, 1):
        print(f"  {i}. {step}")
    print()

async def test_query_scenario():
    """Test a query scenario"""
    print("=== Testing Query Scenario ===")

    engine = DecisionPolicyEngine()

    # Register capabilities
    for capability in create_bootstrap_capabilities():
        engine.register_capability(capability)

    message = "show me the status of the running services"
    ctx = {"network_access": True}

    plan = await engine.decide_and_plan(message, ctx)

    print(f"Strategy: {plan.strategy}")
    print(f"Confidence: {plan.confidence:.2f}")
    print(f"Plan: {json.dumps(plan.plan, indent=2)}")
    print()

async def test_intent_classification():
    """Test intent classification on various messages"""
    print("=== Testing Intent Classification ===")

    engine = DecisionPolicyEngine()
    classifier = engine.intent_classifier

    test_messages = [
        "setup the development environment",
        "clone the repository and install dependencies",
        "show me the current status",
        "find all Python files in the project",
        "create a new configuration file",
        "update the database schema",
        "analyze the code quality",
        "check test coverage",
        "this is unclear what I want",
    ]

    for msg in test_messages:
        intent = classifier.classify(msg)
        print(f"'{msg}' -> {intent.name}")
    print()

async def test_capability_matching():
    """Test capability matching logic"""
    print("=== Testing Capability Matching ===")

    engine = DecisionPolicyEngine()

    # Register capabilities
    for capability in create_bootstrap_capabilities():
        engine.register_capability(capability)

    # Test goal synthesis and candidate resolution
    intent = IntentType.BOOTSTRAP
    slots = {"repo_url": "https://github.com/test/repo.git"}
    goal = engine.goal_synthesizer.synthesize(intent, slots, {})

    print(f"Synthesized Goal: {goal.description}")
    print(f"Goal Slots: {goal.slots}")
    print(f"Success Criteria: {goal.success_criteria}")
    print(f"Risk Level: {goal.risk_level}")

    candidates = engine.resolve_candidates(goal)
    print(f"Found {len(candidates)} candidate capabilities:")

    for candidate in candidates:
        match_score = engine.calculate_match_score(candidate, goal, {})
        utility = engine.utility_calculator.calculate(candidate, goal, {})
        print(f"  - {candidate.name}: match={match_score:.2f}, utility={utility:.2f}")
    print()

async def test_bandit_learning():
    """Test multi-armed bandit learning updates"""
    print("=== Testing Bandit Learning ===")

    engine = DecisionPolicyEngine()

    # Simulate successful and failed executions
    capability_id = "git.clone_or_pull"

    # Record some attempts
    engine.update_bandit_stats(capability_id, success=True, cost=0.1, latency=2.0)
    engine.update_bandit_stats(capability_id, success=True, cost=0.1, latency=1.5)
    engine.update_bandit_stats(capability_id, success=False, cost=0.2, latency=5.0)
    engine.update_bandit_stats(capability_id, success=True, cost=0.1, latency=1.8)

    stats = engine.bandit_stats[capability_id]
    print(f"Capability: {capability_id}")
    print(f"Attempts: {stats['attempts']}")
    print(f"Wins: {stats['wins']}")
    print(f"Win Rate: {stats['wins'] / stats['attempts']:.1%}")
    print(f"Average Reward: {stats['avg_reward']:.3f}")
    print()

async def main():
    """Run all tests"""
    await test_intent_classification()
    await test_capability_matching()
    await test_bootstrap_scenario()
    await test_query_scenario()
    await test_bandit_learning()

    print("=== Decision Policy v1 Testing Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
