#!/usr/bin/env python3
"""Final diagnostic to understand exact matching process"""

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


def diagnose_exact_matching() -> None:
    """Diagnose the exact matching process step by step"""

    # Setup engine with very permissive thresholds
    config = PolicyConfig()
    config.min_match = 0.1  # Very low match threshold
    config.min_utility = 0.01  # Very low utility threshold
    config.beta_cost = 0.1  # Very low cost penalty

    engine = DecisionPolicyEngine(config=config)

    # Create a simple capability that should match
    capability = CapabilityNode(
        id="git_tool",
        name="Git Tool",
        description="git repository clone manage",
        schema={
            "input_schema": {
                "properties": {
                    "repo_url": {"type": "string"},
                }
            }
        },
        preconditions=[],
        side_effects=[],
        circuit_open=False,
        cost_hint=0.05,
        avg_latency=0.1,
        attempts=1,
        wins=1,
    )

    # Add capability to engine
    engine.capabilities = {"git_tool": capability}

    # Create goal that should match
    goal = Goal(
        description="clone git repository",
        intent=IntentType.CREATE,
        slots={"repo_url": "https://github.com/test/repo"},
        risk_level=RiskLevel.LOW,
        success_criteria=["complete"],
        constraints=[],
    )

    logger.info("🔍 EXACT MATCHING DIAGNOSIS")
    logger.info("=" * 50)
    logger.info(f"Goal: '{goal.description}'")
    logger.info(f"Capability: '{capability.description}'")
    logger.info(
        f"Config: min_match={config.min_match}, min_utility={config.min_utility}"
    )
    logger.info("")

    # Step 1: Text similarity
    text_sim = engine.text_similarity(goal.description, capability.description)
    logger.info(f"1️⃣ Text similarity: {text_sim:.3f}")

    # Check words overlap
    goal_words = set(goal.description.lower().split())
    cap_words = set(capability.description.lower().split())
    overlap = goal_words.intersection(cap_words)
    union = goal_words.union(cap_words)
    logger.info(f"   Goal words: {goal_words}")
    logger.info(f"   Cap words: {cap_words}")
    logger.info(f"   Overlap: {overlap}")
    logger.info(f"   Union: {union}")
    logger.info(
        f"   Calculated: {len(overlap)}/{len(union)} = {len(overlap) / len(union):.3f}"
    )

    # Step 2: Resolve candidates
    candidates = engine.resolve_candidates(goal)
    logger.info(f"2️⃣ Candidates found: {len(candidates)}")

    if not candidates:
        logger.info("❌ No candidates found - checking thresholds:")
        logger.info(f"   Text similarity {text_sim:.3f} > 0.3? {text_sim > 0.3}")

        # Check schema compatibility
        schema_compat = engine.schema_compatible(goal, capability)
        logger.info(f"   Schema compatible? {schema_compat}")

        return

    # Step 3: Calculate match score
    candidate = candidates[0]
    match_score = engine.calculate_match_score(candidate, goal, {})
    logger.info(f"3️⃣ Match score: {match_score:.3f}")
    logger.info(
        f"   Passes min_match {config.min_match}? {match_score >= config.min_match}"
    )

    # Step 4: Calculate utility
    utility = engine.utility_calculator.calculate(candidate, goal, {})
    logger.info(f"4️⃣ Utility: {utility:.3f}")
    logger.info(
        f"   Passes min_utility {config.min_utility}? {utility >= config.min_utility}"
    )

    # Step 5: Full decision process
    logger.info("\n5️⃣ FULL DECISION PROCESS")
    logger.info("-" * 30)

    # Extract slots
    slots = engine.extract_slots(goal.description)
    logger.info(f"Extracted slots: {slots}")

    # Create final goal with extracted slots
    final_goal = Goal(
        description=goal.description,
        intent=IntentType.CREATE,
        slots=slots,
        risk_level=RiskLevel.LOW,
        success_criteria=["complete"],
        constraints=[],
    )

    # Test the full pipeline
    candidates = engine.resolve_candidates(final_goal)
    logger.info(f"Final candidates: {len(candidates)}")

    if candidates:
        for i, cand in enumerate(candidates):
            match_score = engine.calculate_match_score(cand, final_goal, {})
            utility = engine.utility_calculator.calculate(cand, final_goal, {})
            logger.info(
                f"  Candidate {i + 1}: match={match_score:.3f}, utility={utility:.3f}"
            )


def test_alternative_text_similarity() -> None:
    """Test with different text similarity approaches"""

    engine = DecisionPolicyEngine()

    test_pairs = [
        ("clone git repository", "git repository clone manage"),
        ("clone git", "git clone"),
        ("repository clone", "clone repository"),
        ("git", "git"),
    ]

    logger.info("\n🔤 TEXT SIMILARITY TESTS")
    logger.info("=" * 40)

    for goal_text, cap_text in test_pairs:
        sim = engine.text_similarity(goal_text, cap_text)
        logger.info(f"'{goal_text}' vs '{cap_text}': {sim:.3f}")


if __name__ == "__main__":
    logger.info("🚀 Final Matching Diagnosis")
    logger.info("=" * 50)

    diagnose_exact_matching()
    test_alternative_text_similarity()

    logger.info("\n🎉 Diagnosis completed!")
