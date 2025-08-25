"""
Multi-Armed Bandit Optimization Tests

Comprehensive tests for the bandit algorithms, policy engine, reward tracker, and optimization plugin.
"""

import asyncio
import pytest
import tempfile
import time
from typing import Dict, Any

from src.core.optimization.bandits import ThompsonSamplingBandit, UCB1Bandit, EpsilonGreedyBandit, BanditArm
from src.core.optimization.policy_engine import DecisionPolicyEngine, DecisionContext
from src.core.optimization.reward_tracker import RewardTracker, create_success_rate_rule, create_performance_rule
from src.core.optimization.plugin import OptimizationPlugin
from src.core.telemetry.simple_event_bus import SimpleEventBus


class TestBanditAlgorithms:
    """Test multi-armed bandit algorithms."""
    
    def test_thompson_sampling_bandit(self):
        """Test Thompson Sampling bandit algorithm."""
        bandit = ThompsonSamplingBandit()
        
        # Add test arms
        arm1 = bandit.add_arm("arm1", "Option A", {"type": "conservative"})
        arm2 = bandit.add_arm("arm2", "Option B", {"type": "aggressive"})
        
        assert len(bandit.arms) == 2
        assert arm1.arm_id == "arm1"
        assert arm2.name == "Option B"
        
        # Test decision making
        decision = bandit.select_arm({"test": True})
        assert decision.algorithm == "Thompson Sampling"
        assert decision.arm_id in ["arm1", "arm2"]
        assert 0.0 <= decision.confidence <= 1.0
        
        # Test reward updating
        success = bandit.update_reward(decision.decision_id, 0.8)
        assert success
        
        # Verify arm was updated
        selected_arm = bandit.arms[decision.arm_id]
        assert selected_arm.total_pulls == 1
        assert selected_arm.successes == 1
        assert selected_arm.last_reward == 0.8
        
        # Test statistics
        stats = bandit.get_statistics()
        assert stats["algorithm"] == "Thompson Sampling"
        assert stats["total_arms"] == 2
        assert stats["total_pulls"] == 1
        assert stats["total_decisions"] == 1
    
    def test_ucb1_bandit(self):
        """Test UCB1 bandit algorithm."""
        bandit = UCB1Bandit()
        
        # Add test arms
        bandit.add_arm("fast", "Fast Option")
        bandit.add_arm("slow", "Slow Option")
        bandit.add_arm("medium", "Medium Option")
        
        # First decisions should explore all arms
        decisions = []
        for i in range(3):
            decision = bandit.select_arm()
            decisions.append(decision)
            bandit.update_reward(decision.decision_id, 0.5)
        
        # Should have selected each arm once
        selected_arms = {d.arm_id for d in decisions}
        assert len(selected_arms) == 3
        
        # Next decision should favor best performing arm
        next_decision = bandit.select_arm()
        bandit.update_reward(next_decision.decision_id, 0.7)
        
        stats = bandit.get_statistics()
        assert stats["total_pulls"] == 4  # 3 initial + 1 final
    
    def test_epsilon_greedy_bandit(self):
        """Test Epsilon-Greedy bandit algorithm."""
        bandit = EpsilonGreedyBandit(epsilon=0.2)
        
        # Add arms with different performance
        bandit.add_arm("good", "Good Option")
        bandit.add_arm("bad", "Bad Option")
        
        # Train with known rewards
        for i in range(10):
            decision = bandit.select_arm()
            # Good arm gets high reward, bad arm gets low reward
            reward = 0.9 if decision.arm_id == "good" else 0.1
            bandit.update_reward(decision.decision_id, reward)
        
        # Test exploitation behavior
        exploit_decisions = []
        for i in range(20):
            decision = bandit.select_arm()
            exploit_decisions.append(decision)
            reward = 0.9 if decision.arm_id == "good" else 0.1
            bandit.update_reward(decision.decision_id, reward)
        
        # Should mostly select good arm (with some exploration)
        good_selections = sum(1 for d in exploit_decisions if d.arm_id == "good")
        assert good_selections >= 15  # At least 75% should be good arm
        
        stats = bandit.get_statistics()
        good_arm_stats = stats["arms"]["good"]
        assert good_arm_stats["success_rate"] > 0.8


class TestDecisionPolicyEngine:
    """Test decision policy engine."""
    
    @pytest.fixture
    def policy_engine(self):
        return DecisionPolicyEngine()
    
    def test_create_policy(self, policy_engine):
        """Test policy creation."""
        arms = [
            {"id": "option_a", "name": "Option A"},
            {"id": "option_b", "name": "Option B"},
            {"id": "option_c", "name": "Option C"}
        ]
        
        policy_id = policy_engine.create_policy(
            name="Test Policy",
            description="A test policy for demonstration",
            algorithm_type="thompson",
            arms=arms
        )
        
        assert policy_id in policy_engine.policies
        policy = policy_engine.get_policy(policy_id)
        assert policy.name == "Test Policy"
        assert policy.algorithm_type == "thompson"
        assert len(policy.arms) == 3
        
        # Test invalid algorithm type
        with pytest.raises(ValueError):
            policy_engine.create_policy(
                name="Invalid Policy",
                description="Should fail",
                algorithm_type="invalid",
                arms=arms
            )
    
    @pytest.mark.asyncio
    async def test_make_decision(self, policy_engine):
        """Test decision making."""
        arms = [
            {"id": "fast", "name": "Fast Approach"},
            {"id": "thorough", "name": "Thorough Approach"}
        ]
        
        policy_id = policy_engine.create_policy(
            name="Approach Policy",
            description="Choose between fast and thorough approaches",
            algorithm_type="ucb1",
            arms=arms
        )
        
        context = DecisionContext(
            session_id="test_session",
            user_id="test_user",
            task_type="optimization"
        )
        
        decision = await policy_engine.make_decision(policy_id, context)
        
        assert decision.policy_id == policy_id
        assert decision.bandit_decision.arm_id in ["fast", "thorough"]
        assert decision.context.session_id == "test_session"
        assert not decision.feedback_received
        
        # Test feedback
        feedback_success = await policy_engine.provide_feedback(decision.decision_id, 0.8)
        assert feedback_success
        assert decision.feedback_received
        assert decision.reward == 0.8
    
    def test_policy_statistics(self, policy_engine):
        """Test policy statistics."""
        arms = [{"id": "test_arm", "name": "Test Arm"}]
        policy_id = policy_engine.create_policy(
            name="Stats Test",
            description="Test statistics",
            algorithm_type="epsilon_greedy",
            arms=arms,
            epsilon=0.1
        )
        
        stats = policy_engine.get_policy_statistics(policy_id)
        assert stats is not None
        assert stats["policy"]["name"] == "Stats Test"
        assert stats["bandit"]["algorithm"] == "Epsilon-Greedy"
        assert stats["decisions"]["total"] == 0
        
        # Test global statistics
        global_stats = policy_engine.get_all_statistics()
        assert global_stats["engine"]["total_policies"] == 1
        assert policy_id in global_stats["policies"]


class TestRewardTracker:
    """Test reward tracking system."""
    
    @pytest.fixture
    def reward_tracker(self):
        return RewardTracker()
    
    @pytest.mark.asyncio
    async def test_record_reward(self, reward_tracker):
        """Test reward recording."""
        decision_id = "test_decision_123"
        
        reward_id = await reward_tracker.record_reward(
            decision_id=decision_id,
            reward_value=0.75,
            reward_type="immediate",
            source="test",
            metadata={"test": True}
        )
        
        assert reward_id is not None
        
        rewards = reward_tracker.get_decision_rewards(decision_id)
        assert len(rewards) == 1
        assert rewards[0].reward_value == 0.75
        assert rewards[0].source == "test"
        
        total_reward = reward_tracker.get_total_reward(decision_id)
        assert total_reward == 0.75
        
        latest_reward = reward_tracker.get_latest_reward(decision_id)
        assert latest_reward.reward_value == 0.75
    
    def test_reward_rules(self, reward_tracker):
        """Test automatic reward rules."""
        # Add success rate rule
        success_rule = create_success_rate_rule()
        reward_tracker.rules[success_rule.rule_id] = success_rule
        
        # Add performance rule
        perf_rule = create_performance_rule()
        reward_tracker.rules[perf_rule.rule_id] = perf_rule
        
        assert len(reward_tracker.rules) == 2
        
        stats = reward_tracker.get_rule_statistics()
        assert stats["total_rules"] == 2
        assert stats["active_rules"] == 2
        assert len(stats["rules"]) == 2
    
    @pytest.mark.asyncio
    async def test_automatic_rewards(self, reward_tracker):
        """Test automatic reward calculation."""
        # Add success rule
        success_rule = create_success_rate_rule()
        reward_tracker.rules[success_rule.rule_id] = success_rule
        
        decision_id = "auto_test_decision"
        
        # Test success context
        success_context = {"success": True}
        reward_ids = await reward_tracker.calculate_automatic_rewards(decision_id, success_context)
        
        assert len(reward_ids) == 1
        rewards = reward_tracker.get_decision_rewards(decision_id)
        assert len(rewards) == 1
        assert rewards[0].reward_value == 1.0  # Success should give max reward
        assert rewards[0].source == "rule:Success Rate"
        
        # Test failure context  
        decision_id_2 = "auto_test_decision_2"
        failure_context = {"error": True}
        reward_ids_2 = await reward_tracker.calculate_automatic_rewards(decision_id_2, failure_context)
        
        assert len(reward_ids_2) == 1
        rewards_2 = reward_tracker.get_decision_rewards(decision_id_2)
        assert rewards_2[0].reward_value == 0.0  # Error should give min reward


@pytest.mark.asyncio
class TestOptimizationPlugin:
    """Test optimization plugin integration."""
    
    async def test_plugin_setup_and_shutdown(self):
        """Test plugin setup and shutdown."""
        event_bus = SimpleEventBus()
        plugin = OptimizationPlugin()
        
        await plugin.setup(event_bus=event_bus)
        await plugin.start()
        assert plugin.event_bus is event_bus
        assert len(plugin.reward_tracker.rules) >= 2  # Default rules
        
        await plugin.shutdown()
        # Plugin should shutdown gracefully
    
    async def test_policy_creation_through_plugin(self):
        """Test creating policies through the plugin."""
        event_bus = SimpleEventBus()
        plugin = OptimizationPlugin()
        await plugin.setup(event_bus=event_bus)
        await plugin.start()
        
        arms = [
            {"id": "strategy_a", "name": "Strategy A"},
            {"id": "strategy_b", "name": "Strategy B"}
        ]
        
        policy_id = await plugin.create_policy(
            name="Strategy Policy",
            description="Choose between strategies",
            algorithm_type="thompson",
            arms=arms
        )
        
        assert policy_id in plugin.policy_engine.policies
        
        policies = plugin.get_policies()
        assert len(policies) == 1
        assert policies[0]["name"] == "Strategy Policy"
        
        await plugin.shutdown()
    
    async def test_decision_making_and_feedback(self):
        """Test making decisions and providing feedback."""
        event_bus = SimpleEventBus()
        plugin = OptimizationPlugin()
        await plugin.setup(event_bus=event_bus)
        await plugin.start()
        
        # Create policy
        arms = [
            {"id": "approach_1", "name": "Approach 1"},
            {"id": "approach_2", "name": "Approach 2"}
        ]
        
        policy_id = await plugin.create_policy(
            name="Approach Policy",
            description="Choose approach",
            algorithm_type="epsilon_greedy",
            arms=arms
        )
        
        # Make decision
        decision = await plugin.make_decision(
            policy_id=policy_id,
            session_id="test_session",
            user_id="test_user",
            task_type="test_task"
        )
        
        assert decision.policy_id == policy_id
        assert decision.bandit_decision.arm_id in ["approach_1", "approach_2"]
        
        # Provide feedback
        feedback_success = await plugin.provide_feedback(
            decision_id=decision.decision_id,
            reward=0.9,
            source="test_feedback"
        )
        assert feedback_success
        
        # Check statistics
        stats = plugin.get_global_statistics()
        assert stats["plugin_metrics"]["decisions_made"] == 1
        assert stats["plugin_metrics"]["rewards_collected"] == 1
        assert stats["plugin_metrics"]["policies_created"] == 1
        
        await plugin.shutdown()
    
    async def test_event_handling(self):
        """Test event-driven operation."""
        event_bus = SimpleEventBus()
        plugin = OptimizationPlugin()
        await plugin.setup(event_bus=event_bus)
        await plugin.start()
        
        # Create a policy for the plugin to use
        arms = [{"id": "auto_arm", "name": "Auto Arm"}]
        policy_id = await plugin.create_policy(
            name="Auto Policy",
            description="Automatic decision policy",
            algorithm_type="ucb1",
            arms=arms
        )
        
        # Simulate a task completion event for reward calculation
        from src.core.events import create_event
        task_event = create_event(
            "task_completed",
            source_plugin="TestPlugin",
            decision_id="test_decision_123",
            success=True,
            performance_score=0.8
        )
        
        # Emit event directly via simple event bus
        await event_bus.emit("task_completed", 
                            decision_id="test_decision_123",
                            success=True,
                            performance_score=0.8)
        
        # Give some time for event processing
        await asyncio.sleep(0.1)
        
        await plugin.shutdown()


async def run_optimization_tests():
    """Run all optimization tests."""
    print("🎯 Running Multi-Armed Bandit Optimization Tests")
    
    # Test bandit algorithms
    print("Testing Bandit Algorithms...")
    test_bandits = TestBanditAlgorithms()
    test_bandits.test_thompson_sampling_bandit()
    test_bandits.test_ucb1_bandit()
    test_bandits.test_epsilon_greedy_bandit()
    print("  ✓ Bandit algorithms tests passed")
    
    # Test policy engine
    print("Testing Policy Engine...")
    test_engine = TestDecisionPolicyEngine()
    
    # Test policy creation
    engine1 = DecisionPolicyEngine()
    test_engine.test_create_policy(engine1)
    
    # Test decision making  
    engine2 = DecisionPolicyEngine()
    await test_engine.test_make_decision(engine2)
    
    # Test statistics
    engine3 = DecisionPolicyEngine()
    test_engine.test_policy_statistics(engine3)
    
    print("  ✓ Policy engine tests passed")
    
    # Test reward tracker
    print("Testing Reward Tracker...")
    test_rewards = TestRewardTracker()
    tracker = RewardTracker()
    await test_rewards.test_record_reward(tracker)
    test_rewards.test_reward_rules(tracker)
    await test_rewards.test_automatic_rewards(tracker)
    print("  ✓ Reward tracker tests passed")
    
    # Test optimization plugin
    print("Testing Optimization Plugin...")
    test_plugin = TestOptimizationPlugin()
    await test_plugin.test_plugin_setup_and_shutdown()
    await test_plugin.test_policy_creation_through_plugin()
    await test_plugin.test_decision_making_and_feedback()
    await test_plugin.test_event_handling()
    print("  ✓ Optimization plugin tests passed")
    
    print("✅ All optimization tests passed!")


if __name__ == "__main__":
    asyncio.run(run_optimization_tests())