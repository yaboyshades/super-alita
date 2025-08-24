#!/usr/bin/env python3
"""
Enhanced Decision Policy Trials with Fixed Utility Calculation
"""

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


class ImprovedDecisionTrials:
    """Enhanced decision trials with realistic utility calculations"""

    def __init__(self):
        self.results = []
        self.engine = None

    def setup_improved_engine(self) -> DecisionPolicyEngine:
        """Setup engine with improved configuration and realistic capabilities"""

        # Create optimized config
        config = PolicyConfig()
        config.beta_cost = 0.3  # Reduce cost penalty
        config.alpha_latency = 0.1  # Reduce latency penalty
        config.min_utility = 0.05  # Lower utility threshold
        config.min_match = 0.2  # Lower match threshold

        engine = DecisionPolicyEngine(config=config)

        # Create realistic capabilities with proper cost/latency values
        capabilities = {
            "git_tools": CapabilityNode(
                id="git_tools",
                name="Git Repository Tools",
                description="Clone manage analyze git repositories version control",
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
                cost_hint=0.15,  # Low cost
                avg_latency=0.3,  # Medium latency
                attempts=8,
                wins=7,  # Good success rate
            ),
            "service_monitor": CapabilityNode(
                id="service_monitor",
                name="Service Status Monitor",
                description="Monitor check system service status health monitoring",
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
                attempts=15,
                wins=14,  # Excellent success rate
            ),
            "code_analyzer": CapabilityNode(
                id="code_analyzer",
                name="Code Quality Analyzer",
                description="Analyze code quality static analysis review inspect",
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
                cost_hint=0.25,  # Medium cost
                avg_latency=0.5,  # Higher latency
                attempts=6,
                wins=5,  # Good success rate
            ),
            "config_editor": CapabilityNode(
                id="config_editor",
                name="Configuration File Editor",
                description="Edit update modify configuration files settings",
                schema={
                    "input_schema": {
                        "properties": {
                            "file_path": {"type": "string"},
                            "updates": {"type": "object"},
                        }
                    }
                },
                preconditions=["file_write_access"],
                side_effects=["config_changes"],
                circuit_open=False,
                cost_hint=0.1,  # Low cost
                avg_latency=0.2,  # Low latency
                attempts=10,
                wins=9,  # Excellent success rate
            ),
            "web_search": CapabilityNode(
                id="web_search",
                name="Web Search Engine",
                description="Search web internet content retrieve information",
                schema={
                    "input_schema": {
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "number"},
                        }
                    }
                },
                preconditions=["internet_access"],
                side_effects=["network_requests"],
                circuit_open=False,
                cost_hint=0.08,  # Low cost
                avg_latency=0.4,  # Medium latency
                attempts=12,
                wins=10,  # Good success rate
            ),
            "deployment_manager": CapabilityNode(
                id="deployment_manager",
                name="Multi-Environment Deployment",
                description="Deploy parallel services multiple environments orchestration",
                schema={
                    "input_schema": {
                        "properties": {
                            "services": {"type": "array"},
                            "environments": {"type": "array"},
                        }
                    }
                },
                preconditions=["deployment_access", "orchestration_ready"],
                side_effects=["service_changes", "resource_allocation"],
                circuit_open=False,
                cost_hint=0.4,  # Higher cost but reasonable
                avg_latency=0.8,  # High latency
                attempts=4,
                wins=3,  # Good success rate
            ),
        }

        engine.capabilities = capabilities
        logger.info(f"✅ Engine setup with {len(capabilities)} capabilities")
        return engine

    def run_improved_trial(
        self,
        scenario_name: str,
        goal_description: str,
        intent: IntentType,
        risk_level: RiskLevel,
    ) -> dict:
        """Run a single trial with improved matching"""

        goal = Goal(
            description=goal_description,
            intent=intent,
            slots=self.engine.extract_slots(goal_description),
            risk_level=risk_level,
            success_criteria=["task_complete"],
            constraints=["reasonable_time"],
        )

        # Get candidates and calculate utilities
        candidates = self.engine.resolve_candidates(goal)
        scored_candidates = []

        for candidate in candidates:
            match_score = self.engine.calculate_match_score(candidate, goal, {})
            if match_score < self.engine.config.min_match:
                continue

            utility = self.engine.utility_calculator.calculate(candidate, goal, {})
            if utility < self.engine.config.min_utility:
                continue

            scored_candidates.append((utility, candidate, match_score))

        if not scored_candidates:
            plan = self.engine.safe_fallback_plan(goal, None)
            strategy = plan.strategy
            confidence = plan.confidence
            cost = plan.estimated_cost
        else:
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            strategy = self.engine.pick_strategy(scored_candidates, goal)
            plan = self.engine.plan_builder.build_plan(
                strategy, scored_candidates, goal, None
            )
            confidence = plan.confidence
            cost = plan.estimated_cost

        result = {
            "scenario": scenario_name,
            "strategy": strategy.name,
            "confidence": confidence,
            "cost": cost,
            "steps": len(plan.plan),
            "candidates_found": len(candidates),
            "candidates_qualified": len(scored_candidates),
        }

        logger.info(
            f"✅ {scenario_name}: {strategy.name} strategy, confidence {confidence:.2f}, {len(scored_candidates)} qualified candidates"
        )
        return result

    def run_comprehensive_improved_trials(self) -> None:
        """Run comprehensive trials with improved matching"""

        self.engine = self.setup_improved_engine()

        scenarios = [
            (
                "git_repository_clone",
                "clone a git repository from github",
                IntentType.CREATE,
                RiskLevel.LOW,
            ),
            (
                "service_status_check",
                "check if nginx service is running",
                IntentType.MONITOR,
                RiskLevel.LOW,
            ),
            (
                "code_quality_analysis",
                "analyze Python code for quality issues",
                IntentType.ANALYZE,
                RiskLevel.MEDIUM,
            ),
            (
                "config_file_update",
                "update database configuration file",
                IntentType.MODIFY,
                RiskLevel.MEDIUM,
            ),
            (
                "web_content_search",
                "search the web for documentation",
                IntentType.QUERY,
                RiskLevel.LOW,
            ),
            (
                "parallel_service_deployment",
                "deploy services to multiple environments in parallel",
                IntentType.CREATE,
                RiskLevel.HIGH,
            ),
            (
                "system_health_monitoring",
                "monitor system health and performance",
                IntentType.MONITOR,
                RiskLevel.LOW,
            ),
            (
                "code_refactoring_task",
                "refactor legacy code for better maintainability",
                IntentType.MODIFY,
                RiskLevel.MEDIUM,
            ),
        ]

        logger.info("🧪 Running comprehensive improved trials")

        for scenario_name, description, intent, risk in scenarios:
            logger.info(f"🧪 Trial: {scenario_name}")
            result = self.run_improved_trial(scenario_name, description, intent, risk)
            self.results.append(result)

        self.generate_improved_summary()

    def generate_improved_summary(self) -> None:
        """Generate comprehensive trial summary"""

        if not self.results:
            logger.info("❌ No results to summarize")
            return

        strategy_counts = {}
        total_confidence = 0
        total_cost = 0

        for result in self.results:
            strategy = result["strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            total_confidence += result["confidence"]
            total_cost += result["cost"]

        avg_confidence = total_confidence / len(self.results)
        avg_cost = total_cost / len(self.results)

        print("\n" + "=" * 60)
        print("📊 IMPROVED DECISION POLICY TRIAL RESULTS")
        print("=" * 60)
        print(f"🎯 Total Scenarios: {len(self.results)}")
        print(f"⚡ Average Confidence: {avg_confidence:.2f}")
        print(f"💰 Average Cost: {avg_cost:.2f}")
        print(f"📈 Strategy Distribution: {dict(strategy_counts)}")
        print()
        print("📋 Detailed Results:")

        for result in self.results:
            print(
                f"  {result['scenario']}: {result['strategy']} "
                f"(confidence: {result['confidence']:.2f}, "
                f"cost: {result['cost']:.2f}, "
                f"candidates: {result['candidates_qualified']}/{result['candidates_found']})"
            )

        print("=" * 60)


def main():
    """Main execution function"""

    print("🚀 Improved Decision Policy v1 Trials")
    print("=" * 50)

    trials = ImprovedDecisionTrials()
    trials.run_comprehensive_improved_trials()

    print("\n🎉 Improved trials completed!")


if __name__ == "__main__":
    main()
