#!/usr/bin/env python3
"""
Decision Policy v1 Integration Trials
=====================================

This module runs comprehensive trials of the Decision Policy v1 implementation
integrated with the actual Super Alita infrastructure, including:

- Real tool registry integration
- MCP server capabilities
- Event bus communication
- FastAPI runtime execution
- Live scenario testing

The trials simulate real-world agent interactions to validate the decision-making
architecture under realistic conditions.
"""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Import Super Alita core components
from src.core.decision_policy_v1 import (
    CapabilityNode,
    DecisionPolicyEngine,
    IntentType,
    create_bootstrap_capabilities,
)


class DecisionPolicyTrials:
    """Comprehensive integration trials for Decision Policy v1"""

    def __init__(self):
        self.engine = DecisionPolicyEngine()
        self.event_bus = None
        self.results = []
        self.trial_start_time = None

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    async def setup_environment(self):
        """Initialize the trial environment"""
        self.logger.info("🚀 Setting up Decision Policy v1 trial environment...")

        # Initialize event bus (file-based for trials)
        log_dir = Path("./logs/decision_trials")
        log_dir.mkdir(parents=True, exist_ok=True)

        from src.main import FileEventBus

        self.event_bus = FileEventBus(str(log_dir))

        # Register bootstrap capabilities
        bootstrap_caps = create_bootstrap_capabilities()
        for cap in bootstrap_caps:
            self.engine.register_capability(cap)

        # Register additional trial capabilities
        trial_caps = self.create_trial_capabilities()
        for cap in trial_caps:
            self.engine.register_capability(cap)

        self.logger.info(f"✅ Registered {len(self.engine.capabilities)} capabilities")

        # Log trial setup event
        await self.event_bus.emit(
            {
                "event_type": "trial_setup",
                "capabilities_count": len(self.engine.capabilities),
                "trial_id": f"trial_{int(time.time())}",
            }
        )

    def create_trial_capabilities(self) -> list[CapabilityNode]:
        """Create additional capabilities for trial scenarios"""
        return [
            CapabilityNode(
                id="search.web",
                name="Web Search",
                description="Search the web for information",
                schema={
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer"},
                        },
                        "required": ["query"],
                    }
                },
                preconditions=["network_access"],
                side_effects=[],
                cost_hint=0.1,
                registry_type="normal",
            ),
            CapabilityNode(
                id="file.read",
                name="Read File",
                description="Read contents of a file",
                schema={
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "encoding": {"type": "string"},
                        },
                        "required": ["file_path"],
                    }
                },
                preconditions=["filesystem_access"],
                side_effects=[],
                cost_hint=0.05,
                registry_type="normal",
            ),
            CapabilityNode(
                id="file.write",
                name="Write File",
                description="Write content to a file",
                schema={
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                            "encoding": {"type": "string"},
                        },
                        "required": ["file_path", "content"],
                    }
                },
                preconditions=["filesystem_access"],
                side_effects=["file_modification"],
                cost_hint=0.05,
                registry_type="normal",
            ),
            CapabilityNode(
                id="code.execute",
                name="Execute Code",
                description="Execute Python code safely",
                schema={
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "timeout": {"type": "integer"},
                        },
                        "required": ["code"],
                    }
                },
                preconditions=["python_available"],
                side_effects=["code_execution"],
                cost_hint=0.2,
                registry_type="normal",
            ),
            CapabilityNode(
                id="analysis.code_quality",
                name="Code Quality Analysis",
                description="Analyze code quality and suggest improvements",
                schema={
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "language": {"type": "string"},
                        },
                        "required": ["code"],
                    }
                },
                preconditions=["analysis_tools"],
                side_effects=[],
                cost_hint=0.15,
                registry_type="mcp",
            ),
        ]

    async def run_trial_scenario(
        self, name: str, message: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a single trial scenario"""
        self.logger.info(f"🧪 Running trial: {name}")

        start_time = time.time()

        try:
            # Record trial start event
            await self.event_bus.emit(
                {
                    "event_type": "trial_start",
                    "trial_name": name,
                    "message": message,
                    "context": context,
                }
            )

            # Run decision policy
            plan = await self.engine.decide_and_plan(message, context)

            execution_time = time.time() - start_time

            # Record trial result
            result = {
                "trial_name": name,
                "success": True,
                "execution_time": execution_time,
                "plan": {
                    "run_id": plan.run_id,
                    "strategy": plan.strategy.name,
                    "confidence": plan.confidence,
                    "estimated_cost": plan.estimated_cost,
                    "step_count": len(plan.plan),
                    "risk_factors": plan.risk_factors,
                },
                "message": message,
                "context_keys": list(context.keys()),
                "timestamp": datetime.now(UTC).isoformat(),
            }

            await self.event_bus.emit({"event_type": "trial_complete", **result})

            self.logger.info(
                f"✅ Trial '{name}' completed: {plan.strategy.name} strategy, confidence {plan.confidence:.2f}"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            error_result = {
                "trial_name": name,
                "success": False,
                "execution_time": execution_time,
                "error": str(e),
                "message": message,
                "context_keys": list(context.keys()),
                "timestamp": datetime.now(UTC).isoformat(),
            }

            await self.event_bus.emit({"event_type": "trial_error", **error_result})

            self.logger.error(f"❌ Trial '{name}' failed: {e}")

            return error_result

    async def run_comprehensive_trials(self):
        """Run the full suite of decision policy trials"""
        self.trial_start_time = time.time()

        # Define trial scenarios
        scenarios = [
            {
                "name": "bootstrap_repository",
                "message": "clone the latest version of https://github.com/example/project.git and set it up for development",
                "context": {
                    "git_available": True,
                    "python_available": True,
                    "pip_available": True,
                    "filesystem_access": True,
                    "network_access": True,
                    "service_available": True,
                },
            },
            {
                "name": "information_query",
                "message": "what is the current status of the running services and show me the logs",
                "context": {
                    "network_access": True,
                    "filesystem_access": True,
                    "service_available": True,
                },
            },
            {
                "name": "code_analysis",
                "message": "analyze the code quality of the main.py file and suggest improvements",
                "context": {
                    "filesystem_access": True,
                    "analysis_tools": True,
                    "python_available": True,
                },
            },
            {
                "name": "file_operations",
                "message": "read the configuration file and update the database connection settings",
                "context": {"filesystem_access": True, "config_available": True},
            },
            {
                "name": "search_and_execute",
                "message": "search for the latest Python best practices and create a sample implementation",
                "context": {
                    "network_access": True,
                    "filesystem_access": True,
                    "python_available": True,
                },
            },
            {
                "name": "high_risk_operation",
                "message": "delete all temporary files and rebuild the entire project from scratch",
                "context": {
                    "filesystem_access": True,
                    "git_available": True,
                    "python_available": True,
                    "destructive_allowed": True,
                },
            },
            {
                "name": "missing_capabilities",
                "message": "launch a rocket to mars and establish a colony",
                "context": {"network_access": True},
            },
            {
                "name": "parallel_workflow",
                "message": "simultaneously run tests, generate documentation, and deploy to staging",
                "context": {
                    "python_available": True,
                    "git_available": True,
                    "network_access": True,
                    "deployment_tools": True,
                    "testing_tools": True,
                    "documentation_tools": True,
                },
            },
        ]

        self.logger.info(
            f"🎯 Starting comprehensive trials with {len(scenarios)} scenarios"
        )

        # Run all scenarios
        for scenario in scenarios:
            result = await self.run_trial_scenario(
                scenario["name"], scenario["message"], scenario["context"]
            )
            self.results.append(result)

            # Brief pause between trials
            await asyncio.sleep(0.1)

        # Generate summary
        await self.generate_trial_summary()

    async def generate_trial_summary(self):
        """Generate comprehensive trial summary"""
        total_time = time.time() - self.trial_start_time

        # Calculate statistics
        successful_trials = [r for r in self.results if r["success"]]
        failed_trials = [r for r in self.results if not r["success"]]

        strategy_counts = {}
        confidence_scores = []
        execution_times = []

        for result in successful_trials:
            if "plan" in result:
                strategy = result["plan"]["strategy"]
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                confidence_scores.append(result["plan"]["confidence"])
            execution_times.append(result["execution_time"])

        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        )
        avg_execution_time = (
            sum(execution_times) / len(execution_times) if execution_times else 0
        )

        summary = {
            "trial_summary": {
                "total_trials": len(self.results),
                "successful_trials": len(successful_trials),
                "failed_trials": len(failed_trials),
                "success_rate": (
                    len(successful_trials) / len(self.results) if self.results else 0
                ),
                "total_execution_time": total_time,
                "average_execution_time": avg_execution_time,
                "average_confidence": avg_confidence,
                "strategy_distribution": strategy_counts,
                "capabilities_tested": len(self.engine.capabilities),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            "detailed_results": self.results,
        }

        # Save detailed results
        results_file = Path("./logs/decision_trials/trial_results.json")
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Emit summary event
        await self.event_bus.emit(
            {"event_type": "trials_summary", **summary["trial_summary"]}
        )

        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("📊 DECISION POLICY v1 TRIAL RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(
            f"✅ Success Rate: {summary['trial_summary']['success_rate']:.1%}"
        )
        self.logger.info(f"⚡ Avg Execution Time: {avg_execution_time:.3f}s")
        self.logger.info(f"🎯 Avg Confidence: {avg_confidence:.2f}")
        self.logger.info(f"🔧 Capabilities Tested: {len(self.engine.capabilities)}")
        self.logger.info(f"📈 Strategy Distribution: {strategy_counts}")
        self.logger.info(f"📂 Detailed results saved to: {results_file}")
        self.logger.info("=" * 60)

        return summary

    async def run_bandit_learning_trial(self):
        """Test the multi-armed bandit learning system"""
        self.logger.info("🎰 Testing multi-armed bandit learning...")

        # Simulate multiple executions with different outcomes
        capabilities = ["git.clone_or_pull", "file.read", "search.web"]

        for cap_id in capabilities:
            # Simulate varying success rates and performance
            for i in range(10):
                success = (i % 3) != 0  # 66% success rate
                cost = 0.05 + (i * 0.01)  # Increasing cost
                latency = 1.0 + (i * 0.1)  # Increasing latency

                self.engine.update_bandit_stats(cap_id, success, cost, latency)

        # Test utility calculations
        test_context = {"network_access": True, "filesystem_access": True}

        for cap_id, capability in self.engine.capabilities.items():
            if cap_id in capabilities:
                goal = self.engine.goal_synthesizer.synthesize(
                    IntentType.QUERY, {"test": True}, test_context
                )
                utility = self.engine.utility_calculator.calculate(
                    capability, goal, test_context
                )

                stats = self.engine.bandit_stats.get(cap_id, {})
                self.logger.info(
                    f"🎯 {cap_id}: utility={utility:.3f}, "
                    f"attempts={stats.get('attempts', 0)}, "
                    f"wins={stats.get('wins', 0)}, "
                    f"avg_reward={stats.get('avg_reward', 0):.3f}"
                )

        await self.event_bus.emit(
            {
                "event_type": "bandit_learning_test",
                "capabilities_tested": len(capabilities),
                "bandit_stats": {
                    cap_id: self.engine.bandit_stats.get(cap_id, {})
                    for cap_id in capabilities
                },
            }
        )


async def main():
    """Main entry point for decision policy trials"""
    print("🚀 Starting Decision Policy v1 Integration Trials")
    print("=" * 60)

    trials = DecisionPolicyTrials()

    try:
        # Setup environment
        await trials.setup_environment()

        # Run comprehensive trials
        await trials.run_comprehensive_trials()

        # Test bandit learning
        await trials.run_bandit_learning_trial()

        print("\n🎉 All trials completed successfully!")
        print("📂 Check ./logs/decision_trials/ for detailed results")

    except Exception as e:
        print(f"\n❌ Trials failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
