#!/usr/bin/env python3
"""
Enhanced Decision Policy v1 Trials with Improved Matching
=========================================================

This enhanced version improves the text similarity matching to trigger
more realistic decision-making scenarios and test the full range of
strategy types.
"""

import asyncio
import logging
import time

from src.core.decision_policy_v1 import (
    CapabilityNode,
    DecisionPolicyEngine,
    create_bootstrap_capabilities,
)


class EnhancedDecisionTrials:
    """Enhanced trials with improved capability matching"""

    def __init__(self):
        self.engine = DecisionPolicyEngine()
        self.results = []

        # Enhanced text similarity with keyword expansion
        self.keyword_mappings = {
            "git": [
                "clone",
                "repository",
                "repo",
                "version",
                "latest",
                "pull",
                "fetch",
            ],
            "environment": [
                "setup",
                "install",
                "dependencies",
                "development",
                "bootstrap",
            ],
            "status": ["show", "current", "running", "services", "logs", "health"],
            "file": [
                "read",
                "write",
                "configuration",
                "config",
                "database",
                "settings",
            ],
            "search": ["web", "find", "latest", "best practices", "information"],
            "analysis": ["analyze", "code", "quality", "suggest", "improvements"],
            "execute": [
                "run",
                "tests",
                "generate",
                "documentation",
                "deploy",
                "staging",
            ],
            "create": ["make", "new", "sample", "implementation"],
        }

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def enhanced_text_similarity(self, text1: str, text2: str) -> float:
        """Enhanced text similarity with keyword expansion"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Basic word overlap
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        base_score = overlap / max(1, union)

        # Enhanced scoring with keyword mappings
        enhanced_score = base_score

        for category, keywords in self.keyword_mappings.items():
            # Check if any keywords from this category appear in both texts
            text1_has_category = any(keyword in text1.lower() for keyword in keywords)
            text2_has_category = any(keyword in text2.lower() for keyword in keywords)

            if text1_has_category and text2_has_category:
                enhanced_score += 0.3  # Boost for category match

        return min(1.0, enhanced_score)

    def setup_enhanced_engine(self):
        """Setup engine with enhanced matching"""
        # Monkey patch the text similarity method
        self.engine.text_similarity = self.enhanced_text_similarity

        # Register all capabilities
        bootstrap_caps = create_bootstrap_capabilities()
        for cap in bootstrap_caps:
            self.engine.register_capability(cap)

        # Enhanced trial capabilities with better descriptions
        enhanced_caps = [
            CapabilityNode(
                id="search.web_info",
                name="Web Information Search",
                description="search web find information latest best practices",
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
                id="file.config_operations",
                name="Configuration File Operations",
                description="read write configuration file database connection settings",
                schema={
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "operation": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["file_path", "operation"],
                    }
                },
                preconditions=["filesystem_access"],
                side_effects=["file_modification"],
                cost_hint=0.08,
                registry_type="normal",
            ),
            CapabilityNode(
                id="analysis.code_review",
                name="Code Quality Analysis",
                description="analyze code quality main file suggest improvements",
                schema={
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "analysis_type": {"type": "string"},
                        },
                        "required": ["file_path"],
                    }
                },
                preconditions=["analysis_tools", "filesystem_access"],
                side_effects=[],
                cost_hint=0.15,
                registry_type="mcp",
            ),
            CapabilityNode(
                id="service.status_check",
                name="Service Status Monitor",
                description="show current status running services logs health check",
                schema={
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "service_type": {"type": "string"},
                            "include_logs": {"type": "boolean"},
                        },
                    }
                },
                preconditions=["service_available"],
                side_effects=[],
                cost_hint=0.05,
                registry_type="normal",
            ),
            CapabilityNode(
                id="workflow.parallel_execution",
                name="Parallel Workflow Runner",
                description="run tests generate documentation deploy staging simultaneously",
                schema={
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "tasks": {"type": "array"},
                            "parallel": {"type": "boolean"},
                        },
                        "required": ["tasks"],
                    }
                },
                preconditions=[
                    "testing_tools",
                    "documentation_tools",
                    "deployment_tools",
                ],
                side_effects=["deployment", "test_execution"],
                cost_hint=0.3,
                streamable=True,
                registry_type="mcp",
            ),
        ]

        for cap in enhanced_caps:
            self.engine.register_capability(cap)

        self.logger.info(
            f"✅ Enhanced engine setup with {len(self.engine.capabilities)} capabilities"
        )

    async def run_enhanced_trial(self, name: str, message: str, context: dict) -> dict:
        """Run enhanced trial with better matching"""
        self.logger.info(f"🧪 Enhanced trial: {name}")

        start_time = time.time()
        plan = await self.engine.decide_and_plan(message, context)
        execution_time = time.time() - start_time

        result = {
            "trial_name": name,
            "execution_time": execution_time,
            "strategy": plan.strategy.name,
            "confidence": plan.confidence,
            "step_count": len(plan.plan),
            "estimated_cost": plan.estimated_cost,
            "risk_factors": plan.risk_factors,
            "message": message,
        }

        self.logger.info(
            f"✅ {name}: {plan.strategy.name} strategy, confidence {plan.confidence:.2f}"
        )
        return result

    async def run_enhanced_scenarios(self):
        """Run enhanced scenarios that should trigger different strategies"""

        scenarios = [
            {
                "name": "git_repository_clone",
                "message": "clone the repository and setup environment",
                "context": {
                    "git_available": True,
                    "python_available": True,
                    "filesystem_access": True,
                    "network_access": True,
                },
            },
            {
                "name": "service_status_query",
                "message": "show current status of running services",
                "context": {"service_available": True, "network_access": True},
            },
            {
                "name": "code_analysis_task",
                "message": "analyze code quality of main file",
                "context": {"analysis_tools": True, "filesystem_access": True},
            },
            {
                "name": "config_file_update",
                "message": "read configuration file and update settings",
                "context": {"filesystem_access": True, "config_available": True},
            },
            {
                "name": "web_search_task",
                "message": "search for latest best practices",
                "context": {"network_access": True},
            },
            {
                "name": "parallel_deployment",
                "message": "run tests and deploy to staging simultaneously",
                "context": {
                    "testing_tools": True,
                    "deployment_tools": True,
                    "documentation_tools": True,
                    "python_available": True,
                },
            },
        ]

        for scenario in scenarios:
            result = await self.run_enhanced_trial(
                scenario["name"], scenario["message"], scenario["context"]
            )
            self.results.append(result)
            await asyncio.sleep(0.05)

        # Generate summary
        strategy_counts = {}
        confidence_scores = []

        for result in self.results:
            strategy = result["strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            confidence_scores.append(result["confidence"])

        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        print("\n" + "=" * 60)
        print("📊 ENHANCED DECISION POLICY TRIAL RESULTS")
        print("=" * 60)
        print(f"🎯 Total Scenarios: {len(self.results)}")
        print(f"⚡ Average Confidence: {avg_confidence:.2f}")
        print(f"📈 Strategy Distribution: {strategy_counts}")

        # Show detailed results
        print("\n📋 Detailed Results:")
        for result in self.results:
            print(
                f"  {result['trial_name']}: {result['strategy']} "
                f"(confidence: {result['confidence']:.2f}, "
                f"steps: {result['step_count']}, "
                f"cost: {result['estimated_cost']:.2f})"
            )

        print("=" * 60)


async def main():
    """Run enhanced decision policy trials"""
    print("🚀 Enhanced Decision Policy v1 Trials")
    print("=" * 50)

    trials = EnhancedDecisionTrials()
    trials.setup_enhanced_engine()
    await trials.run_enhanced_scenarios()

    print("\n🎉 Enhanced trials completed!")


if __name__ == "__main__":
    asyncio.run(main())
