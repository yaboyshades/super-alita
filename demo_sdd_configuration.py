#!/usr/bin/env python3
"""
Simple Demo: SDD Workflow Configuration and Architecture

This script shows the configuration and architecture of the unified
orchestrator and SDD workflow without complex async operations.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestration.unified_orchestrator import UnifiedRunConfig  # noqa: E402


def demo_configurations():
    """Show different orchestrator configurations."""
    print("🎬 Super Alita Unified Orchestrator - Configuration Demo")
    print("=" * 60)

    # Demo 1: Basic Planning Configuration
    print("\n📋 Demo 1: Basic Planning Configuration")
    basic_config = UnifiedRunConfig(
        prompt="Create a web scraper for article extraction",
        run_id="demo-001",
        session_id="basic-session",
        enable_specification=False,
        enable_planning=True,
        enable_tasks=False,
        enable_consensus=True,
        enable_code_generation=False,
        enable_validation=False,
        timeout_s=60,
    )

    print(f"   Prompt: {basic_config.prompt}")
    print("   Enabled Stages: Planning ✓, Consensus ✓")
    print(f"   SDD Mode: {basic_config.sdd_mode}")
    print(f"   Timeout: {basic_config.timeout_s}s")

    # Demo 2: Full SDD Workflow Configuration
    print("\n🏛️ Demo 2: Full SDD Workflow with Constitutional Validation")
    sdd_config = UnifiedRunConfig(
        prompt="Build a production ML model following constitutional principles",
        run_id="demo-sdd-002",
        session_id="sdd-session",
        enable_specification=True,
        enable_planning=True,
        enable_tasks=True,
        enable_consensus=True,
        enable_validation=True,
        sdd_mode=True,
        constitutional_threshold=0.75,
        sdd_template_dir="templates/sdd",
        timeout_s=120,
    )

    print(f"   Prompt: {sdd_config.prompt}")
    print(
        "   Enabled Stages: Specification ✓, Planning ✓, Tasks ✓, Consensus ✓, Validation ✓"
    )
    print(f"   SDD Mode: {sdd_config.sdd_mode} ✅")
    print(f"   Constitutional Threshold: {sdd_config.constitutional_threshold}")
    print(f"   Template Directory: {sdd_config.sdd_template_dir}")

    # Demo 3: Production Code Generation Configuration
    print("\n🚀 Demo 3: Production Code Generation Pipeline")
    production_config = UnifiedRunConfig(
        prompt="Implement a secure user authentication system",
        run_id="demo-prod-003",
        session_id="production-session",
        enable_specification=True,
        enable_planning=True,
        enable_tasks=True,
        enable_consensus=True,
        enable_code_generation=True,
        enable_validation=True,
        enable_scoring=True,
        sdd_mode=True,
        constitutional_threshold=0.80,
        test_first=True,
        file_path="src/auth/user_auth.py",
        language="python",
        timeout_s=180,
    )

    print(f"   Prompt: {production_config.prompt}")
    print("   Full Pipeline: All stages enabled ✅")
    print(f"   Test-First Development: {production_config.test_first}")
    print(f"   Target File: {production_config.file_path}")
    print(f"   Constitutional Threshold: {production_config.constitutional_threshold}")

    return basic_config, sdd_config, production_config


def demo_sdd_workflow_stages():
    """Show the SDD workflow stages and constitutional gates."""
    print("\n" + "=" * 60)
    print("🏗️ SDD Workflow Stages & Constitutional Gates")
    print("=" * 60)

    stages = [
        {
            "name": "sdd_specify_stage",
            "description": "Constitutional specification generation with existing solution research",
            "constitutional_gate": "Specification_Gate",
            "outputs": [
                "Requirements specification",
                "Existing solution research (Article I: Library-First)",
                "Testability requirements (Article II: Test-First)",
                "Simplicity constraints (Article III: Simplicity Gate)",
                "Constitutional compliance score",
            ],
        },
        {
            "name": "sdd_plan_stage",
            "description": "Plan creation with constitutional compliance validation",
            "constitutional_gate": "Planning_Gate",
            "outputs": [
                "Implementation plan with phases",
                "Library evaluation and selection",
                "Test design allocation",
                "Integration phases definition",
                "Constitutional compliance validation",
            ],
        },
        {
            "name": "sdd_tasks_stage",
            "description": "Task breakdown with simplicity and testability gates",
            "constitutional_gate": "Implementation_Gate",
            "outputs": [
                "Atomic, actionable task breakdown",
                "Test creation prerequisites",
                "Simplicity metrics validation",
                "Constitutional compliance checking",
                "Integration testing requirements",
            ],
        },
        {
            "name": "sdd_validate_stage",
            "description": "End-to-end constitutional compliance verification",
            "constitutional_gate": "Integration_Gate",
            "outputs": [
                "Comprehensive constitutional audit",
                "All six articles compliance check",
                "Quality metrics validation",
                "Integration testing results",
                "Final compliance report",
            ],
        },
    ]

    for i, stage in enumerate(stages, 1):
        print(f"\n🔧 Stage {i}: {stage['name']}")
        print(f"   Description: {stage['description']}")
        print(f"   Constitutional Gate: {stage['constitutional_gate']}")
        print("   Key Outputs:")
        for output in stage["outputs"]:
            print(f"     • {output}")


def demo_constitutional_framework():
    """Show the constitutional framework integration."""
    print("\n" + "=" * 60)
    print("🏛️ Constitutional Framework - Six Core Articles")
    print("=" * 60)

    articles = [
        {
            "number": "I",
            "name": "Library-First Development",
            "description": "Research and adopt existing solutions before creating new ones",
            "sdd_integration": [
                "Existing solution research in /specify phase",
                "Library evaluation in /plan phase",
                "Dependency management in /tasks phase",
            ],
            "validation_criteria": [
                "Documented research of existing libraries",
                "Justification for new vs. existing solutions",
                "Dependency impact assessment",
            ],
        },
        {
            "number": "II",
            "name": "Test-First Development",
            "description": "80% test coverage minimum, comprehensive test suites",
            "sdd_integration": [
                "Testability requirements in /specify phase",
                "Test design allocation in /plan phase",
                "Test creation prerequisites in /tasks phase",
            ],
            "validation_criteria": [
                "Test coverage ≥ 80%",
                "Test-first implementation order",
                "Comprehensive test suite design",
            ],
        },
        {
            "number": "III",
            "name": "Simplicity Gate",
            "description": "Functions <50 lines, cyclomatic complexity <10, clear interfaces",
            "sdd_integration": [
                "Simplicity constraints in /specify phase",
                "Complexity budgeting in /plan phase",
                "Simplicity metrics validation in /tasks phase",
            ],
            "validation_criteria": [
                "Function length < 50 lines",
                "Cyclomatic complexity < 10",
                "Clear, minimal interfaces",
            ],
        },
        {
            "number": "IV",
            "name": "Integration-First Testing",
            "description": "End-to-end integration testing over unit testing",
            "sdd_integration": [
                "Integration requirements in /specify phase",
                "Integration phases in /plan phase",
                "Integration testing in /tasks phase",
            ],
            "validation_criteria": [
                "Integration test coverage prioritized",
                "End-to-end workflow validation",
                "System-level behavior verification",
            ],
        },
        {
            "number": "V",
            "name": "Clarity and Unambiguity",
            "description": "Clear documentation, explicit interfaces, no ambiguous behavior",
            "sdd_integration": [
                "Clear requirements in /specify phase",
                "Explicit interfaces in /plan phase",
                "Documentation requirements in /tasks phase",
            ],
            "validation_criteria": [
                "Comprehensive documentation",
                "Explicit interface definitions",
                "Unambiguous behavior specifications",
            ],
        },
        {
            "number": "VI",
            "name": "Counterfactual Justification",
            "description": "Document decisions with reasoning and alternatives considered",
            "sdd_integration": [
                "Decision rationale in /specify phase",
                "Alternative analysis in /plan phase",
                "Implementation justification in /tasks phase",
            ],
            "validation_criteria": [
                "Decision documentation",
                "Alternative solutions considered",
                "Reasoning and trade-offs explained",
            ],
        },
    ]

    for article in articles:
        print(f"\n📜 Article {article['number']}: {article['name']}")
        print(f"   Description: {article['description']}")
        print("   SDD Integration:")
        for integration in article["sdd_integration"]:
            print(f"     • {integration}")
        print("   Validation Criteria:")
        for criteria in article["validation_criteria"]:
            print(f"     • {criteria}")


def demo_system_architecture():
    """Show the unified system architecture."""
    print("\n" + "=" * 60)
    print("🏗️ Unified System Architecture")
    print("=" * 60)

    components = {
        "Unified Orchestrator": {
            "file": "src/orchestration/unified_orchestrator.py",
            "description": "Stage-based pipeline orchestrator with constitutional validation",
            "capabilities": [
                "Event-driven stage execution",
                "SDD workflow integration",
                "Constitutional compliance monitoring",
                "Graceful error handling and degradation",
                "Real-time observability and metrics",
            ],
        },
        "SDD Configuration": {
            "file": "src/sdd/config.py",
            "description": "SDD workflow configuration and constitutional gates",
            "capabilities": [
                "Workflow stage definitions",
                "Constitutional threshold management",
                "Template and validator configuration",
                "Quality gate specifications",
            ],
        },
        "SDD Router": {
            "file": "src/sdd/router.py",
            "description": "FastAPI endpoints for SDD workflow",
            "capabilities": [
                "POST /sdd/specify endpoint",
                "POST /sdd/plan endpoint",
                "POST /sdd/tasks endpoint",
                "POST /sdd/validate endpoint",
                "Mangle reasoning integration",
            ],
        },
        "Constitutional Memory": {
            "file": "memory/sdd/constitutional_sdd_framework.md",
            "description": "Constitutional framework mapping for SDD",
            "capabilities": [
                "Article-to-stage mapping",
                "Compliance scoring guidelines",
                "Validation criteria definitions",
                "Best practices documentation",
            ],
        },
        "SDD Templates": {
            "file": "templates/sdd/",
            "description": "Markdown templates for SDD phases",
            "capabilities": [
                "Specification template (specification.md)",
                "Planning template (plan.md)",
                "Task breakdown template (tasks.md)",
                "Constitutional validation integration",
            ],
        },
    }

    for name, info in components.items():
        print(f"\n🔧 {name}")
        print(f"   File: {info['file']}")
        print(f"   Description: {info['description']}")
        print("   Capabilities:")
        for capability in info["capabilities"]:
            print(f"     • {capability}")


def main():
    """Run all configuration demos."""
    print("🎯 Super Alita - Unified Orchestrator & SDD Workflow")
    print("Configuration and Architecture Demonstration")

    # Show configurations
    demo_configurations()

    # Show SDD workflow stages
    demo_sdd_workflow_stages()

    # Show constitutional framework
    demo_constitutional_framework()

    # Show system architecture
    demo_system_architecture()

    print("\n" + "=" * 60)
    print("✅ CONFIGURATION DEMO COMPLETED!")
    print("=" * 60)
    print()
    print("🎯 Next Steps:")
    print("1. Review the configuration options for your use case")
    print("2. Set up SDD templates in templates/sdd/")
    print("3. Configure constitutional memory in memory/sdd/")
    print("4. Use the unified orchestrator for your development workflow")
    print("5. Monitor constitutional compliance throughout development")


if __name__ == "__main__":
    main()
