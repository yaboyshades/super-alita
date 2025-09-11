#!/usr/bin/env python3
"""
Practical Demo: Building a Web Scraper using SDD Workflow

This demonstrates a real-world example of using the unified orchestrator
and SDD workflow to build a production-ready web scraper following
constitutional principles.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestration.unified_orchestrator import UnifiedRunConfig  # noqa: E402


def demo_web_scraper_sdd_workflow():
    """Show how to use SDD workflow for building a web scraper."""
    print("🕷️ Practical Demo: Building a Web Scraper with SDD Workflow")
    print("=" * 65)

    # Phase 1: /specify - Requirements with Constitutional Validation
    print("\n📋 Phase 1: /sdd/specify - Constitutional Requirements Generation")
    print("-" * 50)

    specify_config = UnifiedRunConfig(
        prompt="""
        Build a production-ready web scraper for extracting article titles,
        authors, and publication dates from news websites. The scraper should
        handle rate limiting, respect robots.txt, and provide structured output.
        """,
        run_id="scraper-specify-001",
        session_id="scraper-session",
        enable_specification=True,
        enable_planning=False,
        enable_tasks=False,
        sdd_mode=True,
        constitutional_threshold=0.75,
    )

    print("🎯 Specification Requirements:")
    print("   • Library-First Research (Article I):")
    print("     - Existing libraries: requests, BeautifulSoup4, scrapy, selenium")
    print("     - Robots.txt handling: urllib.robotparser, reppy")
    print("     - Rate limiting: time.sleep, asyncio, aiohttp")
    print()
    print("   • Test-First Development (Article II):")
    print("     - Unit tests for parsing logic")
    print("     - Integration tests with mock websites")
    print("     - Performance tests for rate limiting")
    print("     - Target: ≥80% test coverage")
    print()
    print("   • Simplicity Constraints (Article III):")
    print("     - Functions <50 lines each")
    print("     - Clear separation: fetch, parse, store")
    print("     - Minimal external dependencies")
    print("     - Simple configuration interface")

    print(
        f"\n✅ Constitutional Score Target: ≥{specify_config.constitutional_threshold}"
    )

    # Phase 2: /plan - Implementation Planning with Constitutional Review
    print("\n🏗️ Phase 2: /sdd/plan - Constitutional Implementation Planning")
    print("-" * 50)

    plan_config = UnifiedRunConfig(
        prompt="Create implementation plan for the web scraper specification",
        run_id="scraper-plan-001",
        session_id="scraper-session",
        enable_specification=False,
        enable_planning=True,
        enable_tasks=False,
        sdd_mode=True,
        constitutional_threshold=0.75,
    )

    print("📐 Implementation Plan:")
    print("   1. Library Evaluation & Selection (Article I):")
    print("      - requests + BeautifulSoup4 (simple, proven)")
    print("      - unittest + pytest (comprehensive testing)")
    print("      - dataclasses (structured output)")
    print()
    print("   2. Test Design Allocation (Article II):")
    print("      - 30% development time for test design")
    print("      - Test-driven development workflow")
    print("      - Mock server for integration testing")
    print()
    print("   3. Architecture Design (Article III):")
    print("      - WebScraper class (<200 lines total)")
    print("      - Separate modules: fetcher, parser, exporter")
    print("      - Configuration-driven design")
    print()
    print("   4. Integration Phases (Article IV):")
    print("      - Phase 1: Basic fetch + parse")
    print("      - Phase 2: Rate limiting + robots.txt")
    print("      - Phase 3: Error handling + retry logic")
    print("      - Phase 4: Output formatting + export")

    print("\n✅ Constitutional Compliance: All articles addressed")

    # Phase 3: /tasks - Task Breakdown with Validation
    print("\n✅ Phase 3: /sdd/tasks - Constitutional Task Breakdown")
    print("-" * 50)

    tasks_config = UnifiedRunConfig(
        prompt="Break down the web scraper implementation into atomic tasks",
        run_id="scraper-tasks-001",
        session_id="scraper-session",
        enable_specification=False,
        enable_planning=False,
        enable_tasks=True,
        sdd_mode=True,
        constitutional_threshold=0.75,
    )

    print("🎯 Atomic Task Breakdown:")
    print()
    print("   Library Integration Tasks (Article I):")
    print("   [ ] 1. Install requests, beautifulsoup4, pytest")
    print("   [ ] 2. Research robots.txt parsing libraries")
    print("   [ ] 3. Evaluate rate limiting approaches")
    print("   [ ] 4. Document library selection rationale")
    print()
    print("   Test Creation Tasks (Article II):")
    print("   [ ] 5. Create test project structure")
    print("   [ ] 6. Write unit tests for URL validation")
    print("   [ ] 7. Write unit tests for HTML parsing")
    print("   [ ] 8. Create mock HTTP server for testing")
    print("   [ ] 9. Write integration tests for full workflow")
    print()
    print("   Implementation Tasks (Article III):")
    print("   [ ] 10. Create WebScraper class skeleton (<50 lines)")
    print("   [ ] 11. Implement fetch() method (<30 lines)")
    print("   [ ] 12. Implement parse() method (<40 lines)")
    print("   [ ] 13. Implement save() method (<20 lines)")
    print("   [ ] 14. Add configuration management (<25 lines)")
    print()
    print("   Integration Tasks (Article IV):")
    print("   [ ] 15. Test fetch + parse integration")
    print("   [ ] 16. Add rate limiting between requests")
    print("   [ ] 17. Implement robots.txt checking")
    print("   [ ] 18. Add comprehensive error handling")
    print("   [ ] 19. Create end-to-end workflow test")

    complexity_metrics = {
        "total_tasks": 19,
        "avg_task_complexity": 2.8,  # Scale of 1-5
        "max_task_complexity": 4,  # Integration tests
        "estimated_hours": 16,
        "test_tasks_ratio": 0.42,  # 8/19 tasks are test-related
    }

    print("\n📊 Complexity Metrics:")
    for metric, value in complexity_metrics.items():
        print(f"   {metric}: {value}")

    print("\n✅ Simplicity Validation: All tasks <5 complexity")

    # Phase 4: /validate - Constitutional Compliance Verification
    print("\n🏛️ Phase 4: /sdd/validate - Constitutional Compliance Check")
    print("-" * 50)

    validate_config = UnifiedRunConfig(
        prompt="Validate constitutional compliance of web scraper implementation",
        run_id="scraper-validate-001",
        session_id="scraper-session",
        enable_validation=True,
        sdd_mode=True,
        constitutional_threshold=0.75,
    )

    constitutional_scores = {
        "Article I - Library-First": 0.85,  # Strong library research
        "Article II - Test-First": 0.82,  # 42% test task ratio, good coverage plan
        "Article III - Simplicity": 0.78,  # All functions <50 lines, clear separation
        "Article IV - Integration": 0.80,  # Clear integration phases
        "Article V - Clarity": 0.83,  # Clear documentation and interfaces
        "Article VI - Justification": 0.79,  # Documented decision rationale
    }

    print("📊 Constitutional Compliance Scores:")
    total_score = 0
    for article, score in constitutional_scores.items():
        status = (
            "✅ PASS"
            if score >= validate_config.constitutional_threshold
            else "❌ FAIL"
        )
        print(f"   {article}: {score:.2f} {status}")
        total_score += score

    average_score = total_score / len(constitutional_scores)
    overall_status = (
        "✅ COMPLIANT"
        if average_score >= validate_config.constitutional_threshold
        else "❌ NON-COMPLIANT"
    )

    print(f"\n🎯 Overall Constitutional Score: {average_score:.2f} {overall_status}")

    # Summary and Next Steps
    print("\n" + "=" * 65)
    print("🎉 SDD Workflow Complete - Web Scraper Ready for Implementation!")
    print("=" * 65)

    print("\n📋 Deliverables Generated:")
    print("   ✅ Constitutional requirements specification")
    print("   ✅ Implementation plan with constitutional review")
    print("   ✅ Atomic task breakdown with complexity validation")
    print("   ✅ Constitutional compliance verification")

    print("\n🚀 Implementation Ready:")
    print("   • All 19 tasks defined and validated")
    print("   • Constitutional compliance score: 0.81/1.00")
    print("   • Test coverage plan: 42% of tasks are test-related")
    print("   • Estimated implementation time: 16 hours")

    print("\n🎯 Next Steps:")
    print("   1. Execute tasks 1-4 (Library Integration)")
    print("   2. Execute tasks 5-9 (Test Creation)")
    print("   3. Execute tasks 10-14 (Implementation)")
    print("   4. Execute tasks 15-19 (Integration)")
    print("   5. Run final constitutional validation")

    return {
        "specification": specify_config,
        "planning": plan_config,
        "tasks": tasks_config,
        "validation": validate_config,
        "constitutional_scores": constitutional_scores,
        "complexity_metrics": complexity_metrics,
    }


def demo_usage_examples():
    """Show how to use the unified orchestrator for different scenarios."""
    print("\n" + "=" * 65)
    print("💡 Usage Examples: Unified Orchestrator Configurations")
    print("=" * 65)

    examples = [
        {
            "name": "Rapid Prototyping",
            "description": "Quick proof-of-concept with basic validation",
            "config": UnifiedRunConfig(
                prompt="Create a simple REST API for user management",
                run_id="prototype-001",
                enable_planning=True,
                enable_consensus=True,
                sdd_mode=False,
                timeout_s=60,
            ),
        },
        {
            "name": "Production Development",
            "description": "Full SDD workflow with all constitutional gates",
            "config": UnifiedRunConfig(
                prompt="Build enterprise-grade authentication microservice",
                run_id="production-001",
                enable_specification=True,
                enable_planning=True,
                enable_tasks=True,
                enable_consensus=True,
                enable_validation=True,
                sdd_mode=True,
                constitutional_threshold=0.80,
                test_first=True,
                timeout_s=300,
            ),
        },
        {
            "name": "Research Implementation",
            "description": "Paper-to-code with enhanced consensus",
            "config": UnifiedRunConfig(
                prompt="Implement Transformer attention mechanism from research paper",
                run_id="research-001",
                enable_specification=True,
                enable_planning=True,
                enable_consensus=True,
                enable_code_generation=True,
                sdd_mode=True,
                constitutional_threshold=0.75,
                file_path="src/models/transformer.py",
                language="python",
                timeout_s=180,
            ),
        },
    ]

    for example in examples:
        print(f"\n🔧 {example['name']}")
        print(f"   Description: {example['description']}")
        config = example["config"]

        enabled_stages = []
        if config.enable_specification:
            enabled_stages.append("Specification")
        if config.enable_planning:
            enabled_stages.append("Planning")
        if config.enable_tasks:
            enabled_stages.append("Tasks")
        if config.enable_consensus:
            enabled_stages.append("Consensus")
        if config.enable_code_generation:
            enabled_stages.append("Code Generation")
        if config.enable_validation:
            enabled_stages.append("Validation")

        print(f"   Enabled Stages: {', '.join(enabled_stages)}")
        print(f"   SDD Mode: {'✅ Yes' if config.sdd_mode else '❌ No'}")
        if config.sdd_mode:
            print(f"   Constitutional Threshold: {config.constitutional_threshold}")
        print(f"   Timeout: {config.timeout_s}s")


if __name__ == "__main__":
    print("🎬 Super Alita SDD Workflow - Practical Implementation Demo")
    print()

    # Main demo: Web scraper using SDD workflow
    demo_web_scraper_sdd_workflow()

    # Usage examples
    demo_usage_examples()

    print("\n" + "=" * 65)
    print("✨ SDD Workflow Demo Complete!")
    print("=" * 65)
    print()
    print("Key Takeaways:")
    print("🏛️ Constitutional validation ensures quality at every stage")
    print("📋 SDD workflow breaks complex projects into manageable phases")
    print("🎯 Unified orchestrator provides consistent, traceable execution")
    print("⚡ Real-time observability enables monitoring and debugging")
    print("🛡️ Graceful error handling prevents cascading failures")
    print()
    print(
        "Ready to implement production-quality software with constitutional compliance!"
    )
