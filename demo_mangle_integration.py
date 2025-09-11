#!/usr/bin/env python3
"""
Demonstration script for Mangle integration in SDD framework.

This script shows how to use the new Mangle-enhanced SDD capabilities.
"""

import tempfile
from pathlib import Path

from src.sdd.enhanced_sdd_framework import EnhancedSDDFramework
from src.sdd.mangle_reasoner import MangleReasoner
from src.sdd.mangle_rules import get_available_queries, get_query_for_question


def demo_mangle_reasoner():
    """Demonstrate the Mangle reasoner capabilities."""
    print("🧠 Mangle Reasoner Demo")
    print("=" * 50)

    # Create a temporary workspace for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        reasoner = MangleReasoner(str(workspace))

        # Demo 1: Query mapping
        print("\n📋 Natural Language to Query Mapping:")
        test_questions = [
            "what functions are untested",
            "what features are incomplete",
            "what violates constitution",
            "unknown question pattern",
        ]

        for question in test_questions:
            query = get_query_for_question(question)
            print(f"  '{question}' → {query}")

        # Demo 2: Available queries
        print(f"\n🔍 Available Query Patterns ({len(get_available_queries())} total):")
        for i, pattern in enumerate(get_available_queries()[:5], 1):
            print(f"  {i}. {pattern}")
        print("  ... and more")

        # Demo 3: Fact generation (mock)
        print("\n📊 Knowledge Graph Statistics:")
        stats = reasoner.get_fact_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")


def demo_enhanced_framework():
    """Demonstrate the enhanced SDD framework."""
    print("\n🚀 Enhanced SDD Framework Demo")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        framework = EnhancedSDDFramework(workspace)

        # Demo natural language queries
        test_queries = [
            "what functions are untested",
            "what features are incomplete",
            "quality issues",
        ]

        print("\n💬 Natural Language Questions:")
        for query in test_queries:
            print(f"\n  Question: '{query}'")
            try:
                result = framework.ask_question(query)
                print(f"    Success: {result['success']}")
                print(f"    Query Used: {result['query_used']}")
                print(f"    Results: {len(result['results'])} items found")
                print(f"    Execution Time: {result['execution_time']:.3f}s")
            except Exception as e:
                print(f"    Error: {e}")

        # Demo constitutional compliance
        print("\n⚖️ Constitutional Compliance Analysis:")
        try:
            compliance = framework.validate_constitutional_compliance()
            print(f"    Mangle Analysis: {bool(compliance.get('mangle_analysis'))}")
            print(f"    Summary: {bool(compliance.get('summary'))}")
            print(f"    Recommendations: {len(compliance.get('recommendations', []))}")
        except Exception as e:
            print(f"    Error: {e}")

        # Demo code quality analysis
        print("\n🔍 Code Quality Analysis:")
        try:
            quality = framework.analyze_code_quality()
            print(f"    Quality Issues: {bool(quality.get('quality_issues'))}")
            print(f"    Incomplete Work: {bool(quality.get('incomplete_work'))}")
            print(f"    Metrics: {bool(quality.get('quality_metrics'))}")
            print(f"    Recommendations: {len(quality.get('recommendations', []))}")
        except Exception as e:
            print(f"    Error: {e}")


def demo_cli_commands():
    """Demonstrate the CLI commands that are available."""
    print("\n💻 Enhanced CLI Commands")
    print("=" * 50)

    cli_commands = {
        "ask": "Ask natural language questions about code",
        "validate": "Validate constitutional compliance",
        "trace": "Trace code elements to specifications",
        "analyze": "Analyze code quality and issues",
        "stats": "Show knowledge graph statistics",
        "clear_cache": "Clear all analysis caches",
        "specify": "Create feature specifications (SDD)",
        "plan": "Generate implementation plans (SDD)",
        "tasks": "Break down plans into tasks (SDD)",
        "untested": "List untested functions",
        "incomplete": "List incomplete features",
    }

    print("\n📋 Available Commands:")
    for cmd, desc in cli_commands.items():
        print(f"  sdd {cmd:<12} - {desc}")

    print("\n💡 Usage Examples:")
    print("  sdd ask 'what functions are untested'")
    print("  sdd validate")
    print("  sdd trace 'MyClass.my_method'")
    print("  sdd analyze")
    print("  sdd stats")


def demo_api_endpoints():
    """Show the new API endpoints that are available."""
    print("\n🌐 Enhanced API Endpoints")
    print("=" * 50)

    endpoints = {
        "POST /sdd/ask": "Ask natural language questions",
        "GET /sdd/validate": "Constitutional compliance analysis",
        "POST /sdd/trace": "Code-to-spec traceability",
        "GET /sdd/analyze/quality": "Code quality analysis",
        "GET /sdd/stats": "Knowledge graph statistics",
        "POST /sdd/cache/invalidate": "Clear all caches",
        "GET /sdd/untested-functions": "List untested functions",
        "GET /sdd/incomplete-features": "List incomplete features",
        "GET /sdd/constitutional-violations": "Constitutional violations",
        "GET /sdd/health": "Enhanced health check",
    }

    print("\n📡 New Mangle Reasoning Endpoints:")
    for endpoint, desc in endpoints.items():
        print(f"  {endpoint:<30} - {desc}")

    print("\n🔧 Enhanced SDD Endpoints (existing with Mangle):")
    print("  POST /sdd/specify                - Create specifications")
    print("  POST /sdd/plan                   - Generate plans")
    print("  POST /sdd/tasks                  - Create task breakdowns")


if __name__ == "__main__":
    print("🎯 Mangle Integration for SDD Framework")
    print("Code Knowledge Graph & Deductive Reasoning")
    print("=" * 60)

    demo_mangle_reasoner()
    demo_enhanced_framework()
    demo_cli_commands()
    demo_api_endpoints()

    print("\n✅ Mangle Integration Demo Complete!")
    print("\nNext Steps:")
    print("1. Start the FastAPI server: uvicorn app:app --reload --port 8080")
    print("2. Use the CLI: python -m src.sdd.sdd_cli ask 'what functions are untested'")
    print("3. Test the API: curl http://localhost:8080/sdd/health")
    print("4. Run the test suite: pytest tests/test_mangle_integration.py -v")
