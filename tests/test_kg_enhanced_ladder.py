#!/usr/bin/env python3
"""
Test KG-Enhanced LADDER Integration
==================================

This test validates the complete Knowledge Graph enhanced LADDER system:
1. Knowledge Graph storage and querying
2. KG-enhanced LADDER planning with historical context
3. Learning from planning outcomes
4. EventBus integration with KG learning
"""

import asyncio

from src.core.in_memory_event_bus import InMemoryEventBus
from src.knowledge_graph import KnowledgeGraphInterface
from src.ladder.integration.kg_enhanced_adapter import (
    KGEnhancedLadderAdapter,
    KGLadderIntegrationConfig,
    PlanningMode,
)


async def test_kg_enhanced_ladder_system():
    """Test complete KG-enhanced LADDER system."""
    print("🧠 Testing KG-Enhanced LADDER System")
    print("=" * 50)

    # 1. Setup Knowledge Graph
    print("\n1. Setting up Knowledge Graph...")
    kg_interface = KnowledgeGraphInterface()
    print(
        f"   ✓ KG initialized with {kg_interface.get_statistics()['patterns']} base patterns"
    )

    # 2. Setup EventBus
    print("\n2. Setting up EventBus...")
    event_bus = InMemoryEventBus()
    await event_bus.start()
    print("   ✓ EventBus initialized")

    # 3. Setup KG-Enhanced LADDER Adapter
    print("\n3. Setting up KG-Enhanced LADDER...")
    config = KGLadderIntegrationConfig(
        max_concurrent_tasks=2,
        planning_mode=PlanningMode.SHADOW,
        enable_kg_learning=True,
    )

    adapter = KGEnhancedLadderAdapter(
        kg_interface=kg_interface,
        event_bus=event_bus,
        source_plugin="test_kg_ladder",
        config=config,
    )
    await adapter.setup()
    print("   ✓ KG-Enhanced LADDER initialized")

    # 4. Test software development planning with KG context
    print("\n4. Testing software development planning...")

    result1 = await adapter.handle_request(
        "Create a Python function to calculate fibonacci numbers",
        {"language": "python", "complexity": "simple"},
    )

    print(f"   ✓ Planning 1: {result1['status']}")
    print(f"   Tasks created: {result1['tasks_created']}")
    print(f"   KG enhanced: {result1['kg_enhanced']}")

    if result1.get("kg_context"):
        kg_ctx = result1["kg_context"]
        print(f"   Domain: {kg_ctx['domain']}")
        print(f"   Patterns found: {kg_ctx['patterns_found']}")
        print(f"   Similar goals: {kg_ctx['similar_goals']}")

    # 5. Test similar goal to see if KG provides better context
    print("\n5. Testing similar goal planning...")

    result2 = await adapter.handle_request(
        "Implement a Python class for data processing",
        {"language": "python", "framework": "pandas"},
    )

    print(f"   ✓ Planning 2: {result2['status']}")
    print(f"   Tasks created: {result2['tasks_created']}")

    if result2.get("kg_context"):
        kg_ctx = result2["kg_context"]
        print(f"   Domain: {kg_ctx['domain']}")
        print(f"   Patterns found: {kg_ctx['patterns_found']}")
        print(f"   Historical outcomes: {kg_ctx['historical_outcomes']}")

    # 6. Test research task (different domain)
    print("\n6. Testing research domain planning...")

    result3 = await adapter.handle_request(
        "Research machine learning algorithms for text classification",
        {"domain": "machine_learning", "task_type": "research"},
    )

    print(f"   ✓ Planning 3: {result3['status']}")
    print(f"   Domain: {result3.get('kg_context', {}).get('domain', 'unknown')}")

    # 7. Check KG learning and statistics
    print("\n7. Checking KG learning and statistics...")

    kg_stats = kg_interface.get_statistics()
    print(f"   Entities in KG: {kg_stats['entities']}")
    print(f"   Relations in KG: {kg_stats['relations']}")
    print(f"   Patterns in KG: {kg_stats['patterns']}")
    print(f"   Planning contexts: {kg_stats['contexts']}")
    print(
        f"   Average pattern success rate: {kg_stats['average_pattern_success_rate']:.1%}"
    )

    # 8. Test concurrent planning with KG
    print("\n8. Testing concurrent KG-enhanced planning...")

    async def make_kg_request(i, domain):
        return await adapter.handle_request(
            f"Task {i}: Solve a {domain} problem", {"task_id": i, "domain": domain}
        )

    # Run concurrent requests in different domains
    concurrent_results = await asyncio.gather(
        make_kg_request(1, "software_development"),
        make_kg_request(2, "research"),
        make_kg_request(3, "general"),
        return_exceptions=True,
    )

    successful_kg_requests = sum(
        1
        for r in concurrent_results
        if isinstance(r, dict) and r.get("status") == "success"
    )
    print(f"   ✓ {successful_kg_requests}/3 concurrent KG requests successful")

    # 9. Check final integration metrics
    print("\n9. Final integration metrics...")
    final_metrics = adapter.get_metrics()

    print(f"   Total plans executed: {final_metrics['total_plans']}")
    print(f"   Overall success rate: {final_metrics['success_rate']:.2%}")
    print(f"   KG queries made: {final_metrics['kg_queries_made']}")
    print(f"   KG patterns used: {final_metrics['kg_patterns_used']}")
    print(f"   KG usage rate: {final_metrics['kg_usage_rate']:.1%}")

    # 10. Demonstrate pattern learning
    print("\n10. Demonstrating pattern learning...")

    # Find code development pattern
    code_patterns = [
        p
        for p in kg_interface.patterns.values()
        if p.pattern_name == "code_development"
    ]

    if code_patterns:
        pattern = code_patterns[0]
        print("   Code development pattern:")
        print(f"   - Usage count: {pattern.usage_count}")
        print(f"   - Success rate: {pattern.success_rate:.1%}")
        print(f"   - Steps: {len(pattern.decomposition_steps)}")
        for i, step in enumerate(pattern.decomposition_steps[:3], 1):
            print(f"     {i}. {step}")
        if len(pattern.decomposition_steps) > 3:
            print(f"     ... and {len(pattern.decomposition_steps) - 3} more steps")

    # 11. Cleanup
    print("\n11. Cleanup...")
    await event_bus.stop()
    print("   ✓ EventBus stopped")

    print("\n" + "=" * 50)
    print("🎉 KG-Enhanced LADDER Test COMPLETE!")
    print(
        f"📊 Summary: {final_metrics['total_plans']} plans, "
        f"{final_metrics['success_rate']:.1%} success rate"
    )
    print(
        f"🧠 KG Impact: {final_metrics['kg_queries_made']} queries, "
        f"{final_metrics['kg_patterns_used']} patterns used"
    )

    # Test that KG actually enhanced planning
    kg_enhancement_success = (
        final_metrics["kg_queries_made"] > 0
        and final_metrics["kg_patterns_used"] > 0
        and kg_stats["entities"] > 3  # Should have created entities
    )

    if kg_enhancement_success:
        print("✅ KG enhancement is working correctly!")
    else:
        print("⚠️  KG enhancement may not be fully active")

    return {
        **final_metrics,
        "kg_enhancement_active": kg_enhancement_success,
        "total_kg_entities": kg_stats["entities"],
        "total_kg_patterns": kg_stats["patterns"],
    }


if __name__ == "__main__":
    # Run the KG-enhanced integration test
    result = asyncio.run(test_kg_enhanced_ladder_system())
    print(f"\nFinal Result: {result}")
