#!/usr/bin/env python3
"""
Planner Integration Demo
========================

This demo shows how the new Planner component works with semantic search,
graph expansion, IO contract validation, and plan caching.
"""

import sys
sys.path.insert(0, '/home/runner/work/super-alita/super-alita')

from src.core.planner import Planner, VectorIndex, GraphStore

def demo_planner():
    print("🎯 Planner Integration Demo")
    print("=" * 50)
    
    # Set up a sample graph with atoms and their contracts
    print("\n1. Setting up GraphStore with sample atoms...")
    gs = GraphStore()
    
    # Define some sample atoms with their I/O contracts
    gs.add_atom("WebSearch", input_fields={"query"}, output_fields={"results", "links"})
    gs.add_atom("Summarize", input_fields={"results"}, output_fields={"summary", "key_points"})
    gs.add_atom("CodeGen", input_fields={"summary"}, output_fields={"code", "tests"})
    gs.add_atom("Validate", input_fields={"code", "tests"}, output_fields={"validation_report"})
    
    # Define possible transitions between atoms
    gs.add_edge("WebSearch", "Summarize")
    gs.add_edge("Summarize", "CodeGen")
    gs.add_edge("CodeGen", "Validate")
    
    print(f"   ✅ Added {len(gs.input_contract)} atoms to graph")
    
    # Create a demo VectorIndex that simulates semantic similarity
    print("\n2. Setting up VectorIndex with semantic similarity...")
    
    class DemoVectorIndex(VectorIndex):
        def find_similar(self, query: str, top_k: int = 3):
            # Simulate semantic similarity based on keywords
            similarities = []
            
            if "search" in query.lower() or "find" in query.lower():
                similarities.append(("WebSearch", 0.95))
            if "summary" in query.lower() or "summarize" in query.lower():
                similarities.append(("Summarize", 0.85))
            if "code" in query.lower() or "generate" in query.lower():
                similarities.append(("CodeGen", 0.80))
            if "test" in query.lower() or "validate" in query.lower():
                similarities.append(("Validate", 0.75))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
    
    vi = DemoVectorIndex()
    print("   ✅ VectorIndex configured with semantic matching")
    
    # Create the planner
    print("\n3. Creating Planner with max_steps=4, cache_size=10...")
    planner = Planner(vector_index=vi, graph_store=gs, max_steps=4, cache_size=10)
    print("   ✅ Planner initialized")
    
    # Test different queries
    test_queries = [
        "search for Python tutorials and generate code examples",
        "find information about machine learning",
        "summarize the research and validate results", 
        "search for Python tutorials and generate code examples",  # Duplicate to test caching
    ]
    
    print("\n4. Testing plan generation...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: '{query}'")
        event = planner.propose_plan(query)
        plan = event.plan
        
        if len(plan.steps) > 0:
            print(f"   📋 Generated plan with {len(plan.steps)} steps:")
            for j, step in enumerate(plan.steps, 1):
                print(f"      {j}. {step.atom_id} - {step.description}")
            print(f"   💡 Rationale: {plan.rationale}")
        else:
            print(f"   ❌ No plan generated: {plan.rationale}")
    
    # Test contract compatibility
    print("\n5. Testing contract compatibility...")
    print(f"   WebSearch -> Summarize: {gs.is_contract_compatible('WebSearch', 'Summarize')}")
    print(f"   Summarize -> CodeGen: {gs.is_contract_compatible('Summarize', 'CodeGen')}")
    print(f"   CodeGen -> Validate: {gs.is_contract_compatible('CodeGen', 'Validate')}")
    print(f"   WebSearch -> CodeGen: {gs.is_contract_compatible('WebSearch', 'CodeGen')}")  # Should be False
    
    # Test caching statistics
    print(f"\n6. Cache statistics:")
    print(f"   Cached plans: {len(planner._plan_cache)}")
    print(f"   Cache keys: {list(planner._plan_cache.keys())}")
    
    print("\n🎉 Demo completed successfully!")

if __name__ == "__main__":
    demo_planner()