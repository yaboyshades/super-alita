#!/usr/bin/env python3
"""
LeanRAG Live Demonstration
Shows hierarchical KG aggregation, LCA retrieval, and situation brief generation
"""

import asyncio

import networkx as nx
import numpy as np

from cortex.adapters.leanrag_adapter import build_situation_brief
from cortex.kg.leanrag import LeanRAG
from cortex.proxy.prompting import build_prompt_bundle


class DemoEmbedder:
    """Simple embedder for demonstration purposes"""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self._word_vectors = {}

    def embed_text(self, text: str) -> np.ndarray:
        """Generate consistent embeddings based on text content"""
        if text in self._word_vectors:
            return self._word_vectors[text]

        # Simple hash-based embedding for demo consistency
        hash_val = hash(text) % (2**31)
        np.random.seed(hash_val)
        embedding = np.random.normal(0, 1, self.dimension)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        self._word_vectors[text] = embedding
        return embedding


def create_demo_knowledge_graph() -> nx.DiGraph:
    """Create a realistic knowledge graph for demonstration"""
    G = nx.DiGraph()

    # Add hierarchical nodes representing an AI agent system
    nodes = [
        # High-level concepts
        (
            "ai_agent_system",
            "concept",
            "An intelligent software system capable of autonomous reasoning and action",
        ),
        (
            "cognitive_architecture",
            "concept",
            "The underlying structure that enables intelligent behavior",
        ),
        (
            "knowledge_management",
            "concept",
            "Systems for organizing and retrieving information",
        ),
        # Mid-level components
        (
            "neural_networks",
            "component",
            "Computational models inspired by biological neural networks",
        ),
        (
            "knowledge_graphs",
            "component",
            "Graph-based knowledge representation systems",
        ),
        (
            "natural_language_processing",
            "component",
            "AI systems that understand and generate human language",
        ),
        (
            "planning_algorithms",
            "component",
            "Algorithms for goal-oriented decision making",
        ),
        # Specific implementations
        (
            "transformer_models",
            "implementation",
            "Attention-based neural network architectures",
        ),
        (
            "graph_neural_networks",
            "implementation",
            "Neural networks that operate on graph-structured data",
        ),
        (
            "retrieval_augmented_generation",
            "implementation",
            "Systems that combine retrieval with generation",
        ),
        (
            "hierarchical_planning",
            "implementation",
            "Multi-level planning approaches",
        ),
        # Concrete instances
        ("gpt_models", "instance", "Large language models developed by OpenAI"),
        ("bert_models", "instance", "Bidirectional transformer models"),
        ("neo4j_database", "instance", "Graph database management system"),
        ("a_star_algorithm", "instance", "Best-first search algorithm"),
        # Specific tasks and problems
        (
            "question_answering",
            "task",
            "Automatically answering questions posed in natural language",
        ),
        (
            "code_generation",
            "task",
            "Automatically generating source code from specifications",
        ),
        (
            "knowledge_extraction",
            "task",
            "Extracting structured knowledge from unstructured text",
        ),
        (
            "multi_step_reasoning",
            "task",
            "Solving problems that require multiple reasoning steps",
        ),
        # Current situation nodes
        (
            "user_query_analysis",
            "situation",
            "Analyzing the current user's request for appropriate response",
        ),
        (
            "context_retrieval",
            "situation",
            "Retrieving relevant background information for the current task",
        ),
        (
            "response_generation",
            "situation",
            "Generating an appropriate response based on retrieved context",
        ),
    ]

    for node_id, node_type, description in nodes:
        G.add_node(node_id, type=node_type, description=description, name=node_id)

    # Add hierarchical relationships (aggregation)
    aggregation_edges = [
        # High-level to mid-level
        ("ai_agent_system", "cognitive_architecture"),
        ("ai_agent_system", "knowledge_management"),
        ("cognitive_architecture", "neural_networks"),
        ("cognitive_architecture", "planning_algorithms"),
        ("knowledge_management", "knowledge_graphs"),
        ("knowledge_management", "natural_language_processing"),
        # Mid-level to implementations
        ("neural_networks", "transformer_models"),
        ("neural_networks", "graph_neural_networks"),
        ("knowledge_graphs", "retrieval_augmented_generation"),
        ("planning_algorithms", "hierarchical_planning"),
        # Implementations to instances
        ("transformer_models", "gpt_models"),
        ("transformer_models", "bert_models"),
        ("knowledge_graphs", "neo4j_database"),
        ("hierarchical_planning", "a_star_algorithm"),
        # Tasks within components
        ("natural_language_processing", "question_answering"),
        ("natural_language_processing", "code_generation"),
        ("knowledge_management", "knowledge_extraction"),
        ("cognitive_architecture", "multi_step_reasoning"),
        # Current situation hierarchy
        ("ai_agent_system", "user_query_analysis"),
        ("knowledge_management", "context_retrieval"),
        ("cognitive_architecture", "response_generation"),
    ]

    for parent, child in aggregation_edges:
        G.add_edge(parent, child, kind="aggregates", weight=1.0)

    # Add semantic relationships (relates)
    semantic_edges = [
        # Cross-cutting relationships
        ("transformer_models", "question_answering"),
        ("graph_neural_networks", "knowledge_extraction"),
        ("retrieval_augmented_generation", "multi_step_reasoning"),
        ("gpt_models", "code_generation"),
        ("bert_models", "knowledge_extraction"),
        # Current situation relationships
        ("user_query_analysis", "question_answering"),
        ("context_retrieval", "retrieval_augmented_generation"),
        ("response_generation", "transformer_models"),
        ("multi_step_reasoning", "hierarchical_planning"),
    ]

    for source, target in semantic_edges:
        G.add_edge(source, target, kind="relates", weight=0.8)

    return G


async def demo_leanrag_pipeline():
    """Demonstrate the complete LeanRAG pipeline"""
    print("🚀 LeanRAG Live Demonstration")
    print("=" * 50)

    # Initialize components
    embedder = DemoEmbedder()
    leanrag = LeanRAG(embedder=embedder)

    # Create demo knowledge graph
    print("\n📊 Building Knowledge Graph...")
    kg = create_demo_knowledge_graph()
    print(
        f"   Created graph with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges"
    )

    # Show graph structure
    print("\n🏗️  Graph Structure Overview:")
    node_types: dict[str, list[str]] = {}
    for node, data in kg.nodes(data=True):
        node_type = data.get("type", "unknown")
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append(node)

    for node_type, nodes in node_types.items():
        print(f"   {node_type.title()}: {len(nodes)} nodes")
        if len(nodes) <= 5:
            print(f"      {', '.join(nodes)}")
        else:
            print(f"      {', '.join(nodes[:3])}, ... (+{len(nodes) - 3} more)")

    # Build hierarchical embeddings
    print("\n🧠 Building Hierarchical Embeddings...")
    hierarchy_graph = leanrag.build_hierarchy(kg)
    print(f"   Hierarchy built with {hierarchy_graph.number_of_nodes()} nodes")

    # Demonstrate retrieval for different scenarios
    scenarios = [
        {
            "name": "Code Generation Task",
            "query": "How can I generate Python code automatically using AI?",
        },
        {
            "name": "Knowledge Extraction Challenge",
            "query": "I need to extract structured information from research papers",
        },
        {
            "name": "Multi-Step Reasoning Problem",
            "query": "Help me solve complex problems that require multiple reasoning steps",
        },
        {
            "name": "Current Agent Situation",
            "query": "Analyze the user's current request and provide contextual response",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 Scenario {i}: {scenario['name']}")
        print(f"   Query: '{scenario['query']}'")

        # Retrieve relevant subgraph using LeanRAG
        result = leanrag.retrieve(hierarchy_graph, scenario["query"])
        subgraph = result["subgraph"]
        print(
            f"   Retrieved subgraph with {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges"
        )

        # Show retrieved nodes by type
        retrieved_types: dict[str, list[str]] = {}
        for node in subgraph.nodes():
            node_data = kg.nodes[node]
            node_type = node_data.get("type", "unknown")
            if node_type not in retrieved_types:
                retrieved_types[node_type] = []
            retrieved_types[node_type].append(node)

        print("   Retrieved nodes by type:")
        for node_type, nodes in retrieved_types.items():
            print(f"      {node_type}: {', '.join(nodes)}")

        # Show situation brief
        brief = result["brief"]
        print("\n📋 Situation Brief:")
        print(f"   {brief}")

        # Show LCA analysis
        if result["lca"]:
            lca_node = result["lca"]
            lca_data = kg.nodes[lca_node]
            print(f"\n🔍 Lowest Common Ancestor: {lca_node}")
            print(f"   Description: {lca_data.get('description', 'No description')}")

        # Show seed nodes
        if result["seeds"]:
            print(f"\n🌱 Seed nodes: {[seed[0] for seed in result['seeds']]}")

        print("-" * 40)


async def demo_adapter_integration():
    """Demonstrate the adapter integration with real queries"""
    print("\n🔗 Adapter Integration Demonstration")
    print("=" * 50)

    # Create demo knowledge graph
    kg = create_demo_knowledge_graph()
    embedder = DemoEmbedder()

    queries = [
        "Help me implement a neural network for text classification",
        "I need to build a knowledge graph database for my research project",
        "How can I create an AI agent that can plan multi-step tasks?",
        "Debug this Python code that's supposed to generate embeddings",
    ]

    print("\n📨 Processing Queries through Adapter...")
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: '{query}'")

        # Use the adapter to generate situation brief
        result = build_situation_brief(kg, query, embedder)

        print(f"Adapter enabled: {result['enabled']}")
        if result["enabled"]:
            print(f"Brief: {result['brief']}")
            print(f"Strategy: {result['strategy']}")
            print(f"Subgraph size: {result['subgraph_size']}")
            if result["lca"]:
                print(f"LCA: {result['lca']}")
            print(f"Seeds: {[seed[0] for seed in result['seeds']]}")


async def demo_prompt_builder():
    """Demonstrate the prompt builder with LeanRAG context"""
    print("\n📝 Prompt Builder Demonstration")
    print("=" * 50)

    # Build prompt bundle
    history = [
        {"role": "user", "content": "Generate a Python function for graph traversal"},
        {
            "role": "assistant",
            "content": "I'll help you create a graph traversal function.",
        },
    ]

    tools = []  # Simplified for demo

    context_text = "The user needs help with graph algorithms, specifically traversal methods that can be used in AI planning systems."

    reminders_text = None

    prompt_bundle = build_prompt_bundle(
        history=history,
        tools=tools,
        context_text=context_text,
        reminders_text=reminders_text,
    )

    print("\n📋 Generated Prompt Bundle:")
    print(f"System Prompt:\n{prompt_bundle.system}")
    print(f"\nContext:\n{prompt_bundle.context}")
    print(f"\nHistory length: {len(prompt_bundle.history)}")
    print(f"Tools: {len(prompt_bundle.tools)}")
    print(f"Metadata: {prompt_bundle.metadata}")


async def main():
    """Run the complete LeanRAG demonstration"""
    try:
        # Core LeanRAG pipeline
        await demo_leanrag_pipeline()

        # Adapter integration
        await demo_adapter_integration()

        # Prompt builder integration
        await demo_prompt_builder()

        print("\n✅ LeanRAG Demonstration Complete!")
        print("\nKey Features Demonstrated:")
        print("• Hierarchical knowledge graph construction")
        print("• LCA-based concept retrieval")
        print("• Situation brief generation")
        print("• Adapter integration with real queries")
        print("• Prompt builder with context integration")
        print("• Multi-scenario reasoning support")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
