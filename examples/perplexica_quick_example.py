#!/usr/bin/env python3
"""
Quick Example: Using Perplexica Search in Super Alita

This example shows how to use the new Perplexica search capabilities 
that are now integrated into the Super Alita agent.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plugins.perplexica_search_plugin import PerplexicaSearchPlugin, SearchMode

# Setup logging
logging.basicConfig(level=logging.INFO)


async def main():
    """Quick example of using Perplexica search capabilities."""
    print("🔍 Super Alita Perplexica Search Example")
    print("=" * 50)
    
    # Create the plugin (would normally be done by the agent system)
    plugin = PerplexicaSearchPlugin()
    
    # Mock setup (in real usage, this is handled by the agent)
    class MockEventBus:
        async def subscribe(self, event_type, handler): pass
        async def publish(self, event): pass
    
    await plugin.setup(MockEventBus(), None, {})
    await plugin.start()
    
    # Example 1: Basic web search with AI reasoning
    print("\n📊 Example 1: Web Search with AI Analysis")
    response = await plugin.search(
        query="latest developments in artificial intelligence 2024",
        search_mode=SearchMode.WEB,
        max_results=5,
        include_reasoning=True
    )
    
    print(f"Query: {response.query}")
    print(f"Summary: {response.summary}")
    print(f"AI Reasoning: {response.reasoning[:200]}...")
    print(f"Results: {len(response.sources)} sources found")
    print(f"Confidence: {response.confidence_score:.2f}")
    
    # Example 2: Academic search
    print("\n📚 Example 2: Academic Search")
    response = await plugin.search(
        query="machine learning transformers research",
        search_mode=SearchMode.ACADEMIC,
        max_results=3,
        include_reasoning=True
    )
    
    print(f"Query: {response.query}")
    print(f"Summary: {response.summary}")
    print(f"Citations: {len(response.citations)} available")
    
    # Example 3: Show available tools for MCP integration
    print("\n🛠️ Example 3: Available Tools")
    tools = plugin.get_tools()
    for tool in tools:
        print(f"Tool: {tool['name']}")
        print(f"Description: {tool['description']}")
        print(f"Parameters: {list(tool['parameters'].keys())}")
    
    # Example 4: Check search history
    print("\n📝 Example 4: Search History")
    history = await plugin.get_search_history()
    print(f"Recent searches: {len(history)}")
    for i, search in enumerate(history[-3:], 1):  # Show last 3
        print(f"  {i}. {search.query} ({search.search_mode}) - {search.confidence_score:.2f}")
    
    await plugin.shutdown()
    
    print("\n✅ Example Complete!")
    print("\nNext Steps:")
    print("1. The plugin is now integrated into main_unified.py")
    print("2. It will automatically start with the Super Alita agent")
    print("3. You can use it via events: emit('perplexica_search', {query: '...'})")
    print("4. It's available as an MCP tool for VS Code integration")
    print("5. It provides AI reasoning on top of basic search results")


if __name__ == "__main__":
    asyncio.run(main())