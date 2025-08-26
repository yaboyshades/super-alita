#!/usr/bin/env python3
"""
Perplexica Integration Demo for Super Alita

Demonstrates the new AI-powered search capabilities integrated into the agent system.
"""

import asyncio
import logging
import os
from typing import Any, Dict

from src.plugins.perplexica_search_plugin import PerplexicaSearchPlugin, SearchMode
from src.atoms.web_agent_atom import WebAgentAtom
from src.core.events import create_event

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockEventBus:
    """Mock event bus for demonstration."""
    
    def __init__(self):
        self.events = []
        
    async def subscribe(self, event_type: str, handler):
        logger.info(f"Subscribed to {event_type}")
        
    async def publish(self, event):
        self.events.append(event)
        logger.info(f"Published event: {event.event_type}")
        
    async def emit(self, event_type: str, **kwargs):
        event = create_event(event_type, **kwargs)
        await self.publish(event)


class MockStore:
    """Mock store for demonstration."""
    
    def __init__(self):
        self.data = {}
        
    async def upsert(self, **kwargs):
        logger.info(f"Stored data: {kwargs.get('content', {}).get('type', 'unknown')}")


async def demo_perplexica_basic_search():
    """Demonstrate basic Perplexica search functionality."""
    print("\n🔍 === Basic Perplexica Search Demo ===")
    
    # Create mock dependencies
    event_bus = MockEventBus()
    store = MockStore()
    
    # Create and setup plugin
    plugin = PerplexicaSearchPlugin()
    config = {}
    await plugin.setup(event_bus, store, config)
    await plugin.start()
    
    # Perform different types of searches
    queries = [
        ("artificial intelligence latest developments", SearchMode.WEB),
        ("machine learning research papers", SearchMode.ACADEMIC),
        ("python programming tutorial", SearchMode.VIDEO),
        ("cryptocurrency news", SearchMode.NEWS),
    ]
    
    for query, mode in queries:
        print(f"\n📊 Searching: '{query}' (mode: {mode})")
        
        response = await plugin.search(
            query=query,
            search_mode=mode,
            max_results=5,
            include_reasoning=True
        )
        
        print(f"✅ Summary: {response.summary}")
        print(f"🧠 Reasoning: {response.reasoning[:100]}...")
        print(f"📚 Sources: {len(response.sources)} results")
        print(f"📖 Citations: {len(response.citations)} citations")
        print(f"❓ Follow-ups: {len(response.follow_up_questions)} questions")
        print(f"🎯 Confidence: {response.confidence_score:.2f}")
        print(f"⏱️ Time: {response.processing_time:.2f}s")
    
    await plugin.shutdown()


async def demo_perplexica_with_web_agent():
    """Demonstrate Perplexica integration with existing WebAgentAtom."""
    print("\n🔗 === Perplexica + WebAgent Integration Demo ===")
    
    # Create web agent
    web_agent = WebAgentAtom()
    
    # Create mock dependencies
    event_bus = MockEventBus()
    store = MockStore()
    
    # Setup web agent (minimal setup for demo)
    web_agent.event_bus = event_bus
    web_agent.store = store
    
    # Create Perplexica plugin with web agent integration
    plugin = PerplexicaSearchPlugin()
    config = {"web_agent": web_agent}
    await plugin.setup(event_bus, store, config)
    await plugin.start()
    
    # Perform search that will use WebAgentAtom backend
    query = "Super Alita agent architecture"
    
    print(f"🔍 Searching with WebAgent backend: '{query}'")
    
    response = await plugin.search(
        query=query,
        search_mode=SearchMode.WEB,
        max_results=5,
        include_reasoning=True
    )
    
    print(f"✅ Enhanced Search Results:")
    print(f"   Summary: {response.summary}")
    print(f"   Sources: {len(response.sources)}")
    print(f"   AI Reasoning Available: {bool(response.reasoning)}")
    print(f"   Citations: {len(response.citations)}")
    
    # Show search history
    history = await plugin.get_search_history()
    print(f"📝 Search History: {len(history)} searches")
    
    await plugin.shutdown()


async def demo_perplexica_event_integration():
    """Demonstrate Perplexica event-driven integration."""
    print("\n📡 === Event-Driven Search Demo ===")
    
    # Create components
    event_bus = MockEventBus()
    store = MockStore()
    plugin = PerplexicaSearchPlugin()
    
    # Setup plugin
    config = {}
    await plugin.setup(event_bus, store, config)
    await plugin.start()
    
    # Simulate search request event
    search_event = {
        "query": "quantum computing breakthroughs",
        "search_mode": "academic",
        "max_results": 3,
        "session_id": "demo_session"
    }
    
    print(f"📨 Simulating search request event...")
    print(f"   Query: {search_event['query']}")
    print(f"   Mode: {search_event['search_mode']}")
    
    # Handle the event (simulates external request)
    mock_event = type('MockEvent', (), {'data': search_event})()
    await plugin._handle_search_request(mock_event)
    
    # Check that result events were published
    print(f"📤 Events published: {len(event_bus.events)}")
    for event in event_bus.events:
        if hasattr(event, 'event_type'):
            print(f"   - {event.event_type}")
    
    await plugin.shutdown()


async def demo_perplexica_tools_integration():
    """Demonstrate Perplexica tools for MCP integration."""
    print("\n🛠️ === Tools Integration Demo ===")
    
    # Create plugin
    plugin = PerplexicaSearchPlugin()
    
    # Get available tools
    tools = plugin.get_tools()
    
    print(f"🔧 Available Tools: {len(tools)}")
    for tool in tools:
        print(f"   - {tool['name']}: {tool['description']}")
        print(f"     Parameters: {list(tool['parameters'].keys())}")
    
    # Show health status
    event_bus = MockEventBus()
    store = MockStore()
    await plugin.setup(event_bus, store, {})
    
    health = await plugin.health_check()
    print(f"\n🏥 Health Status:")
    print(f"   Plugin: {health['plugin']}")
    print(f"   Status: {health['status']}")
    print(f"   LLM Available: {health['llm_available']}")
    print(f"   Web Agent Available: {health['web_agent_available']}")
    print(f"   Supported Modes: {health['supported_modes']}")


async def demo_comparison_with_basic_search():
    """Compare Perplexica vs basic WebAgent search."""
    print("\n⚖️ === Perplexica vs Basic Search Comparison ===")
    
    query = "machine learning frameworks comparison"
    
    # Basic WebAgent search
    print("\n🔍 Basic WebAgent Search:")
    web_agent = WebAgentAtom()
    try:
        basic_result = await web_agent.call(query, web_k=3, github_k=0)
        print(f"   Results: {len(basic_result.get('web', []))}")
        print(f"   Summary: {basic_result.get('summary', 'No summary')}")
        print("   Features: Basic search results only")
    except Exception as e:
        print(f"   Basic search failed: {e}")
        print("   (Expected if SearXNG not available)")
    
    # Perplexica enhanced search
    print("\n🧠 Perplexica Enhanced Search:")
    event_bus = MockEventBus()
    store = MockStore()
    plugin = PerplexicaSearchPlugin()
    await plugin.setup(event_bus, store, {})
    await plugin.start()
    
    enhanced_result = await plugin.search(
        query=query,
        search_mode=SearchMode.WEB,
        max_results=3,
        include_reasoning=True
    )
    
    print(f"   Results: {len(enhanced_result.sources)}")
    print(f"   Summary: {enhanced_result.summary}")
    print(f"   AI Reasoning: {'✅' if enhanced_result.reasoning else '❌'}")
    print(f"   Citations: {len(enhanced_result.citations)}")
    print(f"   Follow-ups: {len(enhanced_result.follow_up_questions)}")
    print(f"   Confidence: {enhanced_result.confidence_score:.2f}")
    print("   Features: AI analysis, reasoning, citations, follow-ups")
    
    await plugin.shutdown()


async def main():
    """Run all Perplexica integration demos."""
    print("🚀 Super Alita Perplexica Integration Demo")
    print("=" * 60)
    
    try:
        await demo_perplexica_basic_search()
        await demo_perplexica_with_web_agent()
        await demo_perplexica_event_integration()
        await demo_perplexica_tools_integration()
        await demo_comparison_with_basic_search()
        
        print("\n✅ All Perplexica Integration Demos Complete!")
        print("\n🎯 Key Benefits Demonstrated:")
        print("• AI-powered search with reasoning and summarization")
        print("• Multiple search modes (web, academic, video, news, etc.)")
        print("• Source citation and fact-checking capabilities")
        print("• Follow-up question generation")
        print("• Integration with existing WebAgentAtom")
        print("• Event-driven architecture compatibility")
        print("• MCP tool integration for VS Code")
        print("• Enhanced confidence scoring")
        print("• Search history and analytics")
        
        print("\n🔧 Usage in Super Alita:")
        print("1. Import PerplexicaSearchPlugin in your agent setup")
        print("2. Configure with existing WebAgentAtom for backend search")
        print("3. Subscribe to 'perplexica_search' events")
        print("4. Use search modes for different types of queries")
        print("5. Access AI reasoning and citations in responses")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())