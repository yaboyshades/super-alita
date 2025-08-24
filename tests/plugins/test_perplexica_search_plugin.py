"""
Tests for PerplexicaSearchPlugin
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.plugins.perplexica_search_plugin import (
    PerplexicaSearchPlugin,
    SearchMode,
    SearchResult,
    PerplexicaResponse,
)


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    event_bus = MagicMock()
    event_bus.subscribe = AsyncMock()
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def mock_store():
    """Mock store for testing."""
    return MagicMock()


@pytest.fixture
def mock_web_agent():
    """Mock WebAgentAtom for testing."""
    web_agent = MagicMock()
    web_agent.call = AsyncMock(return_value={
        "web": [
            {
                "title": "Test Result 1",
                "url": "https://example.com/1",
                "snippet": "This is a test snippet",
                "source": "web"
            },
            {
                "title": "Test Result 2", 
                "url": "https://example.com/2",
                "snippet": "Another test snippet",
                "source": "web"
            }
        ]
    })
    web_agent._searxng_search = AsyncMock(return_value=[
        {
            "title": "SearXNG Result",
            "url": "https://example.com/searxng",
            "snippet": "SearXNG test snippet",
            "source": "web"
        }
    ])
    return web_agent


@pytest.fixture
def plugin(mock_event_bus, mock_store, mock_web_agent):
    """Create a PerplexicaSearchPlugin instance for testing."""
    plugin = PerplexicaSearchPlugin()
    config = {"web_agent": mock_web_agent}
    # Don't await setup in fixture - do it in tests that need it
    plugin.event_bus = mock_event_bus
    plugin.store = mock_store
    plugin.config = config
    plugin.web_agent = mock_web_agent
    return plugin


class TestPerplexicaSearchPlugin:
    """Test cases for PerplexicaSearchPlugin."""

    def test_plugin_properties(self, plugin):
        """Test basic plugin properties."""
        assert plugin.name == "perplexica_search"
        assert "AI-powered search" in plugin.description
        assert plugin.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_setup(self, mock_event_bus, mock_store, mock_web_agent):
        """Test plugin setup."""
        plugin = PerplexicaSearchPlugin()
        config = {"web_agent": mock_web_agent}
        
        await plugin.setup(mock_event_bus, mock_store, config)
        
        assert plugin.event_bus == mock_event_bus
        assert plugin.store == mock_store
        assert plugin.web_agent == mock_web_agent

    @pytest.mark.asyncio
    async def test_start(self, plugin, mock_event_bus):
        """Test plugin start."""
        await plugin.start()
        
        assert plugin.is_running
        # Should subscribe to search events
        mock_event_bus.subscribe.assert_any_call("perplexica_search", plugin._handle_search_request)
        mock_event_bus.subscribe.assert_any_call("enhanced_search", plugin._handle_search_request)

    @pytest.mark.asyncio
    async def test_basic_web_search(self, plugin):
        """Test basic web search functionality."""
        query = "test search query"
        
        response = await plugin.search(
            query=query,
            search_mode=SearchMode.WEB,
            max_results=5,
            include_reasoning=False
        )
        
        assert isinstance(response, PerplexicaResponse)
        assert response.query == query
        assert response.search_mode == SearchMode.WEB
        assert len(response.sources) > 0
        assert response.total_results > 0
        assert response.confidence_score > 0

    @pytest.mark.asyncio
    async def test_search_modes(self, plugin):
        """Test different search modes."""
        query = "artificial intelligence"
        
        for mode in SearchMode:
            response = await plugin.search(
                query=query,
                search_mode=mode,
                max_results=3,
                include_reasoning=False
            )
            
            assert response.search_mode == mode
            assert response.query == query

    @pytest.mark.asyncio
    async def test_search_with_web_agent(self, plugin, mock_web_agent):
        """Test search using WebAgentAtom."""
        query = "machine learning"
        
        response = await plugin.search(query, SearchMode.WEB, include_reasoning=False)
        
        # Should have called the web agent
        mock_web_agent.call.assert_called_once_with(query, web_k=10, github_k=0)
        
        # Should have results from web agent
        assert len(response.sources) == 2
        assert response.sources[0].title == "Test Result 1"
        assert response.sources[1].title == "Test Result 2"

    @pytest.mark.asyncio
    async def test_fallback_search(self, plugin, mock_web_agent):
        """Test fallback search when WebAgentAtom fails."""
        query = "test query"
        
        # Make web agent fail
        mock_web_agent.call.side_effect = Exception("Web agent failed")
        
        response = await plugin.search(query, SearchMode.WEB, include_reasoning=False)
        
        # Should still get a response using fallback
        assert isinstance(response, PerplexicaResponse)
        assert response.query == query

    @pytest.mark.asyncio
    async def test_search_result_parsing(self, plugin):
        """Test parsing of search results."""
        raw_results = [
            {
                "title": "Test Title",
                "url": "https://test.com",
                "snippet": "Test snippet",
                "source": "web",
                "relevance_score": 0.9
            }
        ]
        
        # Mock the mode handler to return our test data
        plugin.mode_handlers[SearchMode.WEB] = AsyncMock(return_value=raw_results)
        
        response = await plugin.search("test", SearchMode.WEB, include_reasoning=False)
        
        assert len(response.sources) == 1
        result = response.sources[0]
        assert isinstance(result, SearchResult)
        assert result.title == "Test Title"
        assert result.url == "https://test.com"
        assert result.snippet == "Test snippet"
        assert result.relevance_score == 0.9

    @pytest.mark.asyncio
    async def test_result_deduplication(self, plugin):
        """Results with same domain (ignoring 'www') and title are deduplicated."""
        raw_results = [
            {
                "title": "Duplicate Title",
                "url": "https://www.example.com/1",
                "snippet": "One",
                "source": "web",
                "relevance_score": 0.4,
            },
            {
                "title": "Duplicate Title",
                "url": "https://example.com/2",
                "snippet": "Two",
                "source": "web",
                "relevance_score": 0.9,
            },
            {
                "title": "Unique Title",
                "url": "https://other.com/3",
                "snippet": "Three",
                "source": "web",
                "relevance_score": 0.8,
            },
        ]

        plugin.mode_handlers[SearchMode.WEB] = AsyncMock(return_value=raw_results)

        response = await plugin.search("dup test", SearchMode.WEB, include_reasoning=False)

        assert len(response.sources) == 2
        urls = [r.url for r in response.sources]
        assert "https://example.com/2" in urls  # higher score kept
        assert "https://example.com/1" not in urls
        assert "https://other.com/3" in urls

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, plugin):
        """Test confidence score calculation."""
        # Test with no results
        assert plugin._calculate_confidence([]) == 0.0
        
        # Test with some results
        results = [
            SearchResult(title="Test", url="http://test.com", snippet="Test", source="web", relevance_score=0.8),
            SearchResult(title="Test2", url="http://test2.com", snippet="Test2", source="web", relevance_score=0.9)
        ]
        
        confidence = plugin._calculate_confidence(results)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.0

    @pytest.mark.asyncio
    async def test_simple_summary_generation(self, plugin):
        """Test simple summary generation."""
        query = "test query"
        results = [
            SearchResult(title="Test Result", url="http://test.com", snippet="Test", source="web")
        ]
        
        summary = plugin._generate_simple_summary(query, results)
        assert query in summary
        assert "Test Result" in summary

    @pytest.mark.asyncio
    async def test_simple_analysis_generation(self, plugin):
        """Test simple analysis when AI is not available."""
        query = "test query"
        results = [
            SearchResult(title="Test Result", url="http://test.com", snippet="Test", source="web")
        ]
        
        summary, reasoning, citations, follow_ups = plugin._generate_simple_analysis(
            query, results, SearchMode.WEB
        )
        
        assert query in summary
        assert "1" in reasoning  # Should mention number of results
        assert len(citations) > 0
        assert len(follow_ups) > 0

    @pytest.mark.asyncio
    async def test_llm_context_preparation(self, plugin):
        """Test LLM context preparation."""
        query = "test query"
        results = [
            SearchResult(title="Result 1", url="http://test1.com", snippet="Snippet 1", source="web"),
            SearchResult(title="Result 2", url="http://test2.com", snippet="Snippet 2", source="web")
        ]
        
        context = plugin._prepare_llm_context(query, results, SearchMode.WEB)
        
        assert "Result 1" in context
        assert "Result 2" in context
        assert "Snippet 1" in context
        assert "http://test1.com" in context

    @pytest.mark.asyncio
    async def test_search_history(self, plugin):
        """Test search history functionality."""
        # Initially empty
        history = await plugin.get_search_history()
        assert len(history) == 0
        
        # Perform a search
        await plugin.search("test query", include_reasoning=False)
        
        # Should have history
        history = await plugin.get_search_history()
        assert len(history) == 1
        assert history[0].query == "test query"

    @pytest.mark.asyncio
    async def test_handle_search_request_event(self, plugin, mock_event_bus):
        """Test handling of search request events."""
        # Mock event
        event_data = {
            "query": "test search",
            "search_mode": "web",
            "max_results": 5,
            "session_id": "test_session"
        }
        
        mock_event = MagicMock()
        mock_event.data = event_data
        
        # Handle the event
        await plugin._handle_search_request(mock_event)
        
        # Should have published a result
        assert mock_event_bus.publish.call_count >= 1

    @pytest.mark.asyncio
    async def test_handle_empty_query_event(self, plugin):
        """Test handling of event with empty query."""
        mock_event = MagicMock()
        mock_event.data = {"query": ""}
        
        # Should not crash with empty query
        await plugin._handle_search_request(mock_event)

    def test_get_tools(self, plugin):
        """Test tool definitions."""
        tools = plugin.get_tools()
        
        assert len(tools) == 1
        tool = tools[0]
        assert tool["name"] == "perplexica_search"
        assert "query" in tool["parameters"]
        assert "search_mode" in tool["parameters"]

    @pytest.mark.asyncio
    async def test_health_check(self, plugin):
        """Test health check functionality."""
        health = await plugin.health_check()
        
        assert health["plugin"] == "perplexica_search"
        assert "llm_available" in health
        assert "web_agent_available" in health
        assert "supported_modes" in health
        assert len(health["supported_modes"]) == len(SearchMode)

    @pytest.mark.asyncio
    async def test_academic_search_mode(self, plugin):
        """Test academic search mode."""
        query = "machine learning research"
        
        response = await plugin.search(
            query=query,
            search_mode=SearchMode.ACADEMIC,
            max_results=3,
            include_reasoning=False
        )
        
        assert response.search_mode == SearchMode.ACADEMIC
        assert response.query == query

    @pytest.mark.asyncio
    async def test_error_handling_in_search(self, plugin, mock_web_agent):
        """Test error handling during search."""
        # Make web agent raise an exception
        mock_web_agent.call.side_effect = Exception("Search failed")
        
        # Should not raise exception but handle gracefully
        response = await plugin.search("test query", include_reasoning=False)
        
        assert isinstance(response, PerplexicaResponse)
        # Should still have some response even if search failed

    @pytest.mark.asyncio 
    async def test_ai_analysis_fallback(self, plugin):
        """Test AI analysis fallback when LLM fails."""
        query = "test query"
        results = [SearchResult(title="Test", url="http://test.com", snippet="Test", source="web")]
        
        # Ensure LLM client is None
        plugin.llm_client = None
        
        summary, reasoning, citations, follow_ups = await plugin._generate_ai_analysis(
            query, results, SearchMode.WEB
        )
        
        # Should get simple analysis
        assert summary is not None
        assert reasoning is not None
        assert len(citations) > 0
        assert len(follow_ups) > 0