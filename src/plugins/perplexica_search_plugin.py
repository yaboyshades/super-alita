"""
Perplexica-like Search Plugin for Super Alita

Provides AI-powered search with reasoning, summarization, and multi-modal search capabilities.
Integrates with the existing WebAgentAtom to extend search functionality.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import aiohttp
from pydantic import BaseModel, Field

from src.core.plugin_interface import PluginInterface
from src.core.events import BaseEvent, create_event

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """Available search modes similar to Perplexica."""
    WEB = "web"
    ACADEMIC = "academic"
    VIDEO = "video"
    NEWS = "news"
    IMAGES = "images"
    REDDIT = "reddit"
    SHOPPING = "shopping"
    WOLFRAM = "wolfram"


class SearchResult(BaseModel):
    """Individual search result with metadata."""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.0
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerplexicaResponse(BaseModel):
    """AI-powered search response with reasoning."""
    query: str
    search_mode: SearchMode
    summary: str
    reasoning: str
    sources: List[SearchResult]
    citations: List[str]
    follow_up_questions: List[str] = Field(default_factory=list)
    confidence_score: float = 0.8
    processing_time: float = 0.0
    total_results: int = 0


class PerplexicaSearchEvent(BaseEvent):
    """Event for requesting Perplexica-style search."""
    event_type: str = "perplexica_search"
    query: str
    search_mode: SearchMode = SearchMode.WEB
    max_results: int = 10
    include_reasoning: bool = True
    session_id: str = "default"


class PerplexicaResultEvent(BaseEvent):
    """Event containing Perplexica search results."""
    event_type: str = "perplexica_result"
    query: str
    response: PerplexicaResponse
    session_id: str = "default"


class PerplexicaSearchPlugin(PluginInterface):
    """
    Perplexica-like AI-powered search plugin.
    
    Provides advanced search capabilities with:
    - AI reasoning and summarization
    - Multiple search modes (web, academic, video, etc.)
    - Source citation and fact-checking
    - Follow-up question generation
    - Integration with existing WebAgentAtom
    """

    def __init__(self):
        super().__init__()
        self.web_agent: Optional[Any] = None
        self.llm_client: Optional[Any] = None
        self.search_history: List[PerplexicaResponse] = []
        self.mode_handlers = {
            SearchMode.WEB: self._search_web,
            SearchMode.ACADEMIC: self._search_academic,
            SearchMode.VIDEO: self._search_video,
            SearchMode.NEWS: self._search_news,
            SearchMode.IMAGES: self._search_images,
            SearchMode.REDDIT: self._search_reddit,
            SearchMode.SHOPPING: self._search_shopping,
            SearchMode.WOLFRAM: self._search_wolfram,
        }

    @property
    def name(self) -> str:
        return "perplexica_search"

    @property
    def description(self) -> str:
        return "AI-powered search with reasoning, summarization, and multi-modal capabilities"

    async def setup(self, event_bus: Any, store: Any, config: Dict[str, Any]) -> None:
        """Initialize the Perplexica search plugin."""
        await super().setup(event_bus, store, config)
        
        # Try to get reference to WebAgentAtom if available
        # This allows us to leverage existing search infrastructure
        self.web_agent = config.get("web_agent")
        
        # Initialize LLM client for reasoning (will use Gemini if available)
        await self._initialize_llm_client()

    async def start(self) -> None:
        """Start the plugin and subscribe to search events."""
        await super().start()
        
        # Subscribe to Perplexica search requests
        await self.subscribe("perplexica_search", self._handle_search_request)
        
        # Also handle generic "enhanced_search" events
        await self.subscribe("enhanced_search", self._handle_search_request)
        
        logger.info("PerplexicaSearchPlugin started - ready for AI-powered search")

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.llm_client and hasattr(self.llm_client, "close"):
            await self.llm_client.close()
        logger.info("PerplexicaSearchPlugin shutdown complete")

    async def _initialize_llm_client(self) -> None:
        """Initialize LLM client for reasoning and summarization."""
        try:
            # Try to use the existing Gemini integration
            import google.generativeai as genai
            import os
            
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key and api_key != "your-key-here":
                genai.configure(api_key=api_key)
                self.llm_client = genai.GenerativeModel('gemini-pro')
                logger.info("Initialized Gemini client for AI reasoning")
            else:
                logger.warning("No valid GEMINI_API_KEY found - reasoning will be simplified")
                
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            self.llm_client = None

    async def _handle_search_request(self, event) -> None:
        """Handle incoming search requests."""
        try:
            # Extract data from event
            if hasattr(event, "data") and isinstance(event.data, dict):
                data = event.data
            else:
                data = event.model_dump() if hasattr(event, "model_dump") else event

            # Extract search parameters
            query = data.get("query", "")
            search_mode = SearchMode(data.get("search_mode", SearchMode.WEB))
            max_results = data.get("max_results", 10)
            include_reasoning = data.get("include_reasoning", True)
            session_id = data.get("session_id", "default")

            if not query:
                logger.warning("Received search request with empty query")
                return

            logger.info(f"🔍 Processing Perplexica search: '{query}' (mode: {search_mode})")

            # Perform the search
            start_time = time.time()
            response = await self.search(
                query=query,
                search_mode=search_mode,
                max_results=max_results,
                include_reasoning=include_reasoning
            )
            response.processing_time = time.time() - start_time

            # Store in history
            self.search_history.append(response)
            if len(self.search_history) > 100:  # Keep last 100 searches
                self.search_history.pop(0)

            # Emit result event
            await self.emit_event(
                "perplexica_result",
                query=query,
                response=response.model_dump(),
                session_id=session_id,
                search_mode=search_mode.value
            )

            # Also emit as tool result for compatibility
            from src.core.events import ToolResultEvent
            
            tool_result = ToolResultEvent(
                source_plugin=self.name,
                conversation_id=session_id,
                tool_call_id=data.get("tool_call_id", f"perplexica_{session_id}"),
                session_id=session_id,
                success=True,
                result=response.model_dump()
            )
            
            await self.event_bus.publish(tool_result)

            logger.info(f"✅ Perplexica search completed in {response.processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Error handling search request: {e}")
            
            # Emit error result
            error_result = ToolResultEvent(
                source_plugin=self.name,
                conversation_id=data.get("session_id", "default"),
                tool_call_id=data.get("tool_call_id", f"perplexica_error_{data.get('session_id', 'default')}"),
                session_id=data.get("session_id", "default"),
                success=False,
                result={"error": str(e)},
                error=str(e)
            )
            
            await self.event_bus.publish(error_result)

    async def search(
        self,
        query: str,
        search_mode: SearchMode = SearchMode.WEB,
        max_results: int = 10,
        include_reasoning: bool = True
    ) -> PerplexicaResponse:
        """
        Perform AI-powered search with reasoning.
        
        Args:
            query: Search query
            search_mode: Type of search to perform
            max_results: Maximum number of results
            include_reasoning: Whether to include AI reasoning
            
        Returns:
            PerplexicaResponse with search results and AI analysis
        """
        logger.info(f"Starting {search_mode} search for: '{query}'")
        
        # Get raw search results based on mode
        handler = self.mode_handlers.get(search_mode, self._search_web)
        raw_results = await handler(query, max_results)
        
        # Convert to SearchResult objects
        search_results = [
            SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=result.get("snippet", ""),
                source=result.get("source", search_mode.value),
                relevance_score=result.get("relevance_score", 0.7),
                timestamp=result.get("timestamp"),
                metadata=result.get("metadata", {})
            )
            for result in raw_results
        ]
        
        # Generate AI reasoning and summary if requested
        if include_reasoning and self.llm_client:
            summary, reasoning, citations, follow_ups = await self._generate_ai_analysis(
                query, search_results, search_mode
            )
        else:
            summary = self._generate_simple_summary(query, search_results)
            reasoning = f"Found {len(search_results)} results for '{query}' using {search_mode} search."
            citations = [f"[{i+1}] {r.title} - {r.url}" for i, r in enumerate(search_results[:5])]
            follow_ups = []

        return PerplexicaResponse(
            query=query,
            search_mode=search_mode,
            summary=summary,
            reasoning=reasoning,
            sources=search_results,
            citations=citations,
            follow_up_questions=follow_ups,
            confidence_score=self._calculate_confidence(search_results),
            total_results=len(search_results)
        )

    async def _search_web(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform web search using existing WebAgentAtom if available."""
        if self.web_agent and hasattr(self.web_agent, "call"):
            try:
                # Use existing WebAgentAtom
                result = await self.web_agent.call(query, web_k=max_results, github_k=0)
                return result.get("web", [])
            except Exception as e:
                logger.warning(f"WebAgentAtom search failed: {e}")
        
        # Fallback to basic search implementation
        return await self._fallback_search(query, max_results, "web")

    async def _search_academic(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search academic sources."""
        # For academic search, we can add "academic" or "scholarly" terms
        academic_query = f"{query} academic research papers scholarly"
        return await self._fallback_search(academic_query, max_results, "academic")

    async def _search_video(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search for videos."""
        # Could integrate with YouTube API or use video-specific search engines
        video_query = f"{query} video tutorial"
        return await self._fallback_search(video_query, max_results, "video")

    async def _search_news(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search news sources."""
        news_query = f"{query} news recent"
        return await self._fallback_search(news_query, max_results, "news")

    async def _search_images(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search for images."""
        return await self._fallback_search(query, max_results, "images")

    async def _search_reddit(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search Reddit."""
        reddit_query = f"{query} site:reddit.com"
        return await self._fallback_search(reddit_query, max_results, "reddit")

    async def _search_shopping(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search shopping sites."""
        shopping_query = f"{query} buy purchase price"
        return await self._fallback_search(shopping_query, max_results, "shopping")

    async def _search_wolfram(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Wolfram Alpha style computational queries."""
        # This would ideally integrate with Wolfram Alpha API
        computational_query = f"{query} calculate compute math"
        return await self._fallback_search(computational_query, max_results, "wolfram")

    async def _fallback_search(self, query: str, max_results: int, source: str) -> List[Dict[str, Any]]:
        """Fallback search implementation when specialized handlers aren't available."""
        try:
            # Try to use the basic web search from WebAgentAtom
            if self.web_agent and hasattr(self.web_agent, "_searxng_search"):
                results = await self.web_agent._searxng_search(query, max_results)
                # Mark the source appropriately
                for result in results:
                    result["source"] = source
                return results
            else:
                # Return placeholder results if no search backend available
                return [
                    {
                        "title": f"Search result for: {query}",
                        "url": f"https://example.com/search?q={query}",
                        "snippet": f"This is a placeholder result for {source} search: {query}",
                        "source": source,
                        "relevance_score": 0.5
                    }
                ]
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

    async def _generate_ai_analysis(
        self, 
        query: str, 
        results: List[SearchResult], 
        search_mode: SearchMode
    ) -> tuple[str, str, List[str], List[str]]:
        """Generate AI-powered analysis of search results."""
        if not self.llm_client:
            return self._generate_simple_analysis(query, results, search_mode)

        try:
            # Prepare context for LLM
            context = self._prepare_llm_context(query, results, search_mode)
            
            # Generate reasoning and summary
            prompt = f"""
You are an AI research assistant analyzing search results. Based on the search query and results below, provide:

1. A comprehensive summary (2-3 sentences)
2. Your reasoning process
3. Key citations
4. 3 follow-up questions

Query: {query}
Search Mode: {search_mode}

Results:
{context}

Please provide a detailed analysis that synthesizes the information and highlights key findings.
"""

            response = await self.llm_client.generate_content_async(prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Parse the response (simplified for now)
            lines = response_text.split('\n')
            summary = ""
            reasoning = ""
            citations = []
            follow_ups = []
            
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if "summary" in line.lower():
                    current_section = "summary"
                elif "reasoning" in line.lower():
                    current_section = "reasoning"
                elif "citation" in line.lower():
                    current_section = "citations"
                elif "follow" in line.lower() and "question" in line.lower():
                    current_section = "follow_ups"
                elif current_section == "summary" and not summary:
                    summary = line
                elif current_section == "reasoning":
                    reasoning += line + " "
                elif current_section == "citations":
                    citations.append(line)
                elif current_section == "follow_ups":
                    follow_ups.append(line)
            
            # Fallback if parsing failed
            if not summary:
                summary = response_text[:200] + "..." if len(response_text) > 200 else response_text
            if not reasoning:
                reasoning = f"AI analysis of {len(results)} results for '{query}'"
            if not citations:
                citations = [f"[{i+1}] {r.title}" for i, r in enumerate(results[:3])]
            if not follow_ups:
                follow_ups = [
                    f"What are the latest developments in {query}?",
                    f"How does {query} compare to alternatives?",
                    f"What are the practical applications of {query}?"
                ]
            
            return summary, reasoning.strip(), citations, follow_ups
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._generate_simple_analysis(query, results, search_mode)

    def _generate_simple_analysis(
        self, 
        query: str, 
        results: List[SearchResult], 
        search_mode: SearchMode
    ) -> tuple[str, str, List[str], List[str]]:
        """Generate simple analysis when AI is not available."""
        summary = f"Found {len(results)} {search_mode} results for '{query}'. "
        if results:
            summary += f"Top result: {results[0].title}"
        
        reasoning = f"Performed {search_mode} search and retrieved {len(results)} results. "
        reasoning += "Results are ranked by relevance and recency."
        
        citations = [f"[{i+1}] {r.title} - {r.url}" for i, r in enumerate(results[:5])]
        
        follow_ups = [
            f"Can you provide more details about {query}?",
            f"What are related topics to {query}?",
            f"How can I learn more about {query}?"
        ]
        
        return summary, reasoning, citations, follow_ups

    def _prepare_llm_context(self, query: str, results: List[SearchResult], search_mode: SearchMode) -> str:
        """Prepare context for LLM analysis."""
        context_parts = []
        for i, result in enumerate(results[:5]):  # Limit to top 5 for context window
            context_parts.append(f"{i+1}. {result.title}\n   {result.snippet}\n   Source: {result.url}\n")
        
        return "\n".join(context_parts)

    def _generate_simple_summary(self, query: str, results: List[SearchResult]) -> str:
        """Generate a simple summary without AI."""
        if not results:
            return f"No results found for '{query}'"
        
        return f"Found {len(results)} results for '{query}'. Top result: {results[0].title}"

    def _calculate_confidence(self, results: List[SearchResult]) -> float:
        """Calculate confidence score based on search results."""
        if not results:
            return 0.0
        
        # Simple confidence calculation based on number and quality of results
        base_confidence = min(len(results) / 10.0, 1.0)  # More results = higher confidence
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        
        return min((base_confidence + avg_relevance) / 2.0, 1.0)

    async def get_search_history(self, limit: int = 10) -> List[PerplexicaResponse]:
        """Get recent search history."""
        return self.search_history[-limit:] if self.search_history else []

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return available tools for this plugin."""
        return [
            {
                "name": "perplexica_search",
                "description": "AI-powered search with reasoning and multi-modal capabilities",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"},
                    "search_mode": {
                        "type": "string", 
                        "enum": [mode.value for mode in SearchMode],
                        "default": "web",
                        "description": "Type of search to perform"
                    },
                    "max_results": {"type": "integer", "default": 10, "description": "Maximum results"},
                    "include_reasoning": {"type": "boolean", "default": True, "description": "Include AI reasoning"}
                }
            }
        ]

    async def health_check(self) -> Dict[str, Any]:
        """Return health status for this plugin."""
        base_health = await super().health_check()
        base_health.update({
            "llm_available": self.llm_client is not None,
            "web_agent_available": self.web_agent is not None,
            "search_history_count": len(self.search_history),
            "supported_modes": [mode.value for mode in SearchMode]
        })
        return base_health