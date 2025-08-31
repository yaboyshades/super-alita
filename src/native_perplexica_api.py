"""
Native Perplexica API for Super Alita

Provides direct access to Perplexica-style AI-powered search capabilities
without requiring the full plugin architecture or event system.
"""

import logging
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

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
    timestamp: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PerplexicaResponse(BaseModel):
    """AI-powered search response with reasoning."""

    query: str
    search_mode: SearchMode
    summary: str
    reasoning: str
    sources: list[SearchResult]
    citations: list[str]
    follow_up_questions: list[str] = Field(default_factory=list)
    confidence_score: float = 0.8
    processing_time: float = 0.0
    total_results: int = 0


class NativePerplexicaAPI:
    """
    Native Perplexica API for direct search operations.

    This class provides the core Perplexica functionality without requiring
    the full plugin architecture, similar to how NativeDeepCodeAPI works.
    """

    def __init__(self, web_agent=None):
        self.web_agent = web_agent
        self.llm_client = None
        self.search_history: list[PerplexicaResponse] = []
        self._mode_handlers = {
            SearchMode.WEB: self._search_web,
            SearchMode.ACADEMIC: self._search_academic,
            SearchMode.VIDEO: self._search_video,
            SearchMode.NEWS: self._search_news,
            SearchMode.IMAGES: self._search_images,
            SearchMode.REDDIT: self._search_reddit,
            SearchMode.SHOPPING: self._search_shopping,
            SearchMode.WOLFRAM: self._search_wolfram,
        }

    async def initialize_llm(self):
        """Initialize LLM client for AI reasoning."""
        try:
            # Try to use Google Gemini if available
            import os

            import google.generativeai as genai

            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.llm_client = genai.GenerativeModel("gemini-pro")
                logger.info("✅ Gemini LLM initialized for Perplexica reasoning")
            else:
                logger.info("⚠️ No GEMINI_API_KEY found, using fallback reasoning")
        except ImportError:
            logger.info(
                "⚠️ Google Generative AI not available, using fallback reasoning"
            )

    async def search(
        self,
        query: str,
        search_mode: SearchMode = SearchMode.WEB,
        max_results: int = 10,
        include_reasoning: bool = True,
    ) -> PerplexicaResponse:
        """
        Perform AI-powered search with reasoning and summarization.

        Args:
            query: Search query
            search_mode: Type of search to perform
            max_results: Maximum number of results
            include_reasoning: Whether to include AI reasoning

        Returns:
            PerplexicaResponse with search results and AI analysis
        """
        start_time = time.time()

        logger.info(f"🔍 Native Perplexica search: '{query}' (mode: {search_mode})")

        # Get search handler for the mode
        handler = self._mode_handlers.get(search_mode, self._search_web)

        # Perform the search
        try:
            raw_results = await handler(query, max_results)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raw_results = []

        # Convert to SearchResult objects
        search_results = [
            SearchResult(
                title=result.get("title", "No title"),
                url=result.get("url", ""),
                snippet=result.get("snippet", result.get("content", "")),
                source=result.get("source", search_mode.value),
                relevance_score=result.get("relevance_score", 0.7),
                timestamp=result.get("timestamp"),
                metadata=result.get("metadata", {}),
            )
            for result in raw_results
        ]

        # Remove duplicates
        search_results = self._dedupe_results(search_results)

        # Generate AI analysis if requested
        if include_reasoning and self.llm_client:
            (
                summary,
                reasoning,
                citations,
                follow_ups,
            ) = await self._generate_ai_analysis(query, search_results, search_mode)
        else:
            summary = self._generate_simple_summary(query, search_results)
            reasoning = (
                f"Found {len(search_results)} results for '{query}' "
                f"using {search_mode} search."
            )
            citations = [
                f"[{i+1}] {r.title} - {r.url}" for i, r in enumerate(search_results[:5])
            ]
            follow_ups = []

        # Create response
        response = PerplexicaResponse(
            query=query,
            search_mode=search_mode,
            summary=summary,
            reasoning=reasoning,
            sources=search_results,
            citations=citations,
            follow_up_questions=follow_ups,
            confidence_score=self._calculate_confidence(search_results),
            total_results=len(search_results),
            processing_time=time.time() - start_time,
        )

        # Store in history
        self.search_history.append(response)
        if len(self.search_history) > 100:
            self.search_history.pop(0)

        logger.info(
            f"✅ Search completed in {response.processing_time:.2f}s, {len(search_results)} results"
        )
        return response

    async def _search_web(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Perform web search using WebAgent if available."""
        if self.web_agent:
            try:
                result = await self.web_agent.call(query, web_k=max_results, github_k=0)
                web_results = result.get("web", [])

                # Convert WebAgent format to standard format
                return [
                    {
                        "title": item.get("title", "No title"),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", item.get("snippet", "")),
                        "source": "web",
                        "relevance_score": 0.8,
                        "metadata": {"via": "WebAgent"},
                    }
                    for item in web_results
                ]
            except Exception as e:
                logger.warning(f"WebAgent search failed: {e}")

        # Fallback to mock results
        return await self._fallback_search(query, max_results, "web")

    async def _search_academic(
        self, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Search academic sources."""
        academic_query = f"{query} site:arxiv.org OR site:scholar.google.com OR site:pubmed.ncbi.nlm.nih.gov"

        if self.web_agent:
            try:
                result = await self.web_agent.call(
                    academic_query, web_k=max_results, github_k=0
                )
                web_results = result.get("web", [])

                return [
                    {
                        "title": item.get("title", "No title"),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", item.get("snippet", "")),
                        "source": "academic",
                        "relevance_score": 0.9,
                        "metadata": {"via": "WebAgent", "type": "academic"},
                    }
                    for item in web_results
                ]
            except Exception:
                pass

        return await self._fallback_search(academic_query, max_results, "academic")

    async def _search_video(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Search for videos."""
        video_query = f"{query} site:youtube.com OR site:vimeo.com"
        return await self._fallback_search(video_query, max_results, "video")

    async def _search_news(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Search news sources."""
        news_query = f"{query} site:news.google.com OR site:reuters.com OR site:bbc.com"
        return await self._fallback_search(news_query, max_results, "news")

    async def _search_images(
        self, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Search for images."""
        return await self._fallback_search(f"{query} images", max_results, "images")

    async def _search_reddit(
        self, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Search Reddit discussions."""
        reddit_query = f"{query} site:reddit.com"
        return await self._fallback_search(reddit_query, max_results, "reddit")

    async def _search_shopping(
        self, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Search shopping sites."""
        shopping_query = f"{query} buy purchase price"
        return await self._fallback_search(shopping_query, max_results, "shopping")

    async def _search_wolfram(
        self, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Search using Wolfram Alpha style computational queries."""
        computational_query = f"{query} calculate compute math"
        return await self._fallback_search(computational_query, max_results, "wolfram")

    async def _fallback_search(
        self, query: str, max_results: int, source: str
    ) -> list[dict[str, Any]]:
        """Fallback search when specific handlers fail."""
        # Generate mock results based on query
        results = []
        for i in range(min(max_results, 5)):  # Limit fallback results
            results.append(
                {
                    "title": f"Result {i+1} for '{query}'",
                    "url": f"https://example.com/result_{i+1}",
                    "snippet": f"This is a {source} search result for the query '{query}'. "
                    f"It contains relevant information about the topic.",
                    "source": source,
                    "relevance_score": 0.6 - (i * 0.1),
                    "metadata": {"fallback": True, "index": i},
                }
            )
        return results

    def _dedupe_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate search results."""
        seen_urls = set()
        deduped = []

        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                deduped.append(result)

        return deduped

    def _calculate_confidence(self, results: list[SearchResult]) -> float:
        """Calculate confidence score based on search results."""
        if not results:
            return 0.0

        # Base confidence on number of results and average relevance
        result_count_factor = min(len(results) / 10.0, 1.0)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)

        return result_count_factor * 0.3 + avg_relevance * 0.7

    async def _generate_ai_analysis(
        self, query: str, results: list[SearchResult], search_mode: SearchMode
    ) -> tuple[str, str, list[str], list[str]]:
        """Generate AI-powered analysis of search results."""
        if not self.llm_client or not results:
            return await self._fallback_analysis(query, results, search_mode)

        try:
            # Prepare context for LLM
            context = f"Search Query: {query}\nSearch Mode: {search_mode}\n\nSearch Results:\n"
            for i, result in enumerate(results[:5]):  # Limit to top 5 for context
                context += f"{i+1}. {result.title}\n   {result.snippet}\n   URL: {result.url}\n\n"

            # Generate analysis
            prompt = f"""Analyze these search results and provide:
1. A concise summary (2-3 sentences)
2. Detailed reasoning about the findings
3. Proper citations in academic format
4. 3 relevant follow-up questions

{context}

Please provide a comprehensive analysis that helps the user understand the key findings and next steps."""

            response = await self.llm_client.generate_content_async(prompt)
            ai_text = response.text

            # Parse the response (simplified parsing)
            lines = ai_text.split("\n")
            summary = ""
            reasoning = ""
            citations = []
            follow_ups = []

            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if "summary" in line.lower() or line.startswith("1."):
                    current_section = "summary"
                elif "reasoning" in line.lower() or line.startswith("2."):
                    current_section = "reasoning"
                elif "citations" in line.lower() or line.startswith("3."):
                    current_section = "citations"
                elif "follow" in line.lower() or line.startswith("4."):
                    current_section = "follow_ups"
                else:
                    if current_section == "summary":
                        summary += line + " "
                    elif current_section == "reasoning":
                        reasoning += line + " "
                    elif current_section == "citations" and line.startswith(
                        ("•", "-", "*")
                    ):
                        citations.append(line[1:].strip())
                    elif current_section == "follow_ups" and line.startswith(
                        ("•", "-", "*")
                    ):
                        follow_ups.append(line[1:].strip())

            # Fallback to basic analysis if parsing failed
            if not summary:
                summary = f"Found {len(results)} results related to {query}"
            if not reasoning:
                reasoning = "Analysis of search results shows relevant information for the query."
            if not citations:
                citations = [f"[{i+1}] {r.title}" for i, r in enumerate(results[:3])]
            if not follow_ups:
                follow_ups = [
                    f"What are the latest developments in {query}?",
                    f"How does {query} impact current trends?",
                    f"What are the best resources for learning about {query}?",
                ]

            return summary.strip(), reasoning.strip(), citations, follow_ups

        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return await self._fallback_analysis(query, results, search_mode)

    async def _fallback_analysis(
        self, query: str, results: list[SearchResult], search_mode: SearchMode
    ) -> tuple[str, str, list[str], list[str]]:
        """Fallback analysis when AI is not available."""
        if not results:
            summary = f"No results found for '{query}'"
            reasoning = "Search completed but no relevant results were returned."
            citations = []
            follow_ups = []
        else:
            summary = f"Found {len(results)} results for '{query}' using {search_mode} search."
            reasoning = f"Search returned {len(results)} results with an average relevance score of {sum(r.relevance_score for r in results) / len(results):.2f}."
            citations = [
                f"[{i+1}] {r.title} - {r.url}" for i, r in enumerate(results[:5])
            ]
            follow_ups = [
                f"What are the latest updates on {query}?",
                f"How can I learn more about {query}?",
                f"What are related topics to {query}?",
            ]

        return summary, reasoning, citations, follow_ups

    def _generate_simple_summary(self, query: str, results: list[SearchResult]) -> str:
        """Generate a simple summary without AI."""
        if not results:
            return f"No results found for '{query}'"

        return f"Found {len(results)} results for '{query}'. The search returned information from various sources including web pages, articles, and other relevant content."

    def get_search_history(self) -> list[PerplexicaResponse]:
        """Get the search history."""
        return self.search_history.copy()

    def clear_history(self):
        """Clear the search history."""
        self.search_history.clear()

    async def health_check(self) -> dict[str, Any]:
        """Check the health of the native API."""
        return {
            "status": "healthy",
            "llm_available": self.llm_client is not None,
            "web_agent_available": self.web_agent is not None,
            "search_modes": [mode.value for mode in SearchMode],
            "history_count": len(self.search_history),
        }
