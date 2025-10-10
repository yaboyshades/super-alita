"""
Native Perplexica Plugin for Super Alita

Provides AI-powered search capabilities without requiring external services.
Integrates directly with the event system and provides MCP tools.
"""

import logging
from typing import Any

from src.core.plugin_interface import PluginInterface
from src.native_perplexica_api import (
    NativePerplexicaAPI,
    PerplexicaResponse,
    SearchMode,
)

logger = logging.getLogger(__name__)


class NativePerplexicaPlugin(PluginInterface):
    """
    Native Perplexica plugin that provides AI-powered search capabilities.

    Similar to NativeDeepCodePlugin, this plugin uses direct function calls
    instead of events for better performance and reliability.
    """

    def __init__(self):
        self.name = "native_perplexica"
        self.api = NativePerplexicaAPI()
        self.event_bus = None
        self.store = None
        self.config = {}

    async def setup(self, event_bus, store=None, config=None):
        """Setup the plugin with dependencies."""
        self.event_bus = event_bus
        self.store = store
        self.config = config or {}

        # Initialize the native API
        await self.api.initialize_llm()

        logger.info("✅ Native Perplexica plugin setup complete")

    async def start(self):
        """Start the plugin."""
        logger.info("🔍 Native Perplexica plugin started")

    async def stop(self):
        """Stop the plugin."""
        logger.info("🛑 Native Perplexica plugin stopped")

    def get_tools(self) -> list[dict[str, Any]]:
        """Get MCP tools provided by this plugin."""
        return [
            {
                "name": "search_web",
                "description": "Search the web for information using AI-powered search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_academic",
                "description": (
                    "Search academic sources (arXiv, Google Scholar, PubMed) "
                    "for research papers and academic content"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Academic search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_news",
                "description": "Search news sources for current events and news",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "News search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_reddit",
                "description": "Search Reddit discussions and communities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Reddit search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_images",
                "description": "Search for images related to a query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Image search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_videos",
                "description": "Search for videos on YouTube and other platforms",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Video search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_perplexica",
                "description": (
                    "General AI-powered search with reasoning and analysis. "
                    "Supports multiple search modes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "search_mode": {
                            "type": "string",
                            "description": "Search mode",
                            "enum": [
                                "web",
                                "academic",
                                "video",
                                "news",
                                "images",
                                "reddit",
                                "shopping",
                                "wolfram",
                            ],
                            "default": "web",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                        "include_reasoning": {
                            "type": "boolean",
                            "description": "Include AI reasoning and analysis",
                            "default": True,
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

    def get_tool_executor(self, tool_name: str):
        """Get the executor function for a specific tool."""
        executor_map = {
            "search_web": self.search_web,
            "search_academic": self.search_academic,
            "search_news": self.search_news,
            "search_reddit": self.search_reddit,
            "search_images": self.search_images,
            "search_videos": self.search_videos,
            "search_perplexica": self.search_perplexica,
        }
        return executor_map.get(tool_name)

    async def search_web(
        self, query: str, max_results: int = 10
    ) -> dict[str, Any]:
        """Search the web for information."""
        try:
            response = await self.api.search(
                query=query,
                search_mode=SearchMode.WEB,
                max_results=max_results,
            )
            return self._format_response(response)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {"error": str(e), "results": []}

    async def search_academic(
        self, query: str, max_results: int = 10
    ) -> dict[str, Any]:
        """Search academic sources."""
        try:
            response = await self.api.search(
                query=query,
                search_mode=SearchMode.ACADEMIC,
                max_results=max_results,
            )
            return self._format_response(response)
        except Exception as e:
            logger.error(f"Academic search failed: {e}")
            return {"error": str(e), "results": []}

    async def search_news(
        self, query: str, max_results: int = 10
    ) -> dict[str, Any]:
        """Search news sources."""
        try:
            response = await self.api.search(
                query=query,
                search_mode=SearchMode.NEWS,
                max_results=max_results,
            )
            return self._format_response(response)
        except Exception as e:
            logger.error(f"News search failed: {e}")
            return {"error": str(e), "results": []}

    async def search_reddit(
        self, query: str, max_results: int = 10
    ) -> dict[str, Any]:
        """Search Reddit discussions."""
        try:
            response = await self.api.search(
                query=query,
                search_mode=SearchMode.REDDIT,
                max_results=max_results,
            )
            return self._format_response(response)
        except Exception as e:
            logger.error(f"Reddit search failed: {e}")
            return {"error": str(e), "results": []}

    async def search_images(
        self, query: str, max_results: int = 10
    ) -> dict[str, Any]:
        """Search for images."""
        try:
            response = await self.api.search(
                query=query,
                search_mode=SearchMode.IMAGES,
                max_results=max_results,
            )
            return self._format_response(response)
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return {"error": str(e), "results": []}

    async def search_videos(
        self, query: str, max_results: int = 10
    ) -> dict[str, Any]:
        """Search for videos."""
        try:
            response = await self.api.search(
                query=query,
                search_mode=SearchMode.VIDEO,
                max_results=max_results,
            )
            return self._format_response(response)
        except Exception as e:
            logger.error(f"Video search failed: {e}")
            return {"error": str(e), "results": []}

    async def search_perplexica(
        self,
        query: str,
        search_mode: str = "web",
        max_results: int = 10,
        include_reasoning: bool = True,
    ) -> dict[str, Any]:
        """General AI-powered search with multiple modes."""
        try:
            # Convert string mode to enum
            mode = SearchMode(search_mode.lower())

            response = await self.api.search(
                query=query,
                search_mode=mode,
                max_results=max_results,
                include_reasoning=include_reasoning,
            )
            return self._format_response(response)
        except Exception as e:
            logger.error(f"Perplexica search failed: {e}")
            return {"error": str(e), "results": []}

    def _format_response(self, response: PerplexicaResponse) -> dict[str, Any]:
        """Format the Perplexica response for tool output."""
        return {
            "query": response.query,
            "search_mode": response.search_mode.value,
            "summary": response.summary,
            "reasoning": response.reasoning,
            "sources": [
                {
                    "title": source.title,
                    "url": source.url,
                    "snippet": source.snippet,
                    "source": source.source,
                    "relevance_score": source.relevance_score,
                    "timestamp": source.timestamp,
                    "metadata": source.metadata,
                }
                for source in response.sources
            ],
            "citations": response.citations,
            "follow_up_questions": response.follow_up_questions,
            "confidence_score": response.confidence_score,
            "total_results": response.total_results,
            "processing_time": response.processing_time,
        }

    async def health_check(self) -> dict[str, Any]:
        """Check the health of the plugin."""
        try:
            health = await self.api.health_check()
            return {
                "status": "healthy",
                "plugin": self.name,
                "api_health": health,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "plugin": self.name,
                "error": str(e),
            }

    async def get_search_history(self) -> list[dict[str, Any]]:
        """Get the search history."""
        try:
            history = self.api.get_search_history()
            return [self._format_response(response) for response in history]
        except Exception as e:
            logger.error(f"Failed to get search history: {e}")
            return []

    async def clear_search_history(self) -> dict[str, str]:
        """Clear the search history."""
        try:
            self.api.clear_history()
            return {"status": "success", "message": "Search history cleared"}
        except Exception as e:
            logger.error(f"Failed to clear search history: {e}")
            return {"status": "error", "error": str(e)}
