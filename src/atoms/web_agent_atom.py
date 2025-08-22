"""
WebAgentAtom  –  the only external skill the system starts with.
Alita-style: search → retrieve → optionally wrap as new atom.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import aiohttp
from dotenv import load_dotenv

from src.core.plugin_interface import PluginInterface  # Use existing base for now

load_dotenv()


class WebAgentAtom(PluginInterface):
    """
    Unified Web & Code Search Atom

    Provides bulletproof tool result emission via emit_tool_result helper.
    """

    async def emit_tool_result(
        self,
        session_id: str,
        success: bool,
        result: dict[str, Any],
        error: str | None = None,
        message: str | None = None,
    ) -> None:
        """
        Emit a tool result event with all required fields validated.

        Args:
            session_id: Session/conversation ID for routing
            success: Whether the tool execution succeeded
            result: Tool execution result data
            error: Optional error message (if success=False)
            message: Optional human-readable message
        """
        from src.core.events import ToolResultEvent

        tool_result_event = ToolResultEvent(
            source_plugin=self.name,  # Use plugin name as source
            conversation_id=session_id,
            tool_name="web_agent",
            success=success,
            result=result,
            error=error,
            message=message,
        )

        # Use event_bus.publish directly with the validated event object
        await self.event_bus.publish(tool_result_event)

    # Atom metadata
    key = "web_agent"
    name = "Unified Web & Code Search"
    description = (
        "Searches the web (via SearXNG) and GitHub for a query. "
        "Can auto-wrap any returned code snippet as a new atom."
    )

    signature = {
        "query": {"type": "string", "description": "Search string."},
        "web_k": {
            "type": "integer",
            "default": 5,
            "description": "Web hits to return.",
        },
        "github_k": {
            "type": "integer",
            "default": 5,
            "description": "GitHub repos/code to return.",
        },
        "auto_wrap": {
            "type": "boolean",
            "default": False,
            "description": "If True, wrap the top GitHub code hit as a new atom.",
        },
    }

    def __init__(self):
        super().__init__()
        self.searxng_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:4000")
        self.gh_token = os.getenv("GITHUB_TOKEN")
        self._handled: set[str] = set()  # Deduplication cache for tool calls

    @property
    def name(self) -> str:
        return "web_agent"

    async def setup(
        self, event_bus_or_workspace, store, config: dict[str, Any]
    ) -> None:
        """Initialize the web agent atom."""
        # Handle both legacy (event_bus) and unified (workspace) architectures
        if hasattr(event_bus_or_workspace, "subscribe") and hasattr(
            event_bus_or_workspace, "update"
        ):
            # This is a workspace (unified architecture)
            self.workspace = event_bus_or_workspace
            await super().setup(None, store, config)  # No event_bus in unified arch
        else:
            # This is an event_bus (legacy architecture)
            await super().setup(event_bus_or_workspace, store, config)

    async def start(self) -> None:
        """Start the web agent and subscribe to search events."""
        await super().start()

        # Subscribe to web search requests from planner
        await self.subscribe("web_search", self._handle_web_search)
        # NEW: Subscribe to tool events from planner
        await self.subscribe("web_agent", self._on_tool_call)

        logger.info(
            "WebAgentAtom started - ready for web and GitHub search via planner"
        )

    async def shutdown(self) -> None:
        """Shutdown the web agent atom."""
        logger.info("WebAgentAtom shutdown complete")

    async def call(
        self, query: str, web_k: int = 5, github_k: int = 5, auto_wrap: bool = False
    ) -> dict[str, Any]:
        """Main search method - can be called directly or via events."""
        logger.info(
            f"🔍 WebAgent.call initiated: query='{query}', web_k={web_k}, github_k={github_k}"
        )

        try:
            async with aiohttp.ClientSession() as sess:
                logger.info("🚀 Starting parallel searches...")
                web_res, gh_res = await asyncio.gather(
                    self._searxng_search(query, web_k),
                    self._github(sess, query, github_k),
                )
                logger.info(
                    f"📊 Search results: web={len(web_res)}, github={len(gh_res)}"
                )

        except aiohttp.ClientConnectorError as e:
            # Fail fast on network connectivity issues - no retry needed
            logger.warning(f"Network connectivity failed: {e}")
            return {
                "error": "SearXNG offline",
                "message": "SearXNG service is not reachable. Please start it with: docker compose up -d",
                "query": query,
                "web": [],
                "github": [],
                "summary": "Service offline - 0 results",
            }
        except Exception as e:
            # Other errors might be transient, allow retry
            logger.error(f"WebAgent call failed: {e}")
            raise

        if auto_wrap and gh_res and gh_res[0].get("source") == "github":
            wrapped = await self._wrap_atom(gh_res[0])
            gh_res[0]["wrapped_atom_id"] = wrapped

        result = {
            "query": query,
            "web": web_res,
            "github": gh_res,
            "summary": f"{len(web_res)} web + {len(gh_res)} GitHub hits",
            "total_results": len(web_res) + len(gh_res),
        }

        # Automatic persistence layer for neural recall
        try:
            if hasattr(self, "store") and self.store and hasattr(self.store, "upsert"):
                await self.store.upsert(
                    content={
                        "type": "search_memory",
                        "query": query,
                        "web": web_res,
                        "github": gh_res,
                        "timestamp": time.time(),
                        "total_results": len(web_res) + len(gh_res),
                    },
                    hierarchy_path=["memory", "search"],
                    owner_plugin="web_agent",
                )
                logger.info("🧠 Persisted search results for neural recall")
        except Exception as e:
            logger.warning(f"Failed to persist search results: {e}")

        logger.info(f"✅ WebAgent.call completed: {result['summary']}")
        return result

    async def _handle_web_search(self, event) -> None:
        """Handle web search events from other plugins."""
        try:
            # Handle WorkspaceEvent structure correctly
            if hasattr(event, "data") and isinstance(event.data, dict):
                data = event.data
            else:
                data = event.model_dump() if hasattr(event, "model_dump") else event

            # Defensive parameter extraction: handle both "query" and "input" (legacy)
            query = data.get("query") or data.get("input") or ""
            web_k = data.get("web_k", 5)
            github_k = data.get("github_k", 5)
            auto_wrap = data.get("auto_wrap", False)

            if not query:
                logger.warning("Web search event received with empty query")
                return

            logger.info(f"🌐 Performing web search: {query}")

            result = await self.call(query, web_k, github_k, auto_wrap)

            # Emit search result event with plan coordination
            await self.emit_event(
                "web_search_result",
                query=query,
                result=result,
                session_id=data.get("session_id", "default"),
                plan_id=data.get("plan_id"),  # Forward plan coordination
                step_index=data.get("step_index"),
            )

            # Also emit tool_result for plan continuation using proper ToolResultEvent
            from src.core.events import ToolResultEvent

            tool_result_event = ToolResultEvent(
                source_plugin="web_agent",  # Required field from BaseEvent
                conversation_id=data.get("session_id", "default"),
                tool_name="web_agent",
                result=result,
                success=(
                    not any("error" in item for item in result.get("web", []))
                    if result
                    else True
                ),
                error=None,
                message=(
                    f"Web search completed: {result.get('summary', 'No summary')}"
                    if result
                    else "No result"
                ),
            )

            await self.event_bus.publish(
                tool_result_event
            )  # Pass the model instance directly

            logger.info(f"✅ Web search completed: {result['summary']}")

        except Exception as e:
            logger.error(f"Error handling web search: {e}")

    async def _on_tool_call(self, event):
        """Handle tool calls from planner."""
        # Robust call_id extraction to prevent KeyError
        call_id = getattr(getattr(event, "tool_call", None), "id", None) or str(event)
        if call_id in self._handled:  # Check for duplicate
            logger.debug("Duplicate tool_call %s ignored", call_id)  # Log duplicate
            return  # Exit if duplicate
        self._handled.add(call_id)  # Add to handled set

        try:
            # Handle WorkspaceEvent structure correctly
            if hasattr(event, "data") and isinstance(event.data, dict):
                data = event.data
            else:
                data = event.model_dump() if hasattr(event, "model_dump") else event

            # Defensive parameter extraction: handle both "query" and "input" (legacy)
            query = data.get("query") or data.get("input") or ""
            web_k = data.get("web_k", 5)
            github_k = data.get("github_k", 5)
            # Fix: Use the same conversation_id that was passed in by the planner
            session_id = (
                event.conversation_id
                if hasattr(event, "conversation_id")
                else data.get("conversation_id", "default")
            )

            if not query:
                logger.warning("Tool call received with empty query")
                return

            logger.info(
                f"🔧 Tool call: web_agent query={query} session_id={session_id}"
            )

            result = await self.call(query, web_k, github_k)

            # Use bulletproof emission
            await self.emit_tool_result(
                session_id=session_id,
                success=(
                    not any("error" in item for item in result.get("web", []))
                    if result
                    else True
                ),
                result=result,
                message=(
                    f"Web search completed: {result.get('summary', 'No summary')}"
                    if result
                    else "No result"
                ),
            )

            logger.info(f"🔧 Tool result emitted for session {session_id}")

        except Exception as e:
            logger.error(f"Error handling tool call: {e}")

            # Use bulletproof emission for errors
            await self.emit_tool_result(
                session_id=session_id,
                success=False,
                result={"error": str(e)},
                error=str(e),
                message=f"Tool call error: {e!s}",
            )

    # ---------- helpers ----------

    async def _searxng_search(self, q: str, k: int) -> list[dict[str, Any]]:
        """
        Search web via SearXNG (more reliable than Perplexica).
        Returns [{"title":..., "url":..., "snippet":...}]
        """
        logger.info(f"🔍 Starting SearXNG search for: '{q}' (limit: {k})")
        try:
            async with aiohttp.ClientSession() as sess:
                params = {"q": q, "format": "json", "engines": "google", "pageno": "1"}
                search_url = f"{self.searxng_url}/search"
                logger.info(f"🌐 Calling SearXNG: {search_url} with params: {params}")

                async with sess.get(
                    search_url, params=params, timeout=aiohttp.ClientTimeout(total=8)
                ) as r:
                    logger.info(f"📡 SearXNG response status: {r.status}")
                    data = await r.json()
                    results = data.get("results", [])[:k]
                    logger.info(
                        f"📊 SearXNG returned {len(results)} results (raw: {len(data.get('results', []))})"
                    )

                    formatted_results = [
                        {
                            "title": hit.get("title", ""),
                            "url": hit.get("url", ""),
                            "snippet": hit.get("content", ""),
                            "source": "web",
                            "engine": hit.get("engine", "searxng"),
                        }
                        for hit in results
                    ]

                    logger.info(
                        f"✅ SearXNG search completed: {len(formatted_results)} formatted results"
                    )
                    return formatted_results

        except Exception as e:
            logger.error(f"❌ SearXNG search failed: {e}")
            logger.warning(f"SearXNG search failed ({e}), falling back to GitHub-only")
            return []

    async def _github(
        self, sess: aiohttp.ClientSession, q: str, k: int
    ) -> list[dict[str, Any]]:
        """Search GitHub repositories."""
        logger.info(f"🐙 Starting GitHub search for: '{q}' (limit: {k})")

        if not self.gh_token or self.gh_token == "your_github_token_here":
            logger.info("ℹ️ GitHub search skipped: No valid GITHUB_TOKEN configured")
            return []

        headers = {
            "Authorization": f"Bearer {self.gh_token}",
            "Accept": "application/vnd.github+json",
        }
        params = {"q": q, "sort": "stars", "per_page": k}

        try:
            logger.info(f"🌐 Calling GitHub API with params: {params}")
            async with sess.get(
                "https://api.github.com/search/repositories",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                logger.info(f"📡 GitHub API response status: {r.status}")
                if r.status == 401:
                    logger.warning(
                        "❌ GitHub API returned 401 - invalid or missing token"
                    )
                    return []

                items = (await r.json()).get("items", [])
                logger.info(f"📊 GitHub returned {len(items)} repositories")

            formatted_results = [
                {
                    "source": "github",
                    "title": i["full_name"],
                    "url": i["html_url"],
                    "snippet": i["description"] or "No description available",
                    "full_name": i["full_name"],
                    "description": i["description"],
                    "language": i["language"],
                    "stars": i["stargazers_count"],
                }
                for i in items
            ]

            logger.info(
                f"✅ GitHub search completed: {len(formatted_results)} formatted results"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"❌ GitHub search failed: {e}")
            logger.warning(f"GitHub search failed: {e}")
            return []

    async def _wrap_atom(self, hit: dict[str, Any]) -> str:
        """
        Fetch README or top-level snippet, wrap as new atom.
        Returns the new atom_id.
        """
        try:
            new_atom = {
                "key": f"github_{hit['full_name'].replace('/', '_')}",
                "name": f"GitHub:{hit['full_name']}",
                "description": hit["description"] or "Auto-wrapped GitHub repo",
                "source_url": hit["url"],
                "language": hit["language"],
                "stars": hit["stars"],
                "type": "github_repo_atom",
            }

            atom_id = f"atom_{new_atom['key']}"

            # Store in neural store if available
            if hasattr(self.store, "upsert"):
                await self.store.upsert(
                    memory_id=atom_id,
                    content=new_atom,
                    hierarchy_path=["tools", "github", "auto_wrapped"],
                )
                logger.info(f"🧬 Auto-wrapped GitHub repo as atom: {atom_id}")
            else:
                logger.warning("Store not available for atom wrapping")

            return atom_id

        except Exception as e:
            logger.error("Error wrapping atom: %s", e)
            return f"error_{hit.get('full_name', 'unknown')}"


logger = logging.getLogger(__name__)
