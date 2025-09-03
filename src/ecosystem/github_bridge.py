# src/ecosystem/github_bridge.py
"""
Provides a concrete implementation of the Copilot Context Enhancer
by bridging to the GitHub Code Search API.
"""
import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib import parse, request

from .master_orchestrator import GitHubExample, ICopilotContextEnhancer

# A type hint for an async HTTP transport function.
HttpTransport = Callable[
    [str, dict[str, str], dict[str, str]], Awaitable[dict[str, Any]]
]


class GitHubCodeSearchBridge:
    """A bridge to the GitHub Code Search API."""

    BASE_URL = "https://api.github.com/search/code"

    def __init__(self, token: str, transport: HttpTransport | None = None):
        if not token:
            raise ValueError("A GitHub token is required.")
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3.text-match+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self.transport = transport or self._default_http_transport

    async def search(self, query: str, per_page: int = 3) -> dict[str, Any]:
        """Performs a code search on GitHub."""
        params = {"q": query, "per_page": str(per_page), "sort": "indexed"}
        return await self.transport(self.BASE_URL, self.headers, params)

    async def _default_http_transport(
        self, url: str, headers: dict[str, str], params: dict[str, str]
    ) -> dict[str, Any]:
        """Default synchronous HTTP transport using urllib."""
        # This is a simple, synchronous implementation.
        # In a real-world scenario, an async library like aiohttp would be preferred.
        req_url = f"{url}?{parse.urlencode(params)}"
        req = request.Request(req_url, headers=headers)
        with request.urlopen(req) as response:
            if response.status == 200:
                return json.loads(response.read().decode("utf-8"))
            else:
                raise RuntimeError(f"GitHub API error: {response.status}")


class CopilotContextEnhancerFromGitHub(ICopilotContextEnhancer):
    """
    Implements the ICopilotContextEnhancer protocol using the GitHub bridge.
    """

    def __init__(
        self,
        bridge: GitHubCodeSearchBridge,
        default_language: str = "python",
        default_org: str | None = None,
    ):
        self.bridge = bridge
        self.default_language = default_language
        self.default_org = default_org

    async def find_github_examples(self, query: str) -> list[GitHubExample]:
        """Finds code examples by querying GitHub."""
        search_query = f"{query} language:{self.default_language}"
        if self.default_org:
            search_query += f" org:{self.default_org}"

        try:
            api_result = await self.bridge.search(search_query)
            return self._normalize_results(api_result)
        except Exception as e:
            print(f"[GitHubBridge] Error searching GitHub: {e}")
            return []

    def _normalize_results(self, api_result: dict[str, Any]) -> list[GitHubExample]:
        """Converts raw GitHub API results into structured GitHubExample objects."""
        examples = []
        for item in api_result.get("items", []):
            # Extract the code snippet from text_matches if available
            code_snippet = ""
            if item.get("text_matches"):
                code_snippet = item["text_matches"][0].get("fragment", "")

            examples.append(
                GitHubExample(
                    repo=item["repository"]["full_name"],
                    path=item["path"],
                    code_snippet=code_snippet,
                    license=item["repository"].get("license", {}).get("spdx_id"),
                )
            )
        return examples
