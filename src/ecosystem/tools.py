"""
Concrete tools for the ecosystem orchestrator that integrate with external
services. These conform to the local protocols to keep the orchestrator
implementation clean and decoupled.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .master_orchestrator import GitHubExample, ICopilotContextEnhancer


class GitHubTool(ICopilotContextEnhancer):
    """
    Implements `ICopilotContextEnhancer` by querying the GitHub Code Search API
    for relevant examples. Uses httpx for async requests.
    """

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.getenv("GITHUB_API_TOKEN")
        if not self.token:
            # Proceed unauthenticated but with limited rate
            pass
        self.headers: dict[str, str] = {
            "Accept": "application/vnd.github.v3.text-match+json",
            "User-Agent": "Super-Alita-Ecosystem/1.0",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

    async def find_github_examples(self, query: str) -> list[GitHubExample]:
        search_query = f"{query} language:python"
        endpoint = f"{self.BASE_URL}/search/code"
        params = {"q": search_query, "per_page": 3}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    endpoint, headers=self.headers, params=params
                )
                response.raise_for_status()
                data = response.json()
                return self._map_response_to_examples(data)
        except Exception:
            return []

    def _map_response_to_examples(
        self, response_data: dict[str, Any]
    ) -> list[GitHubExample]:
        results: list[GitHubExample] = []
        for item in response_data.get("items", []):
            code_snippet = "# snippet unavailable"
            text_matches = item.get("text_matches")
            if text_matches and isinstance(text_matches, list):
                frag = text_matches[0].get("fragment")
                if isinstance(frag, str) and frag.strip():
                    code_snippet = frag

            repo_full = item.get("repository", {}).get(
                "full_name", "unknown/unknown"
            )
            path = item.get("path", "")
            license_info = item.get("repository", {}).get("license")
            license_name = (
                license_info.get("name")
                if isinstance(license_info, dict)
                else None
            )

            results.append(
                GitHubExample(
                    repo=repo_full,
                    path=path,
                    code_snippet=code_snippet,
                    license=license_name or "N/A",
                )
            )
        return results
