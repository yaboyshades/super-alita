
"""GitHub helper utilities for MCP integrations.

The helper class encapsulates HTTP interactions with the GitHub REST API and
also provides graceful fallbacks when credentials or network access are not
available.  It focuses on the primitives that the MCP server exposes as
tools – issue/PR creation, workflow inspection, search, and repository-level
context analysis tailored for the Super Alita codebase.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GitHubResourceDescriptor:
    """Lightweight description of an addressable GitHub resource."""

    uri: str
    name: str
    description: str
    mime_type: str = "application/json"

    def as_dict(self) -> dict[str, str]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


class GitHubMCPTools:
    """GitHub integration helpers used by the MCP server."""

    _DEFAULT_RESOURCES: tuple[GitHubResourceDescriptor, ...] = (
        GitHubResourceDescriptor(
            uri="github://repositories",
            name="GitHub Repositories",
            description="Repositories accessible to the configured token",
        ),
        GitHubResourceDescriptor(
            uri="github://issues",
            name="GitHub Issues",
            description="Open issues and pull requests for the authenticated user",
        ),
        GitHubResourceDescriptor(
            uri="github://actions",
            name="GitHub Actions",
            description="Workflow runs for repositories in scope",
        ),
    )

    def __init__(self, github_token: str | None = None, *, base_url: str = "https://api.github.com") -> None:
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.base_url = base_url.rstrip("/")
        self._default_headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "super-alita-mcp",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.github_token:
            self._default_headers["Authorization"] = f"Bearer {self.github_token}"
        else:
            logger.warning("GitHub token not configured – API calls will be rate limited or disabled.")

    # ------------------------------------------------------------------
    # High level helpers used by the MCP server
    # ------------------------------------------------------------------

    def list_resources(self) -> list[dict[str, str]]:
        """Return metadata for logical GitHub resources."""

        return [descriptor.as_dict() for descriptor in self._DEFAULT_RESOURCES]

    async def read_resource(self, uri: str) -> Any:
        """Resolve a logical resource URI into concrete data."""

        if uri == "github://repositories":
            return await self._get_repositories()
        if uri == "github://issues":
            return await self._get_issues()
        if uri == "github://actions":
            return await self._get_actions()
        raise ValueError(f"Unknown resource URI: {uri}")

    async def create_issue(self, *, repo: str, title: str, body: str = "", labels: list[str] | None = None) -> dict[str, Any]:
        """Create a GitHub issue."""

        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        return await self._request("POST", f"/repos/{repo}/issues", json=payload)

    async def create_pull_request(
        self,
        *,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: str = "",
        draft: bool = False,
    ) -> dict[str, Any]:
        """Create a pull request."""

        payload = {
            "title": title,
            "body": body,
            "head": head,
            "base": base,
            "draft": draft,
        }
        return await self._request("POST", f"/repos/{repo}/pulls", json=payload)

    async def search_code(
        self,
        *,
        query: str,
        repo: str | None = None,
        language: str | None = None,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Perform a GitHub code search."""

        search_query = query
        if repo:
            search_query += f" repo:{repo}"
        if language:
            search_query += f" language:{language}"
        params = {"q": search_query, "per_page": max_results}
        return await self._request("GET", "/search/code", params=params)

    async def get_workflow_runs(
        self,
        *,
        repo: str,
        workflow_id: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Fetch workflow runs for a repository."""

        if workflow_id:
            endpoint = f"/repos/{repo}/actions/workflows/{workflow_id}/runs"
        else:
            endpoint = f"/repos/{repo}/actions/runs"
        params: dict[str, str] = {}
        if status:
            params["status"] = status
        return await self._request("GET", endpoint, params=params)

    async def analyze_super_alita_context(
        self, *, analysis_type: str, focus_area: str | None = None
    ) -> dict[str, Any]:
        """Return repository-aware analysis snapshots."""

        repo_root = Path(".").resolve()
        summary: dict[str, Any] = {
            "analysis_type": analysis_type,
            "repository": "local",
            "focus_area": focus_area,
        }

        if analysis_type == "architecture":
            summary["components"] = self._build_architecture_snapshot()
        elif analysis_type == "dependencies":
            files = ["requirements.txt", "pyproject.toml", "package.json", ".env.example", "Dockerfile"]
            summary["dependencies"] = await self._load_repo_files(repo_root, files)
        elif analysis_type == "patterns":
            summary["patterns"] = self._describe_code_patterns()
        elif analysis_type == "config":
            env_template = await self._load_single_file(repo_root, ".env.example")
            summary["configuration"] = {
                "env_template": env_template,
                "llm_providers": self._detect_llm_adapters(),
                "ports": {"runtime": 8080, "mcp": 8001, "skillset": 8001},
            }
        else:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_repositories(self) -> Any:
        if not self.github_token:
            return {"error": "GITHUB_TOKEN is not configured."}
        params = {"per_page": 50, "sort": "updated"}
        return await self._request("GET", "/user/repos", params=params)

    async def _get_issues(self) -> Any:
        if not self.github_token:
            return {"error": "GITHUB_TOKEN is not configured."}
        params = {"filter": "all", "state": "open", "per_page": 50}
        return await self._request("GET", "/issues", params=params)

    async def _get_actions(self) -> Any:
        if not self.github_token:
            return {"error": "GITHUB_TOKEN is not configured."}
        # GitHub does not expose a global listing for workflow runs, so return a placeholder.
        return {
            "message": "Provide a repository via the get_workflow_runs tool to fetch workflow data."
        }

    async def _load_repo_files(self, root: Path, files: Iterable[str]) -> dict[str, str | None]:
        results: dict[str, str | None] = {}
        for relative in files:
            results[relative] = await self._load_single_file(root, relative)
        return results

    async def _load_single_file(self, root: Path, relative: str) -> str | None:
        candidate = root / relative
        if candidate.exists():
            return await asyncio.to_thread(candidate.read_text, encoding="utf-8")
        if not self.github_token:
            return None
        try:
            data = await self._request("GET", f"/repos/yaboyshades/super-alita/contents/{relative}")
        except Exception as exc:  # pragma: no cover - network errors
            logger.debug("Failed to fetch %s from GitHub: %s", relative, exc)
            return None
        if isinstance(data, dict) and "content" in data:
            try:
                encoded = data["content"]
                return base64.b64decode(encoded).decode("utf-8")
            except Exception:  # pragma: no cover - decoding errors
                return None
        return None

    def _build_architecture_snapshot(self) -> dict[str, Any]:
        return {
            "mcp_server": "Model Context Protocol entrypoints under src/mcp_server/",
            "reug_runtime": "Spec-driven orchestration and streaming agent router",
            "abilities": "Capability adapters located in src/abilities/",
            "orchestration": "Reliability, observability, and scoring pipelines",
            "telemetry": "Telemetry collectors and publishers in src/core/telemetry/",
            "plugins": "Plugin registration and discovery under src/plugins/",
        }

    def _describe_code_patterns(self) -> dict[str, str]:
        return {
            "event_driven": "Redis-backed event bus with plugin observers",
            "plugin_architecture": "PluginInterface contracts with runtime discovery",
            "async_workflows": "FastAPI, asyncio, and streaming orchestration",
            "configuration": "Pydantic-based settings in src/core/settings.py",
            "testing": "pytest suites mirrored under tests/ with contract coverage",
        }

    def _detect_llm_adapters(self) -> list[str]:
        adapters = []
        adapter_root = Path("src/reug_runtime")
        if (adapter_root / "llm_client.py").exists():
            adapters.append("OpenAI-compatible runtime client")
        if Path("src/abilities/mangle").exists():
            adapters.append("Mangle reasoning engine")
        if Path("src/agents").exists():
            adapters.append("Agent toolchain integration")
        return adapters

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        if not self.github_token:
            raise RuntimeError("GitHub token is required for this operation.")

        url = endpoint if endpoint.startswith("http") else f"{self.base_url}{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=self._default_headers, params=params, json=json) as response:
                text = await response.text()
                if response.status >= 400:
                    logger.error("GitHub API error %s: %s", response.status, text)
                try:
                    return await response.json()
                except aiohttp.ContentTypeError:
                    return {"status": response.status, "body": text}


def register_github_tools(app: Any, *, tools: GitHubMCPTools | None = None) -> GitHubMCPTools:
    """Register GitHub-related tools on a FastMCP app instance."""

    if getattr(app, "_github_tools_registered", False):
        existing = getattr(app, "_github_tools_instance", None)
        return existing if isinstance(existing, GitHubMCPTools) else GitHubMCPTools()

    github_tools = tools or GitHubMCPTools()

    def _wrap(coro):
        async def _runner(*args: Any, **kwargs: Any) -> dict[str, Any]:
            try:
                result = await coro(*args, **kwargs)
                return {"ok": True, "data": result}
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.error("GitHub tool invocation failed: %s", exc)
                return {"ok": False, "error": str(exc)}

        return _runner

    @app.tool(
        name="github_list_resources",
        description="List logical GitHub resources available to the MCP server.",
    )
    async def github_list_resources() -> dict[str, Any]:
        return {"resources": github_tools.list_resources()}

    @app.tool(
        name="github_read_resource",
        description="Fetch a logical GitHub resource (e.g. github://repositories).",
    )
    async def github_read_resource(uri: str) -> dict[str, Any]:
        return await _wrap(github_tools.read_resource)(uri=uri)

    @app.tool(
        name="github_create_issue",
        description="Create a GitHub issue in the form owner/repo",
    )
    async def github_create_issue(
        repo: str,
        title: str,
        body: str | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        return await _wrap(github_tools.create_issue)(repo=repo, title=title, body=body or "", labels=labels or [])

    @app.tool(
        name="github_create_pull_request",
        description="Create a GitHub pull request in the form owner/repo",
    )
    async def github_create_pull_request(
        repo: str,
        title: str,
        head: str,
        base: str,
        body: str | None = None,
        draft: bool = False,
    ) -> dict[str, Any]:
        return await _wrap(github_tools.create_pull_request)(
            repo=repo, title=title, head=head, base=base, body=body or "", draft=draft
        )

    @app.tool(
        name="github_search_code",
        description="Search GitHub code across repositories.",
    )
    async def github_search_code(
        query: str,
        repo: str | None = None,
        language: str | None = None,
        max_results: int = 10,
    ) -> dict[str, Any]:
        return await _wrap(github_tools.search_code)(
            query=query, repo=repo, language=language, max_results=max_results
        )

    @app.tool(
        name="github_get_workflow_runs",
        description="Fetch GitHub Actions workflow runs for a repository.",
    )
    async def github_get_workflow_runs(
        repo: str,
        workflow_id: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        return await _wrap(github_tools.get_workflow_runs)(
            repo=repo, workflow_id=workflow_id, status=status
        )

    @app.tool(
        name="analyze_super_alita_context",
        description="Summarise Super Alita repository context for assistants.",
    )
    async def analyze_super_alita_context(
        analysis_type: str,
        focus_area: str | None = None,
    ) -> dict[str, Any]:
        return await _wrap(github_tools.analyze_super_alita_context)(
            analysis_type=analysis_type, focus_area=focus_area
        )

    app._github_tools_registered = True  # type: ignore[attr-defined]
    app._github_tools_instance = github_tools  # type: ignore[attr-defined]
    return github_tools
