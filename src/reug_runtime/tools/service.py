"""Centralized tool catalog and dynamic registration service.

This module provides a unified service for managing tool catalogs,
dynamic tool registration, and MCP persistence. It consolidates
the logic previously scattered across router.py and router_tools.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ToolCatalogService:
    """Service for managing tool catalogs and dynamic registration."""

    def __init__(self, mcp_box_dir: str | None = None) -> None:
        """Initialize the tool catalog service.
        
        Args:
            mcp_box_dir: Directory for MCP spec persistence. Defaults to
                        environment variable MCP_BOX_DIR or '.mcp_box'
        """
        self._mcp_box_dir = Path(mcp_box_dir or os.getenv("MCP_BOX_DIR", ".mcp_box"))
        self._logger = self._get_logger()

    def _get_logger(self) -> Any:
        """Get a logger instance for structured logging."""
        try:
            import logging
            return logging.getLogger(__name__)
        except Exception:
            # Fallback to basic print-based logger
            class PrintLogger:
                def info(self, msg: str) -> None:
                    print(f"INFO: {msg}")
                def warning(self, msg: str) -> None:
                    print(f"WARNING: {msg}")
                def error(self, msg: str) -> None:
                    print(f"ERROR: {msg}")
            return PrintLogger()

    def _ensure_mcp_box(self) -> Path:
        """Ensure MCP box directory exists and return its path."""
        self._mcp_box_dir.mkdir(parents=True, exist_ok=True)
        return self._mcp_box_dir

    def list_tools(self, registry: Any = None) -> list[dict[str, Any]]:
        """Get the complete tool catalog including static and dynamic tools.
        
        Args:
            registry: Optional ability registry to fetch dynamic tools from
            
        Returns:
            List of tool catalog entries
        """
        # Static tool catalog (defined in router_tools.py)
        from . import TOOL_CATALOG
        catalog = list(TOOL_CATALOG)
        self._logger.info(f"Loaded {len(catalog)} static tools")

        # Add dynamic tools from the ability registry
        if registry is not None:
            try:
                if hasattr(registry, "get_available_tools_schema"):
                    dynamic_tools = registry.get_available_tools_schema()
                    self._logger.info(f"Found {len(dynamic_tools)} dynamic tools")

                    # Convert dynamic tool contracts to catalog format
                    for tool_contract in dynamic_tools:
                        tool_name = tool_contract.get("tool_id") or tool_contract.get("name")
                        if tool_name and not any(t["name"] == tool_name for t in catalog):
                            catalog_entry = {
                                "name": tool_name,
                                "description": tool_contract.get("description", ""),
                                "input_schema": tool_contract.get("input_schema", {}),
                                "output_schema": tool_contract.get("output_schema", {}),
                            }
                            catalog.append(catalog_entry)
                            self._logger.info(f"Added dynamic tool: {tool_name}")
                else:
                    self._logger.warning("Registry has no get_available_tools_schema method")
            except Exception as e:
                self._logger.error(f"Failed to load dynamic tools: {e}")

        # Try to load tools from persisted catalog.json
        try:
            catalog_path = self._mcp_box_dir / "catalog.json"
            if catalog_path.exists():
                mcp_catalog_tools = json.loads(catalog_path.read_text(encoding="utf-8"))
                self._logger.info(f"Found {len(mcp_catalog_tools)} tools in catalog.json")

                # Add tools from catalog.json (if not already in the catalog)
                for tool in mcp_catalog_tools:
                    tool_name = tool.get("name")
                    if tool_name and not any(t["name"] == tool_name for t in catalog):
                        catalog.append(tool)
                        self._logger.info(f"Added catalog tool: {tool_name}")
        except Exception as e:
            self._logger.warning(f"Failed to load MCP catalog: {e}")

        self._logger.info(f"Final catalog has {len(catalog)} tools")
        return catalog

    def register_dynamic_tool(self, spec: dict[str, Any]) -> str:
        """Register a dynamic tool and persist its specification.
        
        Args:
            spec: Tool specification dictionary
            
        Returns:
            The tool ID of the registered tool
        """
        tool_id = self._persist_spec(spec)
        self._logger.info(f"Registered dynamic tool: {tool_id}")
        return tool_id

    def _persist_spec(self, spec: dict[str, Any]) -> str:
        """Persist a tool specification to the MCP box.
        
        Args:
            spec: Tool specification dictionary
            
        Returns:
            The tool ID used for persistence
        """
        box = self._ensure_mcp_box()
        tool_id = (
            spec.get("tool_id")
            or spec.get("name")
            or f"mcp_{len(list(box.glob('*.json')))}"
        )
        spec = {"tool_id": tool_id, **spec}
        path = box / f"{tool_id}.json"
        
        try:
            path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            self._logger.info(f"Persisted tool spec: {tool_id} -> {path}")
        except Exception as e:
            self._logger.error(f"Failed to persist tool spec {tool_id}: {e}")
            
        return tool_id

    def ensure_tool_registered(self, tool_name: str, tool_args: dict[str, Any], registry: Any) -> bool:
        """Ensure a tool is registered, creating it dynamically if needed.
        
        This is the unified version of the _ensure_tool logic from the orchestrator.
        
        Args:
            tool_name: Name of the tool to ensure
            tool_args: Arguments that would be passed to the tool
            registry: Ability registry to register the tool with
            
        Returns:
            True if tool is available, False otherwise
        """
        try:
            # Check if tool is already known
            if getattr(registry, "knows", lambda *_: True)(tool_name):
                return True

            # Auto-register based on heuristics
            return self._auto_register_tool(tool_name, tool_args, registry)
        except Exception as e:
            self._logger.error(f"Failed to ensure tool {tool_name}: {e}")
            return False

    def _auto_register_tool(self, tool_name: str, tool_args: dict[str, Any], registry: Any) -> bool:
        """Auto-register a tool based on heuristics."""
        import asyncio
        import urllib.request
        import uuid

        # GitHub proxy
        if (
            any(k in tool_args for k in ("owner", "repo", "path"))
            or "github" in tool_name.lower()
        ):
            contract = {
                "tool_id": tool_name,
                "description": "Proxy to fetch a raw file from GitHub",
                "input_schema": {
                    "type": "object",
                    "required": ["owner", "repo", "path"],
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "path": {"type": "string"},
                        "ref": {"type": "string"},
                    },
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "url": {"type": "string"},
                        "truncated": {"type": "boolean"},
                        "error": {"type": "string"},
                    },
                },
            }

            async def _exec(a: dict[str, Any]) -> dict[str, Any]:
                from typing import cast
                result = await registry.execute(
                    "fetch_github_raw",
                    {
                        "owner": a.get("owner"),
                        "repo": a.get("repo"),
                        "path": a.get("path"),
                    }
                    | ({"ref": a.get("ref")} if a.get("ref") else {}),
                )
                return cast(dict[str, Any], result)

            registry.register_tool(contract=contract, executor=_exec)
            self.register_dynamic_tool({
                "tool_id": tool_name,
                "description": contract["description"],
                "action": "fetch_github_raw",
                "input_schema": contract["input_schema"],
                "output_schema": contract["output_schema"],
            })
            return True

        # URL fetcher
        if ("url" in {k.lower() for k in tool_args}) or (
            any(x in tool_name.lower() for x in ("url", "http", "fetch"))
        ):
            contract = {
                "tool_id": tool_name,
                "description": "Fetch a URL and return UTF-8 text (best-effort)",
                "input_schema": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {"type": "string"},
                        "truncate": {"type": "integer"},
                    },
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "truncated": {"type": "boolean"},
                        "error": {"type": "string"},
                    },
                },
            }

            async def _exec(a: dict[str, Any]) -> dict[str, Any]:
                url = a.get("url")
                if not isinstance(url, str) or not url:
                    return {"error": "missing url"}
                truncate = int(a.get("truncate") or 4000)

                def _do_fetch() -> dict[str, Any]:
                    try:
                        with urllib.request.urlopen(url, timeout=8) as resp:  # nosec B310
                            raw = resp.read()
                        text = raw.decode("utf-8", errors="replace")
                        truncated = False
                        if len(text) > truncate:
                            text = text[:truncate]
                            truncated = True
                        return {"content": text, "truncated": truncated}
                    except Exception as e:
                        return {"content": "", "truncated": False, "error": str(e)}

                return await asyncio.to_thread(_do_fetch)

            registry.register_tool(contract=contract, executor=_exec)
            self.register_dynamic_tool({
                "tool_id": tool_name,
                "description": contract["description"],
                "action": "fetch_url_text",
                "input_schema": contract["input_schema"],
                "output_schema": contract["output_schema"],
            })
            return True

        # Fallback planning tool
        contract = {
            "tool_id": tool_name,
            "description": "Echo a minimal plan for the task",
            "input_schema": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "steps": {"type": "array", "items": {"type": "string"}}
                },
            },
        }

        async def _exec(a: dict[str, Any]) -> dict[str, Any]:
            t = (a.get("task") or "unknown task").strip()
            return {
                "steps": [
                    f"Understand: {t}",
                    "Identify resources",
                    "Execute and verify",
                ]
            }

        registry.register_tool(contract=contract, executor=_exec)
        self.register_dynamic_tool({
            "tool_id": tool_name,
            "description": contract["description"],
            "action": "echo_plan",
            "input_schema": contract["input_schema"],
            "output_schema": contract["output_schema"],
        })
        return True
