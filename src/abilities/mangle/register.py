#!/usr/bin/env python3
"""
Mangle Integration Registration for Super Alita

This module registers the Mangle deductive database programming integration
with Super Alita's ability registry and provides tool contracts.
"""

import logging
from typing import Any

from src.abilities.mangle.mangle_ability import (
    MangleAbility,
    ManglePluginInterface,
)
from src.core.plugin_registry import register_plugin
from src.reug_runtime.mcp_abstractor import abstract_mcp_box

logger = logging.getLogger(__name__)


def register_mangle_abilities(ability_registry, config=None):
    """Register Mangle abilities with the Super Alita ability registry.

    Args:
        ability_registry: The Super Alita ability registry
        config: Optional configuration for Mangle

    Returns:
        True if registration successful, False otherwise
    """
    try:
        mangle_config = (config or {}).get("mangle", {})
        mangle_ability = MangleAbility(mangle_config)

        # mangle_query
        async def _exec_query(args):  # type: ignore
            return await mangle_ability.query(
                args.get("query", ""), args.get("params", {})
            )

        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_query",
                "description": (
                    "Execute a Mangle deductive query against the knowledge "
                    "base"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The Mangle query to execute",
                        },
                        "params": {
                            "type": "object",
                            "description": "Optional parameters",
                        },
                    },
                    "required": ["query"],
                },
            },
            executor=_exec_query,
        )

        # mangle_add_fact
        async def _exec_add_fact(args):  # type: ignore
            return await mangle_ability.add_fact(args.get("fact", ""))

        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_add_fact",
                "description": "Add a fact to the knowledge base",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": "Fact to add",
                        }
                    },
                    "required": ["fact"],
                },
            },
            executor=_exec_add_fact,
        )

        # mangle_add_rule
        async def _exec_add_rule(args):  # type: ignore
            return await mangle_ability.add_rule(
                args.get("name", ""), args.get("rule", "")
            )

        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_add_rule",
                "description": "Add a rule to the knowledge base",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Rule name",
                        },
                        "rule": {
                            "type": "string",
                            "description": "Rule clause",
                        },
                    },
                    "required": ["name", "rule"],
                },
            },
            executor=_exec_add_rule,
        )

        # mangle_analyze_dependencies
        async def _exec_analyze_deps(args):  # type: ignore
            return await mangle_ability.analyze_dependencies(
                args.get("dependencies", [])
            )

        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_analyze_dependencies",
                "description": "Analyze dependencies for issues",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dependencies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "version": {"type": "string"},
                                },
                                "required": ["name", "version"],
                            },
                            "description": "Dependencies list",
                        }
                    },
                    "required": ["dependencies"],
                },
            },
            executor=_exec_analyze_deps,
        )

        # mangle_knowledge_graph
        async def _exec_kg(args):  # type: ignore
            return await mangle_ability.knowledge_graph_query(
                args.get("query", ""), args.get("context", [])
            )

        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_knowledge_graph",
                "description": "Perform a knowledge graph query",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "context": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "relation": {"type": "string"},
                                    "subject": {"type": "string"},
                                    "object": {"type": "string"},
                                },
                                "required": [
                                    "relation",
                                    "subject",
                                    "object",
                                ],
                            },
                        },
                    },
                    "required": ["query"],
                },
            },
            executor=_exec_kg,
        )

        # mangle_explain
        async def _exec_explain(args):  # type: ignore
            return await mangle_ability.explain_query_results(
                args.get("query", "")
            )

        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_explain",
                "description": "Execute a query and explain results",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            executor=_exec_explain,
        )

        # mangle_rule_catalog
        async def _exec_rule_catalog(_args):  # type: ignore
            if hasattr(mangle_ability, "rule_catalog"):
                return await mangle_ability.rule_catalog()
            return {
                "success": True,
                "rules": [
                    {
                        "id": name,
                        "name": name,
                        "description": body[:60],
                    }
                    for name, body in getattr(
                        mangle_ability, "rules", {}
                    ).items()
                ],
                "count": len(getattr(mangle_ability, "rules", {})),
                "fallback": True,
            }

        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_rule_catalog",
                "description": "List available Mangle rules",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
            executor=_exec_rule_catalog,
        )

        # mangle_run_rule
        async def _exec_run_rule(args):  # type: ignore
            return await mangle_ability.run_rule(
                args.get("rule", ""), args.get("query", "")
            )

        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_run_rule",
                "description": "Run a one-off rule with a query",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "rule": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["rule", "query"],
                },
            },
            executor=_exec_run_rule,
        )

        # mangle_grpc_health
        async def _exec_grpc_health(_args):  # type: ignore
            return await mangle_ability.grpc_health()

        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_grpc_health",
                "description": "Check Mangle gRPC connectivity health",
                "input_schema": {"type": "object", "properties": {}},
            },
            executor=_exec_grpc_health,
        )

        logger.info("Mangle abilities registered successfully")
        return True
    except Exception as e:  # pragma: no cover
        logger.error(f"Failed to register Mangle abilities: {e}")
        return False


def register_mangle_plugin(
    _plugin_registry: Any | None = None,
    _config: dict[str, Any] | None = None,
):
    """Register the Mangle plugin with the Super Alita plugin registry.

    The current application startup passes two positional arguments
    (plugin_registry, config). We don't actually need either at the moment
    because plugin registration uses the global registry, but we accept them
    for forward compatibility and to avoid TypeErrors.

    Args:
        _plugin_registry: Optional plugin registry (unused currently)
        _config: Optional configuration (unused currently)

    Returns:
        True if registration successful, False otherwise.
    """
    try:
        plugin = ManglePluginInterface()
        register_plugin("mangle_plugin", plugin)
        logger.info("Mangle plugin registered successfully")
        return True
    except Exception as e:  # pragma: no cover
        logger.error(f"Failed to register Mangle plugin: {e}")
        return False


def export_mangle_rules_to_mcp_box(dir_: str = ".mcp_box") -> dict[str, int]:
    """Mine known rules and export generic Mangle rule tools to MCP-Box.

    Writes mangle_rule_catalog and mangle_run_rule if not already present,
    then regenerates the catalog via abstractor.
    """
    import json
    from pathlib import Path

    box = Path(dir_)
    box.mkdir(parents=True, exist_ok=True)

    base_specs = [
        {
            "tool_id": "mangle_rule_catalog",
            "description": "List available Mangle rules",
            "action": "mangle_rule_catalog",
            "input_schema": {"type": "object", "properties": {}},
            "output_schema": {
                "type": "object",
                "properties": {
                    "rules": {"type": "array"},
                    "count": {"type": "integer"},
                },
            },
        },
        {
            "tool_id": "mangle_run_rule",
            "description": (
                "Execute a one-off Mangle rule with a query over current "
                "facts"
            ),
            "action": "mangle_run_rule",
            "input_schema": {
                "type": "object",
                "required": ["rule", "query"],
                "properties": {
                    "rule": {"type": "string"},
                    "query": {"type": "string"},
                },
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "count": {"type": "integer"},
                },
            },
        },
    ]

    written = 0
    for spec in base_specs:
        path = box / f"{spec['tool_id']}.json"
        if not path.exists():
            path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            written += 1
    index = abstract_mcp_box(box)
    return {"written": written, "catalog": len(index.get("tools", []))}


def export_mangle_to_mcp_box(dir_: str = ".mcp_box") -> dict[str, int]:
    """Export core Mangle abilities as MCP specs into the MCP-Box.

    This aligns with the training-free AgentDistill pathway by packaging
    reusable, parameterized tools that student agents can call directly.

    Args:
        mcp_box_dir: Directory for MCP specs (default .mcp_box)

    Returns:
        Summary counts of specs written and catalog size post-abstract.
    """
    import json
    from pathlib import Path

    box = Path(dir_)
    box.mkdir(parents=True, exist_ok=True)

    specs: list[dict[str, Any]] = [
        {
            "tool_id": "mangle_query",
            "description": (
                "Execute a Mangle deductive query against the knowledge "
                "base"
            ),
            "action": "mangle_query",
            "input_schema": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "context": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "relation": {"type": "string"},
                                "subject": {"type": "string"},
                                "object": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "count": {"type": "integer"},
                },
            },
        },
        {
            "tool_id": "mangle_explain",
            "description": "Execute a Mangle query and return an explanation",
            "action": "mangle_query_explain",
            "input_schema": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "explanation": {"type": "string"},
                    "count": {"type": "integer"},
                },
            },
        },
        {
            "tool_id": "mangle_analyze_dependencies",
            "description": (
                "Analyze project dependencies for known " "vulnerabilities"
            ),
            "action": "mangle_analyze_dependencies",
            "input_schema": {
                "type": "object",
                "required": ["dependencies"],
                "properties": {
                    "dependencies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "version": {"type": "string"},
                            },
                        },
                    }
                },
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "count": {"type": "integer"},
                },
            },
        },
        {
            "tool_id": "mangle_rule_catalog",
            "description": "List available Mangle rules",
            "action": "mangle_rule_catalog",
            "input_schema": {"type": "object", "properties": {}},
            "output_schema": {
                "type": "object",
                "properties": {
                    "rules": {"type": "array"},
                    "count": {"type": "integer"},
                },
            },
        },
        {
            "tool_id": "mangle_run_rule",
            "description": (
                "Execute a one-off Mangle rule with a query over current "
                "facts"
            ),
            "action": "mangle_run_rule",
            "input_schema": {
                "type": "object",
                "required": ["rule", "query"],
                "properties": {
                    "rule": {"type": "string"},
                    "query": {"type": "string"},
                },
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "count": {"type": "integer"},
                },
            },
        },
    ]

    written = 0
    for spec in specs:
        path = box / f"{spec['tool_id']}.json"
        path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        written += 1

    # Rebuild index.json and catalog.json
    index = abstract_mcp_box(box)
    return {"written": written, "catalog": len(index.get("tools", []))}
