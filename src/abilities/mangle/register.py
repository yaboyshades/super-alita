#!/usr/bin/env python3
"""
Mangle Integration Registration for Super Alita

This module registers the Mangle deductive database programming integration
with Super Alita's ability registry and provides tool contracts.
"""

import logging
from typing import Any, Dict, List, Optional

from src.abilities.mangle.mangle_ability import MangleAbility
from src.core.events import create_event
from src.core.plugin_registry import register_plugin
from src.abilities.mangle.mangle_ability import ManglePluginInterface
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
        # Initialize the Mangle ability
        mangle_config = config.get("mangle", {}) if config else {}
        mangle_ability = MangleAbility(mangle_config)

        # Register the query ability
        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_query",
                "description": "Execute a Mangle deductive query against the knowledge base",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The Mangle query to execute"
                        },
                        "params": {
                            "type": "object",
                            "description": "Optional parameters for the query"
                        }
                    },
                    "required": ["query"]
                }
            }, 
            executor=lambda args: mangle_ability.query(
                args.get("query", ""),
                args.get("params", {})
            )
        )

        # Register the add fact ability
        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_add_fact",
                "description": "Add a fact to the Mangle knowledge base",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": "The fact to add to the knowledge base"
                        }
                    },
                    "required": ["fact"]
                }
            },
            executor=lambda args: mangle_ability.add_fact(args.get("fact", ""))
        )

        # Register the add rule ability
        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_add_rule",
                "description": "Add a rule to the Mangle knowledge base",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name for this rule"
                        },
                        "rule": {
                            "type": "string",
                            "description": "The Mangle rule to add"
                        }
                    },
                    "required": ["name", "rule"]
                }
            },
            executor=lambda args: mangle_ability.add_rule(
                args.get("name", ""),
                args.get("rule", "")
            )
        )

        # Register the dependency analysis ability
        ability_registry.register_tool(
            contract={
                "tool_id": "mangle_analyze_dependencies",
                "description": "Analyze project dependencies for vulnerabilities",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dependencies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The dependency name"
                                    },
                                    "version": {
                                        "type": "string",
                                        "description": "The dependency version"
                                    }
                                },
                                "required": ["name", "version"]
                            },
                            "description": "List of dependencies to analyze"
                        }
                    },
                    "required": ["dependencies"]
                }
            },
            executor=lambda args: mangle_ability.analyze_dependencies(
                args.get("dependencies", [])
            )
        )

        # Register the knowledge graph query ability
        ability_registry.register_tool({
            "tool_id": "mangle_knowledge_graph",
            "description": "Perform a knowledge graph query with context",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The knowledge graph query to execute"
                    },
                    "context": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "relation": {
                                    "type": "string",
                                    "description": "The relation type"
                                },
                                "subject": {
                                    "type": "string",
                                    "description": "The subject entity"
                                },
                                "object": {
                                    "type": "string",
                                    "description": "The object entity"
                                }
                            },
                            "required": ["relation", "subject", "object"]
                        },
                        "description": "Context for the knowledge graph query"
                    }
                },
                "required": ["query"]
            }
        }, lambda args: mangle_ability.knowledge_graph_query(
            args.get("query", ""),
            args.get("context", [])
        ))

        # Register the explanation ability
        ability_registry.register_tool({
            "tool_id": "mangle_explain",
            "description": "Execute a query and explain the results",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to explain"
                    }
                },
                "required": ["query"]
            }
        }, lambda args: mangle_ability.explain_query_results(
            args.get("query", "")
        ))

        # Register rule catalog ability
        ability_registry.register_tool({
            "tool_id": "mangle_rule_catalog",
            "description": "List available Mangle rules discovered on disk or current session",
            "input_schema": {"type": "object", "properties": {}},
        }, lambda args: mangle_ability.rule_catalog())

        # Register generic rule runner
        ability_registry.register_tool({
            "tool_id": "mangle_run_rule",
            "description": "Execute a one-off Mangle rule with a query over current facts",
            "input_schema": {
                "type": "object",
                "properties": {
                    "rule": {"type": "string", "description": "Rule clause to add for this run"},
                    "query": {"type": "string", "description": "Query to execute"}
                },
                "required": ["rule", "query"]
            }
        }, lambda args: mangle_ability.run_rule(
            args.get("rule", ""), args.get("query", "")
        ))

        # Dynamically expose per-rule tools if available on disk
        try:
            _register_dynamic_rule_tools(ability_registry, mangle_ability)
        except Exception as e:
            logger.warning(f"Dynamic Mangle rule registration skipped: {e}")

        logger.info("Mangle abilities registered successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to register Mangle abilities: {e}")
        return False


def register_mangle_plugin(_, config=None):
    """Register the Mangle plugin with the Super Alita plugin registry.
    
    Args:
        _: Unused parameter (kept for API compatibility)
        config: Optional configuration for the plugin
        
    Returns:
        True if registration successful, False otherwise
    """
    try:
        # Create plugin instance
        plugin = ManglePluginInterface()
        
        # Register with plugin registry
        register_plugin("mangle_plugin", plugin)
        
        logger.info("Mangle plugin registered successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to register Mangle plugin: {e}")
        return False

def export_mangle_rules_to_mcp_box(mcp_box_dir: str = ".mcp_box") -> dict[str, int]:
    """Mine known rules and export generic Mangle rule tools to MCP-Box.

    Writes mangle_rule_catalog and mangle_run_rule if not already present,
    then regenerates the catalog via abstractor.
    """
    import json
    from pathlib import Path

    box = Path(mcp_box_dir)
    box.mkdir(parents=True, exist_ok=True)

    base_specs = [
        {
            "tool_id": "mangle_rule_catalog",
            "description": "List available Mangle rules",
            "action": "mangle_rule_catalog",
            "input_schema": {"type": "object", "properties": {}},
            "output_schema": {"type": "object", "properties": {"rules": {"type": "array"}, "count": {"type": "integer"}}},
        },
        {
            "tool_id": "mangle_run_rule",
            "description": "Execute a one-off Mangle rule with a query over current facts",
            "action": "mangle_run_rule",
            "input_schema": {
                "type": "object",
                "required": ["rule", "query"],
                "properties": {"rule": {"type": "string"}, "query": {"type": "string"}},
            },
            "output_schema": {"type": "object", "properties": {"results": {"type": "array"}, "count": {"type": "integer"}}},
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

# ------------ Helpers: dynamic rule tooling ------------

def _sanitize(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "rule"


def _parse_rule_head(rule_body: str) -> tuple[str, list[str]]:
    """Return (predicate, [vars]) for the head of a Mangle/Datalog rule.
    Accepts either full rule with ':-' or a standalone head.
    """
    head = rule_body.split(":-", 1)[0].strip().rstrip(".")
    if not head:
        return "", []
    l = head.find("(")
    r = head.rfind(")")
    if l == -1 or r == -1 or r < l:
        return head, []
    pred = head[:l].strip()
    arg_str = head[l + 1 : r]
    vars_ = [a.strip() for a in arg_str.split(",") if a.strip()]
    return pred, vars_


def _register_dynamic_rule_tools(ability_registry, mangle_ability: MangleAbility) -> None:
    from pathlib import Path
    import json

    rules_file = Path("./data/mangle/rules.json")
    if not rules_file.exists():
        return
    rules_data = json.loads(rules_file.read_text(encoding="utf-8"))
    for rid, meta in rules_data.items():
        body = meta.get("body", "")
        pred, vars_ = _parse_rule_head(body)
        if not pred:
            continue
        tool_id = _sanitize(f"mangle_rule_{pred}")
        schema = {
            "tool_id": tool_id,
            "description": meta.get("description", f"Run Mangle rule {pred}"),
            "input_schema": {
                "type": "object",
                "properties": {v: {"type": "string"} for v in vars_},
            },
            "output_schema": {
                "type": "object",
                "properties": {"results": {"type": "array"}, "count": {"type": "integer"}},
            },
        }

        def _make_exec(rule_text: str, head_pred: str, head_vars: list[str]):
            async def _exec(args):
                parts: list[str] = []
                for v in head_vars:
                    val = args.get(v)
                    if val is None or (isinstance(val, str) and not val):
                        parts.append(v)
                    else:
                        try:
                            float(val)
                            parts.append(str(val))
                        except Exception:
                            s = str(val)
                            if not (s.startswith("'") and s.endswith("'")):
                                s = f"'{s}'"
                            parts.append(s)
                query = f"{head_pred}({', '.join(parts)})"
                return await mangle_ability.run_rule(rule_text, query)

            return _exec

        ability_registry.register_tool(schema, _make_exec(body, pred, vars_))
def export_mangle_to_mcp_box(mcp_box_dir: str = ".mcp_box") -> dict[str, int]:
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

    box = Path(mcp_box_dir)
    box.mkdir(parents=True, exist_ok=True)

    specs: list[dict[str, Any]] = [
        {
            "tool_id": "mangle_query",
            "description": "Execute a Mangle deductive query against the knowledge base",
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
            "description": "Analyze project dependencies for known vulnerabilities",
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
                "properties": {"rules": {"type": "array"}, "count": {"type": "integer"}},
            },
        },
        {
            "tool_id": "mangle_run_rule",
            "description": "Execute a one-off Mangle rule with a query over current facts",
            "action": "mangle_run_rule",
            "input_schema": {
                "type": "object",
                "required": ["rule", "query"],
                "properties": {"rule": {"type": "string"}, "query": {"type": "string"}},
            },
            "output_schema": {
                "type": "object",
                "properties": {"results": {"type": "array"}, "count": {"type": "integer"}},
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
