from __future__ import annotations

"""MCP Abstractor: normalize, deduplicate and index MCP specs in .mcp_box.

Creates/updates `.mcp_box/index.json` with a canonical view of MCP specs so the
runtime can avoid overload and surface a coherent, deduplicated catalog.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ID_RE = re.compile(r"[^a-z0-9_]+")


def _sanitize_tool_id(value: str) -> str:
    v = (value or "").strip().lower()
    v = v.replace(" ", "_")
    v = _ID_RE.sub("_", v)
    v = v.strip("_")
    return v or "tool"


def _props_from_schema(schema: dict[str, Any] | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    props = schema.get("properties")
    if isinstance(props, dict):
        return sorted([p for p in props.keys() if isinstance(p, str)])
    return []


def _required_from_schema(schema: dict[str, Any] | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    req = schema.get("required")
    if isinstance(req, list):
        return sorted([r for r in req if isinstance(r, str)])
    return []


def _compute_signature(spec: dict[str, Any]) -> str:
    action = (spec.get("action") or "").strip().lower()
    props = _props_from_schema(spec.get("input_schema"))
    req = _required_from_schema(spec.get("input_schema"))
    sig_obj = {"action": action, "props": props, "req": req}
    raw = json.dumps(sig_obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    tool_id = _sanitize_tool_id(spec.get("tool_id") or spec.get("name") or "tool")
    desc = (spec.get("description") or "").strip()
    action = (spec.get("action") or "").strip().lower() or "custom"
    in_schema = spec.get("input_schema") or {"type": "object"}
    out_schema = spec.get("output_schema") or {"type": "object"}
    return {
        "tool_id": tool_id,
        "description": desc,
        "action": action,
        "input_schema": in_schema,
        "output_schema": out_schema,
    }


@dataclass(slots=True)
class AbstractIndex:
    version: int
    generated_at: str
    total_files: int
    valid_specs: int
    canonical_count: int
    by_action: dict[str, list[str]]
    aliases: dict[str, list[str]]
    tools: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "total_files": self.total_files,
            "valid_specs": self.valid_specs,
            "canonical_count": self.canonical_count,
            "by_action": self.by_action,
            "aliases": self.aliases,
            "tools": self.tools,
        }


def abstract_mcp_box(box_dir: str | Path = ".mcp_box") -> dict[str, Any]:
    """Normalize, deduplicate and index specs under the MCP Box directory."""
    base = Path(box_dir)
    base.mkdir(parents=True, exist_ok=True)
    files = [p for p in base.glob("*.json") if p.name != "index.json"]

    canonical: dict[str, dict[str, Any]] = {}
    sig_to_canonical_id: dict[str, str] = {}
    id_aliases: dict[str, list[str]] = {}
    by_action: dict[str, list[str]] = {}
    tools_list: list[dict[str, Any]] = []

    valid = 0
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            norm = _normalize_spec(data)
            sig = _compute_signature(norm)
            tool_id = norm["tool_id"]
            valid += 1

            # Find or create canonical
            if sig not in sig_to_canonical_id:
                sig_to_canonical_id[sig] = tool_id
                canonical[tool_id] = {
                    **norm,
                    "signature": sig,
                    "files": [p.name],
                    "aliases": [],
                }
            else:
                cid = sig_to_canonical_id[sig]
                # Record file and alias if different id
                if cid in canonical:
                    canonical[cid].setdefault("files", []).append(p.name)
                    if tool_id != cid and tool_id not in canonical[cid]["aliases"]:
                        canonical[cid]["aliases"].append(tool_id)
                        id_aliases.setdefault(cid, []).append(tool_id)

        except Exception:
            # Skip invalid spec files
            continue

    # Build by_action and tools_list
    for cid, item in canonical.items():
        by_action.setdefault(item["action"], []).append(cid)
        tools_list.append(
            {
                "tool_id": cid,
                "action": item["action"],
                "properties": _props_from_schema(item.get("input_schema")),
                "required": _required_from_schema(item.get("input_schema")),
                "files": item.get("files", []),
                "aliases": item.get("aliases", []),
                "signature": item.get("signature"),
            }
        )

    # Sort for stable output
    for k in list(by_action.keys()):
        by_action[k] = sorted(by_action[k])
    tools_list.sort(key=lambda x: x["tool_id"])

    index = AbstractIndex(
        version=1,
        generated_at=datetime.now(UTC).isoformat(),
        total_files=len(files),
        valid_specs=valid,
        canonical_count=len(canonical),
        by_action=by_action,
        aliases=id_aliases,
        tools=tools_list,
    )

    # Write index.json
    (base / "index.json").write_text(
        json.dumps(index.to_dict(), indent=2), encoding="utf-8"
    )

    # Generate catalog.json for direct tool loading
    catalog_tools = []
    for cid, item in canonical.items():
        # Create a simplified tool spec that matches the expected tool catalog format
        catalog_entry = {
            "name": cid,  # Use canonical ID
            "description": item.get("description", ""),
            "input_schema": item.get("input_schema", {"type": "object"}),
            "output_schema": item.get("output_schema", {"type": "object"}),
        }
        catalog_tools.append(catalog_entry)

    # Sort catalog for stable output
    catalog_tools.sort(key=lambda x: x["name"])

    # Write catalog.json
    (base / "catalog.json").write_text(
        json.dumps(catalog_tools, indent=2), encoding="utf-8"
    )

    return index.to_dict()

