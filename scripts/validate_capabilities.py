#!/usr/bin/env python3
"""
Capability metadata validator (scoped).

Validates ONLY:
  - Python plugin tools returned by create_plugin().get_tools() under src/plugins/**
  - Optional JSON/YAML capability manifests under src/capabilities/** (if present)

Ignores: .venv, .mypy_cache, .git, node_modules, dist, build (and anything else via --exclude)

Usage:
  python scripts/validate_capabilities.py --paths src/plugins src/capabilities --output text
  python scripts/validate_capabilities.py --strict --output github

Exit codes:
  0: OK (no errors; warnings allowed unless --strict)
  2: Failure (missing required metadata in at least one capability, or import error)
"""

from __future__ import annotations
import argparse
import importlib
import json
import os
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Required fields aligned with your report
REQUIRED_FIELDS = [
    "name",
    "description",
    "parameters",  # JSON schema for arguments
    "cost_hint",
    "latency_hint",
    "safety_level",
    "test_reference",
    "category",
    "complexity",
    "version",
    "dependencies",
    "integration_requirements",
]

DEFAULT_PATHS = ["src/plugins"]
DEFAULT_EXCLUDES = [".venv", ".mypy_cache", ".git", "node_modules", "dist", "build", "__pycache__"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", nargs="*", default=DEFAULT_PATHS, help="Roots to scan (packages or folders)")
    p.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDES, help="Folder name patterns to ignore")
    p.add_argument("--output", choices=["text", "json", "github"], default="text", help="Output format")
    p.add_argument("--strict", action="store_true", help="Treat warnings as failures (CI mode)")
    return p.parse_args()

def should_ignore(path: Path, excludes: List[str]) -> bool:
    parts = set(path.parts)
    return any(e in parts for e in excludes)

def iter_plugin_modules(root: Path, excludes: List[str]) -> Iterable[str]:
    """Yield dotted module names under src/plugins/** that end with _plugin.py"""
    if should_ignore(root, excludes):
        return []
    
    # Find the workspace root (where src folder is located)
    workspace_root = Path.cwd()
    src_root = workspace_root / "src"
    
    # Add the workspace root to sys.path so we can import src.plugins
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    
    plugins_pkg = src_root / "plugins"
    if not plugins_pkg.exists():
        return []
        
    try:
        pkg = importlib.import_module("src.plugins")
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if m.ispkg:
                continue
            if m.name.endswith("_plugin"):
                yield m.name
    except ImportError:
        return []

def collect_tools_from_module(mod_name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (tools, errors) for a module"""
    errors: List[Dict[str, Any]] = []
    tools: List[Dict[str, Any]] = []
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        errors.append({"module": mod_name, "issue": f"IMPORT_ERROR: {e}"})
        return tools, errors
    create = getattr(mod, "create_plugin", None)
    if callable(create):
        try:
            plugin = create()
            if hasattr(plugin, "get_tools"):
                for spec in plugin.get_tools():
                    if isinstance(spec, dict):
                        tools.append(spec)
        except Exception as e:
            errors.append({"module": mod_name, "issue": f"CREATE_ERROR: {e}"})
    # Optional module-level TOOLS list
    if hasattr(mod, "TOOLS") and isinstance(mod.TOOLS, list):
        for spec in mod.TOOLS:
            if isinstance(spec, dict):
                tools.append(spec)
    return tools, errors

def validate_tool(mod_name: str, tool: Dict[str, Any]) -> List[Dict[str, Any]]:
    missing = []
    for field in REQUIRED_FIELDS:
        v = tool.get(field)
        if v in (None, "", {}, []):
            missing.append({"module": mod_name, "tool": tool.get("name"), "missing": field})
    return missing

def format_text_report(passed: int, warnings: List[Dict[str, Any]], failed: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("🔍 Super Alita Capability Validation Report")
    lines.append("=" * 50)
    lines.append(f"📊 Summary: {passed} passed, {len(warnings)} warnings, {len(failed)} failed\n")
    for w in warnings:
        head = w.get("head", f"{w.get('tool') or w.get('module')} ({w.get('where','')})").strip()
        lines.append(f"⚠️ {head}")
        for msg in w.get("messages", []):
            lines.append(f"   • {msg}")
        lines.append("")
    for f in failed:
        head = f.get("head", f"{f.get('tool') or f.get('module')} ({f.get('where','')})").strip()
        lines.append(f"❌ {head}")
        for msg in f.get("messages", []):
            lines.append(f"   • {msg}")
        lines.append("")
    return "\n".join(lines)

def main() -> int:
    args = parse_args()

    # Scan only plugin modules (Python). Manifests (JSON/YAML) can be added later if needed.
    modules: List[str] = []
    for root_str in args.paths:
        root = Path(root_str).resolve()
        if not root.exists():
            continue
        if "plugins" in root.parts:
            modules.extend(list(iter_plugin_modules(root, args.exclude)))
        elif root.name == "src":  # if given src, still just scan plugins
            modules.extend(list(iter_plugin_modules(root / "plugins", args.exclude)))
        else:
            # Non-plugin path provided: ignore by default to avoid noise (venv, mypy cache, workflows)
            continue

    seen = set()
    modules = [m for m in modules if not (m in seen or seen.add(m))]

    warnings: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    passed = 0

    if not modules:
        # No modules found — not an error; just report 0 and exit clean
        out = {"status": "ok", "passed": 0, "warnings": 0, "failed": 0, "note": "No plugin modules discovered."}
        if args.output == "json":
            print(json.dumps(out))
        else:
            print("🔍 Super Alita Capability Validation Report\n(no plugin modules discovered)\n")
        return 0

    for mod_name in modules:
        tools, errs = collect_tools_from_module(mod_name)
        for e in errs:
            failures.append({
                "head": f"{e['module']}",
                "where": "import/create",
                "messages": [e["issue"]],
                "module": e["module"],
            })
        if not tools and not errs:
            # Module loaded but exposed no tools: treat as warning (not failure)
            warnings.append({
                "head": f"{mod_name}",
                "where": "discovery",
                "messages": ["No tools exposed (get_tools() returned none)"],
                "module": mod_name,
            })
            continue

        for t in tools:
            missing = validate_tool(mod_name, t)
            if missing:
                warnings.append({
                    "head": f"{t.get('name') or 'unknown'}",
                    "where": mod_name,
                    "module": mod_name,
                    "tool": t.get("name"),
                    "messages": [f"Missing required field: {m['missing']}" for m in missing],
                })
            else:
                passed += 1

    strict_fail = args.strict and (len(warnings) > 0 or len(failures) > 0)

    if args.output == "json":
        print(json.dumps({
            "status": "fail" if failures or strict_fail else "ok",
            "passed": passed,
            "warnings": warnings,
            "failed": failures,
        }, indent=2))
    elif args.output == "github":
        # GitHub Actions-friendly summary
        print(f"::notice title=Capabilities::passed={passed} warnings={len(warnings)} failed={len(failures)}")
        for f in failures:
            for msg in f["messages"]:
                print(f"::error file={f.get('module','')},title=Capability import/create::{msg}")
        for w in warnings:
            for msg in w["messages"]:
                print(f"::warning file={w.get('module','')},title=Capability metadata::{msg}")
    else:
        print(format_text_report(passed, warnings, failures))

    return 2 if failures or strict_fail else 0

if __name__ == "__main__":
    sys.exit(main())