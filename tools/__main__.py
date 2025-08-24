#!/usr/bin/env python3
"""
Super Alita Tools Discovery CLI

Usage:
    python -m tools.list [--format text|json] [--category <category>] [--complexity <level>]

Examples:
    python -m tools.list                               # List all tools
    python -m tools.list --format json                 # JSON output  
    python -m tools.list --category utility            # Filter by category
    python -m tools.list --complexity low              # Filter by complexity
"""

import argparse
import importlib
import json
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

def discover_tools() -> List[Dict[str, Any]]:
    """Discover all available tools from plugins"""
    tools = []
    
    # Ensure src is in path
    workspace_root = Path.cwd()
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    
    src_root = workspace_root / "src"
    plugins_pkg = src_root / "plugins"
    if not plugins_pkg.exists():
        return tools
    
    try:
        pkg = importlib.import_module("src.plugins")
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if m.ispkg:
                continue
            if m.name.endswith("_plugin"):
                try:
                    mod = importlib.import_module(m.name)
                    create_fn = getattr(mod, "create_plugin", None)
                    if callable(create_fn):
                        plugin = create_fn()
                        if hasattr(plugin, "get_tools"):
                            plugin_tools = plugin.get_tools()
                            if plugin_tools:
                                for tool in plugin_tools:
                                    if isinstance(tool, dict):
                                        # Add source plugin info
                                        tool["source_plugin"] = m.name.split(".")[-1]
                                        tools.append(tool)
                except Exception as e:
                    # Skip plugins that can't be loaded
                    continue
    except ImportError:
        return tools
    
    return tools

def filter_tools(tools: List[Dict[str, Any]], category: Optional[str] = None, 
                complexity: Optional[str] = None) -> List[Dict[str, Any]]:
    """Filter tools by category and/or complexity"""
    filtered = tools
    
    if category:
        filtered = [t for t in filtered if t.get("category") == category]
    
    if complexity:
        filtered = [t for t in filtered if t.get("complexity") == complexity]
    
    return filtered

def format_text_output(tools: List[Dict[str, Any]]) -> str:
    """Format tools as human-readable text"""
    if not tools:
        return "No tools found matching criteria.\n"
    
    lines = []
    lines.append("🔧 Super Alita Available Tools")
    lines.append("=" * 40)
    lines.append(f"📊 Found {len(tools)} tools\n")
    
    # Group by category
    by_category = {}
    for tool in tools:
        cat = tool.get("category", "uncategorized")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(tool)
    
    for category, cat_tools in sorted(by_category.items()):
        lines.append(f"📂 {category.title()} ({len(cat_tools)} tools)")
        lines.append("-" * 30)
        
        for tool in sorted(cat_tools, key=lambda x: x.get("name", "")):
            name = tool.get("name", "unknown")
            desc = tool.get("description", "No description")
            complexity = tool.get("complexity", "unknown")
            version = tool.get("version", "unknown")
            source = tool.get("source_plugin", "unknown")
            
            lines.append(f"🛠️  {name} (v{version})")
            lines.append(f"   📝 {desc}")
            lines.append(f"   🔧 Complexity: {complexity} | Source: {source}")
            
            # Show parameters if available
            params = tool.get("parameters", {})
            if isinstance(params, dict) and "properties" in params:
                props = params["properties"]
                if props:
                    param_names = list(props.keys())
                    required = params.get("required", [])
                    param_summary = []
                    for p in param_names:
                        if p in required:
                            param_summary.append(f"{p}*")
                        else:
                            param_summary.append(p)
                    lines.append(f"   📋 Parameters: {', '.join(param_summary)}")
            
            lines.append("")
        
        lines.append("")
    
    return "\n".join(lines)

def format_json_output(tools: List[Dict[str, Any]]) -> str:
    """Format tools as JSON"""
    return json.dumps({
        "tools_count": len(tools),
        "tools": tools
    }, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Discover and list Super Alita tools")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                       help="Output format")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--complexity", choices=["low", "medium", "high", "critical"],
                       help="Filter by complexity level")
    
    args = parser.parse_args()
    
    # Discover all available tools
    tools = discover_tools()
    
    # Apply filters
    filtered_tools = filter_tools(tools, args.category, args.complexity)
    
    # Output in requested format
    if args.format == "json":
        print(format_json_output(filtered_tools))
    else:
        print(format_text_output(filtered_tools))

if __name__ == "__main__":
    main()