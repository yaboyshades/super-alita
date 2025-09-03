#!/usr/bin/env python3
"""
Mangle Integration Simple Demo

This is a simplified demonstration of the Mangle integration with Super Alita
that focuses on direct usage of the MangleAbility class.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
root_dir = str(Path(__file__).parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Set up the mock Mangle binary
mock_path = Path(__file__).parent / "mock_mangle.py"
os.environ["MANGLE_BIN_PATH"] = str(mock_path)
try:
    os.chmod(str(mock_path), 0o755)  # Make it executable
except Exception as e:
    print(f"Warning: Could not make mock binary executable: {e}")

from src.abilities.mangle.mangle_ability import MangleAbility


async def simple_dependency_analysis():
    """Demonstrate a simple vulnerability detection with Mangle."""
    print("\n===== Simple Vulnerability Analysis =====")

    # Initialize the Mangle ability
    mangle = MangleAbility()

    # Add vulnerability data as facts
    print("Adding vulnerability data...")
    await mangle.add_fact("vulnerable('log4j', '2.14.0')")
    await mangle.add_fact("vulnerable('junit', '4.13.1')")
    await mangle.add_fact("safe('spring-core', '5.3.20')")

    # Query for vulnerable dependencies
    print("Querying for vulnerable dependencies...")
    query_result = await mangle.query("vulnerable(Name, Version)")

    # Display results
    print(f"Found {query_result.get('count', 0)} vulnerable dependencies:")
    for item in query_result.get("results", []):
        name = item.get("Name", "unknown")
        version = item.get("Version", "unknown")
        print(f"- {name} {version} is vulnerable")

    return query_result


async def simple_knowledge_graph():
    """Demonstrate a simple knowledge graph with Mangle."""
    print("\n===== Simple Knowledge Graph =====")

    # Initialize the Mangle ability
    mangle = MangleAbility()

    # Add component relationships as facts
    print("Building simple knowledge graph...")
    await mangle.add_fact("depends_on('Frontend', 'Backend')")
    await mangle.add_fact("depends_on('Backend', 'Database')")
    await mangle.add_fact("depends_on('Frontend', 'Authentication')")

    # Query for dependencies
    print("Querying component dependencies...")
    query_result = await mangle.query("depends_on(Component, Dependency)")

    # Display results
    print(f"Found {query_result.get('count', 0)} component dependencies:")
    for item in query_result.get("results", []):
        component = item.get("Component", "unknown")
        dependency = item.get("Dependency", "unknown")
        print(f"- {component} depends on {dependency}")

    return query_result


async def main():
    """Run the simplified Mangle demo."""
    print("=== Simplified Mangle Integration Demo ===")

    await simple_dependency_analysis()
    await simple_knowledge_graph()

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
