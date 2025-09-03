# scripts/demo_todo_workflow.py
"""
A command-line script to demonstrate the end-to-end TODO resolution 
workflow.

This script initializes the orchestrator with a real GitHub bridge 
(if a token is provided) and a StdoutEventBus to make the internal 
workings of the system visible.

Usage:
  # With no GitHub integration (uses no-op):
  python scripts/demo_todo_workflow.py "Implement a fast sorting algorithm"

  # With real GitHub search:
  export GITHUB_TOKEN="your_ghp_token_here"
  python scripts/demo_todo_workflow.py \
    "Implement LRU cache decorator in Python" \
    --file "src/utils/cache.py"
"""

import argparse
import asyncio
import os
import sys

# Add the source directory to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ecosystem.eventbus import StdoutEventBus
from src.ecosystem.github_bridge import (
    CopilotContextEnhancerFromGitHub,
    GitHubCodeSearchBridge,
)
from src.ecosystem.master_orchestrator import EcosystemOrchestrator
from src.ecosystem.telemetry import Telemetry


async def main():
    parser = argparse.ArgumentParser(
        description="Demo the REUG TODO Resolution Workflow."
    )
    parser.add_argument("todo_text", type=str, help="The text of the TODO comment.")
    parser.add_argument(
        "--file",
        type=str,
        default="src/example.py",
        help="The file path where the TODO was detected.",
    )
    parser.add_argument(
        "--user", type=str, default="demo_user", help="The user ID for the demo."
    )
    args = parser.parse_args()

    print("--- 🚀 Initializing REUG Ecosystem Orchestrator Demo ---")

    # Use the StdoutEventBus to see all events printed to the console.
    event_bus = StdoutEventBus()
    telemetry = (
        Telemetry()
    )  # Uses Noop sink by default, but collects in-memory counters

    # Wire in the real GitHub bridge if a token is available.
    github_token = os.getenv("GITHUB_TOKEN")
    copilot_enhancer = None
    if github_token:
        print("✅ GITHUB_TOKEN found. Enabling real GitHub Code Search.")
        bridge = GitHubCodeSearchBridge(token=github_token)
        copilot_enhancer = CopilotContextEnhancerFromGitHub(bridge)
    else:
        print("⚠️ GITHUB_TOKEN not found. Using No-Op for GitHub search.")

    # Initialize the orchestrator with our components.
    orchestrator = EcosystemOrchestrator(
        event_bus=event_bus,
        telemetry=telemetry,
        copilot_enhancer=copilot_enhancer,
    )

    print("\n--- ⚡ Simulating Developer Action: TODO Detected ---")
    print(f"User: {args.user}")
    print(f"File: {args.file}")
    print(f'TODO: "{args.todo_text}"')
    print("\n--- 📣 Orchestrator Events (Live) ---")

    # This is the main call to the orchestrator.
    result = await orchestrator.handle_developer_action(
        user_id=args.user,
        action="todo_detected",
        context={"todo_text": args.todo_text, "file_path": args.file},
    )

    print("\n--- ✅ Orchestration Complete: Final Result ---")
    print(f"Workflow Type: {result.get('workflow_type')}")
    print(f"Estimated Effort: {result.get('estimated_effort')}")
    print(f"Confidence: {result.get('confidence')}")

    print("\n--- 🤖 Synthesized Copilot Prompt ---")
    print(result.get("copilot_prompt"))

    print("\n--- ✂️ Generated VSCode Snippets ---")
    for i, snippet in enumerate(result.get("vscode_snippets", [])):
        print(f"Snippet {i+1} (prefix: {snippet['prefix']}):")
        print(snippet["body"])

    print("\n--- 📊 Telemetry Counters ---")
    print(
        f"TODO Workflows Run: {telemetry.get_counter('workflow_runs.todo_resolution')}"
    )


if __name__ == "__main__":
    asyncio.run(main())
