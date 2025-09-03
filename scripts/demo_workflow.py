# scripts/demo_workflow.py
"""
A command-line script to demonstrate the end-to-end TODO resolution 
and code integration workflows.

This script initializes the orchestrator with a real GitHub bridge 
(if a token is provided) and a StdoutEventBus to make the internal 
workings of the system visible.

Usage:
  # TODO workflow with no GitHub integration (uses no-op):
  python scripts/demo_workflow.py todo "Implement a fast sorting algorithm"

  # TODO workflow with real GitHub search:
  export GITHUB_TOKEN="your_ghp_token_here"
  python scripts/demo_workflow.py todo \
    "Implement LRU cache decorator in Python" \
    --file "src/utils/cache.py"

  # Code integration workflow:
  python scripts/demo_workflow.py integrate path/to/code_file.py \
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
    parser = argparse.ArgumentParser(description="Demo the REUG Ecosystem Workflows.")
    
    parser.add_argument(
        "mode", choices=["todo", "integrate"], help="The workflow mode to run."
    )
    parser.add_argument(
        "text_input",
        type=str,
        help="The TODO text or path to a file with code to integrate.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="src/example.py",
        help="The file path where the TODO was detected or where code should be integrated.",
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

    if args.mode == "todo":
        action = "todo_detected"
        context = {"todo_text": args.text_input, "file_path": args.file}
        print(f"\n--- ⚡ Simulating: TODO Detected ---")
        print(f"User: {args.user}")
        print(f"File: {args.file}")
        print(f'TODO: "{args.text_input}"')
    elif args.mode == "integrate":
        action = "code_pasted"
        try:
            with open(args.text_input, "r") as f:
                pasted_code = f.read()
            context = {"pasted_code": pasted_code, "file_path": args.file}
            print(f"\n--- ⚡ Simulating: Code Pasted from '{args.text_input}' ---")
            print(f"User: {args.user}")
            print(f"Target File: {args.file}")
            print(f"Code length: {len(pasted_code)} characters")
        except FileNotFoundError:
            print(f"Error: Input file not found at '{args.text_input}'")
            return

    print("\n--- 📣 Orchestrator Events (Live) ---")

    # This is the main call to the orchestrator.
    result = await orchestrator.handle_developer_action(
        user_id=args.user,
        action=action,
        context=context,
    )

    print("\n--- ✅ Orchestration Complete: Final Result ---")
    print(f"Workflow Type: {result.get('workflow_type')}")

    if args.mode == "integrate":
        print("\n--- ✅ Integration Plan ---")
        print(f"Compliance Score: {result.get('compliance_score', 0.0):.2f}")
        print(f"Issues Found: {result.get('issues_found', 0)}")
        print("\n--- 🤖 Suggested Refactoring Prompts for Copilot ---")
        for i, prompt in enumerate(result.get("refactoring_prompts", [])):
            print(f"{i+1}. {prompt}")

        related_files = result.get("related_files", [])
        if related_files:
            print(f"\n--- 📁 Related Files Found ---")
            for file in related_files:
                print(f"  - {file}")
    else:  # TODO mode
        print(f"Estimated Effort: {result.get('estimated_effort')}")
        print(f"Confidence: {result.get('confidence')}")

        print("\n--- 🤖 Synthesized Copilot Prompt ---")
        print(result.get("copilot_prompt"))

        print("\n--- ✂️ Generated VSCode Snippets ---")
        for i, snippet in enumerate(result.get("vscode_snippets", [])):
            print(f"Snippet {i+1} (prefix: {snippet['prefix']}):")
            print(snippet["body"])

    print("\n--- 📊 Telemetry Counters ---")
    if args.mode == "integrate":
        print(
            f"Integration Workflows Run: {telemetry.get_counter('workflow_runs.integration') if telemetry else 0}"
        )
    else:
        print(
            f"TODO Workflows Run: {telemetry.get_counter('workflow_runs.todo_resolution') if telemetry else 0}"
        )


if __name__ == "__main__":
    asyncio.run(main())
