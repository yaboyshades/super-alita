# src/ecosystem/vscode_bridge.py
"""
A simulated VS Code Adapter that bridges the IDE to the backend orchestrator.

This script demonstrates the developer experience loop:
1. Detects a TODO in a local file.
2. Calls the orchestrator's FastAPI endpoint.
3. Simulates the IDE's reaction (inserting snippets, showing notifications).
"""
import re
from typing import Any

import aiohttp


class VSCodeBridgeSimulator:
    """Simulates a VS Code extension interacting with the orchestrator service."""

    def __init__(
        self,
        orchestrator_url: str = "http://localhost:8080",
        auth_token: str = "dev-token",
    ):
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.auth_token = auth_token
        self.user_id = "vscode_sim_user"

    async def scan_and_process_file(self, file_path: str):
        """Scans a file for TODOs and processes them through the orchestrator."""
        print(f"\n---  simulating vs code bridge for file: {file_path} ---")
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return

        # A simple regex to find TODOs for the simulation
        todo_pattern = re.compile(r"#\s*TODO[:\s]*(.*)", re.IGNORECASE)
        lines = content.splitlines()

        for i, line in enumerate(lines):
            match = todo_pattern.search(line)
            if match:
                todo_text = match.group(1).strip()
                print(
                    f"\n[VSCode Bridge] Detected TODO on line {i+1}: '{todo_text}'"
                )

                # Prepare context for the orchestrator
                context = {
                    "todo_text": todo_text,
                    "file_path": file_path,
                    "line_number": i + 1,
                    "context_lines": lines[
                        max(0, i - 3) : min(len(lines), i + 4)
                    ],
                }

                # Call the orchestrator service
                await self._call_orchestrator(context)

    async def _call_orchestrator(self, context: dict[str, Any]):
        """Makes an HTTP POST request to the orchestrator's FastAPI service."""
        url = f"{self.orchestrator_url}/api/v1/developer-action"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }
        payload = {
            "user_id": self.user_id,
            "action": "todo_detected",
            "context": context,
        }

        async with aiohttp.ClientSession() as session:
            try:
                print("[VSCode Bridge] Sending request to orchestrator...")
                async with session.post(
                    url, headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(
                            "[VSCode Bridge] Received successful response from orchestrator."
                        )
                        self._simulate_ide_actions(result)
                    else:
                        error_text = await response.text()
                        print(
                            f"[VSCode Bridge] Error from orchestrator ({response.status}): {error_text}"
                        )
            except aiohttp.ClientConnectorError:
                print(
                    "[VSCode Bridge] Error: Could not connect to the orchestrator service."
                )
                print(
                    "                 Please ensure the FastAPI service is running."
                )
            except Exception as e:
                print(f"[VSCode Bridge] Unexpected error: {e}")
                print(
                    "                 Connection failed or response malformed."
                )

    def _simulate_ide_actions(self, orchestrator_response: dict[str, Any]):
        """Prints messages to the console simulating what a real IDE extension would do."""
        print("\n--- [Simulated IDE Actions] ---")

        # 1. Simulate showing a toast notification with the Copilot prompt
        prompt = orchestrator_response.get("copilot_prompt", "")
        if prompt:
            print(
                "✅ [TOAST NOTIFICATION] Enhanced Context Ready for Copilot!"
            )
            print(
                f"   Ask Copilot: \"{prompt[:120].replace(chr(10), ' ')}...\""
            )

        # 2. Simulate inserting dynamic snippets
        snippets = orchestrator_response.get("vscode_snippets", [])
        if snippets:
            print(
                f"\n✅ [SNIPPET INSERTION] {len(snippets)} dynamic snippet(s) are now available."
            )
            for i, snippet in enumerate(snippets):
                print(
                    f"   - Snippet {i+1} (prefix: '{snippet.get('prefix')}'):"
                )
                print(
                    f"     Body: {snippet.get('body', '').replace(chr(10), ' ')}"
                )

        print("-------------------------------\n")
