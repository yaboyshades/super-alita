#!/usr/bin/env python3
"""
Tool Lifecycle Plugin for Super Alita
Handles creation, storage, and execution of user-defined tools/atoms
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime
from typing import Any

from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class ToolLifecyclePlugin(PluginInterface):
    """
    Plugin that intercepts and handles tool lifecycle commands:
    - /create-atom name=X code='Y' - Creates and stores a new tool atom
    - /atom-run X args - Executes a stored tool atom with arguments
    """

    def __init__(self):
        super().__init__()
        self.created_atoms = {}  # Cache for quick lookup

    @property
    def name(self) -> str:
        return "tool_lifecycle"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        """Initialize the tool lifecycle plugin."""
        await super().setup(event_bus, store, config)
        logger.info("ToolLifecyclePlugin setup complete")

    async def start(self) -> None:
        """Start the plugin and subscribe to events."""
        await super().start()

        # Subscribe to the same channel as conversation plugin - we'll race to handle commands
        await self.subscribe("conversation_message", self._handle_conversation_commands)
        await self.subscribe("user_message", self._handle_user_commands)

        logger.info("ToolLifecyclePlugin started - listening for tool commands")

    async def _handle_conversation_commands(self, event: Any) -> None:
        """Handle tool commands from structured conversation events."""
        try:
            data = event.model_dump() if hasattr(event, "model_dump") else event
            user_message = data.get("user_message", "").strip()
            session_id = data.get("session_id", "default")

            if await self._process_tool_command(user_message, session_id):
                # Command was handled - prevent conversation plugin from processing
                logger.info(f"Tool command intercepted: {user_message[:50]}...")
                return

        except Exception as e:
            logger.error(f"Error handling conversation command: {e}")

    async def _handle_user_commands(self, event: Any) -> None:
        """Handle tool commands from direct user messages."""
        try:
            # Parse event data
            if hasattr(event, "model_dump"):
                data = event.model_dump()
            elif hasattr(event, "__dict__"):
                data = event.__dict__
            elif isinstance(event, dict):
                data = event
            else:
                try:
                    data = json.loads(str(event))
                except (json.JSONDecodeError, ValueError):
                    return

            user_message = data.get("text", "").strip()
            session_id = "chat_session"  # Default for chat client

            if await self._process_tool_command(user_message, session_id):
                logger.info(
                    f"Tool command intercepted from chat: {user_message[:50]}..."
                )
                return

        except Exception as e:
            logger.error(f"Error handling user command: {e}")

    async def _process_tool_command(self, message: str, session_id: str) -> bool:
        """
        Process tool commands and return True if handled.

        Supported formats:
        - /create-atom name=prime_counter code='import sympy; print(sympy.primepi(int(input())))'
        - /atom-run prime_counter 100
        """
        if not message.startswith("/"):
            return False

        # Parse /create-atom command
        create_match = re.match(r"^/create-atom\s+name=(\w+)\s+code='([^']+)'", message)
        if create_match:
            name, code = create_match.groups()
            await self._create_atom(name, code, session_id)
            return True

        # Parse /atom-run command
        run_match = re.match(r"^/atom-run\s+(\w+)\s+(.+)", message)
        if run_match:
            name, args = run_match.groups()
            await self._run_atom(name, args, session_id)
            return True

        # Also handle simpler format: /atom-run tool_name
        run_simple_match = re.match(r"^/atom-run\s+(\w+)$", message)
        if run_simple_match:
            name = run_simple_match.group(1)
            await self._run_atom(name, "", session_id)
            return True

        return False

    async def _create_atom(self, name: str, code: str, session_id: str) -> None:
        """Create and store a new tool atom."""
        try:
            atom_id = f"atom_{name}_{uuid.uuid4().hex[:8]}"
            atom_data = {
                "name": name,
                "code": code,
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "type": "user_tool",
                "executions": 0,
            }

            # Store in neural store
            if hasattr(self.store, "upsert"):
                await self.store.upsert(
                    memory_id=atom_id,
                    content=atom_data,
                    hierarchy_path=["tools", session_id, name],
                )
            elif hasattr(self.store, "set"):
                await self.store.set(atom_id, atom_data)
            else:
                # Fallback to local cache
                self.created_atoms[name] = atom_data

            logger.info(f"🧬 Atom created: {name}")

            # Send success response
            await self.event_bus._redis.publish(
                "agent_reply",
                json.dumps({"text": f"🧬 Atom '{name}' created and stored in memory"}),
            )

            # Emit event for other plugins
            await self.emit_event(
                "atom_created", name=name, atom_id=atom_id, session_id=session_id
            )

        except Exception as e:
            logger.error(f"Error creating atom {name}: {e}")
            await self.event_bus._redis.publish(
                "agent_reply",
                json.dumps({"text": f"❌ Failed to create atom '{name}': {e}"}),
            )

    async def _run_atom(self, name: str, args: str, session_id: str) -> None:
        """Execute a stored tool atom."""
        try:
            # Find the atom
            atom_data = await self._find_atom(name, session_id)
            if not atom_data:
                await self.event_bus._redis.publish(
                    "agent_reply",
                    json.dumps(
                        {"text": f"❌ Atom '{name}' not found in session {session_id}"}
                    ),
                )
                return

            code = atom_data["code"]
            logger.info(f"🔧 Executing atom: {name} with args: {args}")

            # Execute with real subprocess
            result = await self._safe_execute_code(code, args)

            # Update execution count
            atom_data["executions"] = atom_data.get("executions", 0) + 1

            # Send result
            await self.event_bus._redis.publish(
                "agent_reply",
                json.dumps({"text": f"🔧 Tool '{name}' executed → stdout: {result}"}),
            )

            # Emit event for other plugins
            await self.emit_event(
                "tool_result",
                tool_name=name,
                result=result,
                args=args,
                session_id=session_id,
            )

            logger.info(f"✅ Atom {name} executed → result: {result}")

        except Exception as e:
            logger.error(f"Error executing atom {name}: {e}")
            await self.event_bus._redis.publish(
                "agent_reply", json.dumps({"text": f"❌ Tool execution error: {e}"})
            )

    async def _find_atom(self, name: str, session_id: str) -> dict[str, Any] | None:
        """Find an atom by name in the session."""
        try:
            # Try neural store first
            if hasattr(self.store, "search"):
                results = await self.store.search(
                    {"name": name, "session_id": session_id}
                )
                if results:
                    return results[0]

            # Try local cache
            if name in self.created_atoms:
                atom = self.created_atoms[name]
                if atom.get("session_id") == session_id:
                    return atom

            # Try to find by any method available on store
            if hasattr(self.store, "get_all"):
                all_atoms = await self.store.get_all()
                for _atom_id, atom_data in all_atoms.items():
                    if (
                        isinstance(atom_data, dict)
                        and atom_data.get("name") == name
                        and atom_data.get("session_id") == session_id
                    ):
                        return atom_data

            return None

        except Exception as e:
            logger.warning(f"Error finding atom {name}: {e}")
            return None

    async def _safe_execute_code(self, code: str, args: str) -> str:
        """Safely execute Python code with subprocess."""
        try:
            # Create temporary file with the code
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                # If args provided, modify code to handle input
                if args:
                    # Replace input() calls with the provided args
                    modified_code = code.replace("input()", f'"{args}"')
                    f.write(modified_code)
                else:
                    f.write(code)
                temp_file = f.name

            try:
                # Execute with timeout for safety
                proc = await asyncio.create_subprocess_exec(
                    "python", temp_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=10.0
                )

                if proc.returncode == 0:
                    return stdout.decode().strip()
                return f"Error: {stderr.decode().strip()}"

            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

        except TimeoutError:
            return "Error: Tool execution timed out"
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return f"Error: {e!s}"

    async def get_status(self) -> dict[str, Any]:
        """Get plugin status."""
        return {
            "atoms_created": len(self.created_atoms),
            "available_atoms": list(self.created_atoms.keys()),
            "store_available": bool(self.store),
        }
