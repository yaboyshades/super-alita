"""
Atom Executor Plugin for Super Alita
Executes stored code atoms safely in sandbox environment
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Any

from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class AtomExecutorPlugin(PluginInterface):
    """
    Plugin that executes stored code atoms safely.

    Features:
    - Execute atoms from memory storage
    - Safe subprocess execution with timeout
    - Return stdout/stderr results
    - Track execution history and performance
    """

    def __init__(self):
        super().__init__()
        self.execution_history = []
        self.active_executions = {}

    @property
    def name(self) -> str:
        return "atom_executor"

    async def setup(self, event_bus, store, config: dict[str, Any]) -> None:
        """Initialize the atom executor plugin."""
        await super().setup(event_bus, store, config)
        self.timeout_seconds = config.get("execution_timeout", 30)
        logger.info("AtomExecutorPlugin setup complete")

    async def start(self) -> None:
        """Start the atom executor plugin."""
        await super().start()

        # Subscribe to atom execution events
        await self.subscribe("atom_run", self._handle_atom_run)

        logger.info("AtomExecutorPlugin started - ready to execute atoms")

    async def shutdown(self) -> None:
        """Shutdown the atom executor plugin."""
        logger.info(
            f"AtomExecutorPlugin shutting down - executed {len(self.execution_history)} atoms"
        )

    async def _handle_atom_run(self, event) -> None:
        """Handle atom execution requests."""
        try:
            # Extract event data
            if hasattr(event, "model_dump"):
                data = event.model_dump()
            else:
                data = event.__dict__ if hasattr(event, "__dict__") else event

            tool_name = data.get("tool_name", "")
            args = data.get("args", "")
            requested_by = data.get("requested_by", "system")

            if not tool_name:
                logger.error("Invalid atom run request: missing tool_name")
                return

            logger.info(f"🚀 Executing atom: {tool_name} with args: {args}")

            # Retrieve atom from memory
            atom_data = await self._get_atom(tool_name)

            if not atom_data:
                await self.event_bus._redis.publish(
                    "agent_reply",
                    json.dumps({"text": f"❌ Atom '{tool_name}' not found in memory"}),
                )
                return

            # Execute the atom
            result = await self._execute_atom_safely(atom_data, args)

            # Emit result event
            await self.emit_event(
                "atom_result",
                tool_name=tool_name,
                args=args,
                stdout=result["stdout"],
                stderr=result["stderr"],
                success=result["success"],
                execution_time=result["execution_time"],
            )

            # Send result to user
            if result["success"]:
                await self.event_bus._redis.publish(
                    "agent_reply",
                    json.dumps(
                        {
                            "text": f"🚀 Atom '{tool_name}' executed successfully\n📤 Output: {result['stdout']}\n⏱️ Time: {result['execution_time']:.2f}s"
                        }
                    ),
                )
            else:
                await self.event_bus._redis.publish(
                    "agent_reply",
                    json.dumps(
                        {
                            "text": f"❌ Atom '{tool_name}' execution failed\n📤 Error: {result['stderr']}\n⏱️ Time: {result['execution_time']:.2f}s"
                        }
                    ),
                )

            # Store execution record
            self.execution_history.append(
                {
                    "tool_name": tool_name,
                    "args": args,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                    "requested_by": requested_by,
                }
            )

            logger.info(
                f"✅ Atom execution complete: {tool_name} -> {result['success']}"
            )

        except Exception as e:
            logger.error(f"Error executing atom: {e}")
            await self.event_bus._redis.publish(
                "agent_reply",
                json.dumps({"text": f"❌ Atom execution error: {e!s}"}),
            )

    async def _get_atom(self, tool_name: str) -> dict[str, Any] | None:
        """Retrieve atom from memory storage."""
        try:
            # Try semantic memory first
            if hasattr(self.store, "get"):
                atom_id = f"atom_{tool_name}"
                atom = self.store.get(atom_id)
                if atom and hasattr(atom, "value"):
                    atom_data = atom.value
                    if isinstance(atom_data, dict) and "code" in atom_data:
                        logger.info(
                            f"📚 Retrieved atom '{tool_name}' from semantic memory"
                        )
                        return atom_data

            # Try searching by content if direct lookup fails
            if hasattr(self.store, "embed_text") and hasattr(self.store, "attention"):
                try:
                    query_vec = await self.store.embed_text([f"Tool: {tool_name}"])
                    if query_vec and len(query_vec) > 0:
                        memories = await self.store.attention(query_vec[0], top_k=5)

                        for key, score in memories:
                            atom = self.store.get(key)
                            if atom and hasattr(atom, "value"):
                                atom_data = atom.value
                                if (
                                    isinstance(atom_data, dict)
                                    and atom_data.get("tool_name") == tool_name
                                ):
                                    logger.info(
                                        f"📚 Found atom '{tool_name}' via semantic search (score: {score:.2f})"
                                    )
                                    return atom_data
                except Exception as e:
                    logger.warning(f"Semantic search for atom failed: {e}")

            # Check with atom creator plugin if available
            for plugin_name, plugin in getattr(self.event_bus, "_plugins", {}).items():
                if hasattr(plugin, "get_atom"):
                    atom_data = await plugin.get_atom(tool_name)
                    if atom_data:
                        logger.info(
                            f"📚 Retrieved atom '{tool_name}' from {plugin_name}"
                        )
                        return atom_data

            logger.warning(f"Atom '{tool_name}' not found in any storage")
            return None

        except Exception as e:
            logger.error(f"Error retrieving atom '{tool_name}': {e}")
            return None

    async def _execute_atom_safely(
        self, atom_data: dict[str, Any], args: str
    ) -> dict[str, Any]:
        """Execute atom code safely in subprocess."""
        start_time = time.time()

        try:
            code = atom_data.get("code", "")
            tool_name = atom_data.get("tool_name", "unknown")

            if not code:
                return {
                    "stdout": "",
                    "stderr": "No code found in atom",
                    "success": False,
                    "execution_time": time.time() - start_time,
                }

            # Prepare code with argument injection
            if args:
                # For simple cases, treat args as input
                full_code = f"# Atom: {tool_name}\n# Args: {args}\n\n{code}"
                # If code expects input(), provide it via stdin
                stdin_data = args
            else:
                full_code = f"# Atom: {tool_name}\n\n{code}"
                stdin_data = ""

            # Create temporary file for code
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_code)
                temp_file = f.name

            try:
                # Execute with subprocess and timeout
                proc = await asyncio.create_subprocess_exec(
                    "python",
                    temp_file,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(
                            input=stdin_data.encode() if stdin_data else None
                        ),
                        timeout=self.timeout_seconds,
                    )

                    execution_time = time.time() - start_time

                    return {
                        "stdout": stdout.decode().strip(),
                        "stderr": stderr.decode().strip(),
                        "success": proc.returncode == 0,
                        "execution_time": execution_time,
                    }

                except TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return {
                        "stdout": "",
                        "stderr": f"Execution timed out after {self.timeout_seconds} seconds",
                        "success": False,
                        "execution_time": time.time() - start_time,
                    }

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Execution error: {e!s}",
                "success": False,
                "execution_time": time.time() - start_time,
            }

    async def get_execution_history(self) -> list:
        """Get execution history."""
        return self.execution_history.copy()

    async def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        total = len(self.execution_history)
        if total == 0:
            return {"total_executions": 0, "success_rate": 0.0}

        successful = sum(
            1 for record in self.execution_history if record["result"]["success"]
        )

        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": total - successful,
            "success_rate": successful / total,
            "active_executions": len(self.active_executions),
        }
