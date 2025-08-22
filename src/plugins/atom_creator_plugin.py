"""
Atom Creator Plugin for Super Alita
Creates and stores reusable code atoms in semantic memory
"""

import json
import logging
from datetime import datetime
from typing import Any

from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class AtomCreatorPlugin(PluginInterface):
    """
    Plugin that creates and stores code atoms in semantic memory.

    Features:
    - Create atoms from user commands
    - Store atoms with embeddings for retrieval
    - Validate atom structure and safety
    - Provide atom management capabilities
    """

    def __init__(self):
        super().__init__()
        self.created_atoms = {}

    @property
    def name(self) -> str:
        return "atom_creator"

    async def setup(self, event_bus, store, config: dict[str, Any]) -> None:
        """Initialize the atom creator plugin."""
        await super().setup(event_bus, store, config)
        logger.info("AtomCreatorPlugin setup complete")

    async def start(self) -> None:
        """Start the atom creator plugin."""
        await super().start()

        # Subscribe to atom creation events
        await self.subscribe("atom_create", self._handle_atom_create)

        logger.info("AtomCreatorPlugin started - ready to create atoms")

    async def shutdown(self) -> None:
        """Shutdown the atom creator plugin."""
        logger.info(
            f"AtomCreatorPlugin shutting down - created {len(self.created_atoms)} atoms"
        )

    async def _handle_atom_create(self, event) -> None:
        """Handle atom creation requests."""
        try:
            # Extract event data
            if hasattr(event, "model_dump"):
                data = event.model_dump()
            else:
                data = event.__dict__ if hasattr(event, "__dict__") else event

            tool_name = data.get("tool_name", "")
            description = data.get("description", "")
            code = data.get("code", "")
            created_by = data.get("created_by", "system")

            if not tool_name or not code:
                logger.error("Invalid atom creation request: missing tool_name or code")
                return

            logger.info(f"🧬 Creating atom: {tool_name}")

            # Validate code (basic safety check)
            if await self._validate_code(code):
                # Create atom structure
                atom_data = {
                    "tool_name": tool_name,
                    "description": description,
                    "code": code,
                    "created_by": created_by,
                    "created_at": datetime.now().isoformat(),
                    "type": "executable_atom",
                    "language": "python",
                    "validated": True,
                }

                # Store in memory with embedding
                atom_id = f"atom_{tool_name}"

                if hasattr(self.store, "upsert"):
                    try:
                        # Create searchable content for embedding
                        searchable_content = f"Tool: {tool_name}\nDescription: {description}\nCode: {code}"

                        await self.store.upsert(
                            memory_id=atom_id,
                            content=atom_data,
                            hierarchy_path=["atoms", "tools", tool_name],
                            metadata={
                                "type": "atom",
                                "tool_name": tool_name,
                                "created_by": created_by,
                                "searchable_content": searchable_content,
                            },
                        )

                        # Track locally
                        self.created_atoms[tool_name] = atom_data

                        logger.info(f"✅ Atom '{tool_name}' stored in semantic memory")

                        # Confirm to user via chat
                        await self.event_bus._redis.publish(
                            "agent_reply",
                            json.dumps(
                                {
                                    "text": f"🧬 Atom '{tool_name}' created and stored in semantic memory\n📝 Description: {description}\n🔧 Ready for execution with /atom-run"
                                }
                            ),
                        )

                    except Exception as e:
                        logger.error(f"Failed to store atom in semantic memory: {e}")
                        # Store locally as fallback
                        self.created_atoms[tool_name] = atom_data

                        await self.event_bus._redis.publish(
                            "agent_reply",
                            json.dumps(
                                {
                                    "text": f"🧬 Atom '{tool_name}' created (local storage only - semantic memory unavailable)\n📝 Description: {description}"
                                }
                            ),
                        )
                else:
                    # Store locally only
                    self.created_atoms[tool_name] = atom_data
                    logger.warning(
                        "Semantic memory not available - storing atom locally only"
                    )

                    await self.event_bus._redis.publish(
                        "agent_reply",
                        json.dumps(
                            {
                                "text": f"🧬 Atom '{tool_name}' created (session storage only)\n📝 Description: {description}"
                            }
                        ),
                    )

            else:
                logger.error(f"Code validation failed for atom: {tool_name}")
                await self.event_bus._redis.publish(
                    "agent_reply",
                    json.dumps(
                        {
                            "text": f"❌ Atom '{tool_name}' rejected - code failed safety validation"
                        }
                    ),
                )

        except Exception as e:
            logger.error(f"Error creating atom: {e}")
            await self.event_bus._redis.publish(
                "agent_reply",
                json.dumps({"text": f"❌ Atom creation failed: {e!s}"}),
            )

    async def _validate_code(self, code: str) -> bool:
        """Basic code validation for safety."""
        try:
            # Basic safety checks
            dangerous_imports = [
                "os.system",
                "subprocess.call",
                "eval(",
                "exec(",
                "__import__",
                "open(",
                "file(",
                "input()",
                "raw_input()",
            ]

            code_lower = code.lower()
            for danger in dangerous_imports:
                if danger in code_lower:
                    logger.warning(f"Dangerous code detected: {danger}")
                    return False

            # Try to compile the code
            compile(code, "<atom_code>", "exec")

            return True

        except SyntaxError as e:
            logger.error(f"Syntax error in atom code: {e}")
            return False
        except Exception as e:
            logger.error(f"Code validation error: {e}")
            return False

    async def get_atom(self, tool_name: str) -> dict[str, Any] | None:
        """Retrieve an atom by name."""
        # Try local storage first
        if tool_name in self.created_atoms:
            return self.created_atoms[tool_name]

        # Try semantic memory
        if hasattr(self.store, "get"):
            try:
                atom_id = f"atom_{tool_name}"
                atom = self.store.get(atom_id)
                if atom and hasattr(atom, "value"):
                    return atom.value
            except Exception as e:
                logger.warning(f"Failed to retrieve atom from semantic memory: {e}")

        return None

    async def list_atoms(self) -> dict[str, dict[str, Any]]:
        """List all created atoms."""
        return self.created_atoms.copy()
