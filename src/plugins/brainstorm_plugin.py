#!/usr/bin/env python3

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, environment variables should be set manually

import asyncio
import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

from src.core.config import EMBEDDING_DIM
from src.core.events import BaseEvent
from src.core.plugin_interface import PluginInterface

"""
BrainstormPlugin - Dynamic Atom Gap Discovery
Identifies missing capabilities and creates new atoms to fill gaps
"""

logger = logging.getLogger(__name__)


class AtomGapRequestEvent(BaseEvent):
    """Event requesting brainstorming for missing atoms"""

    event_type: str = "atom_gap_request"
    task: str
    context: dict[str, Any] = {}


class AtomReadyEvent(BaseEvent):
    """Event indicating a new atom is ready"""

    event_type: str = "atom_ready"
    atom: dict[str, Any]


class DynamicAtom:
    """Dynamic atom structure for brainstormed capabilities"""

    def __init__(
        self,
        tool: str,
        code: str,
        description: str = "",
        category: str = "brainstormed",
    ):
        self.tool = tool
        self.code = code
        self.description = description
        self.category = category
        self.created_at = datetime.now().isoformat()

    def model_dump(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "code": self.code,
            "description": self.description,
            "category": self.category,
            "created_at": self.created_at,
            "type": "dynamic_atom",
        }


class BrainstormPlugin(PluginInterface):
    """
    Plugin that identifies capability gaps and brainstorms new atoms to fill them.

    Features:
    - Analyzes current atom inventory
    - Uses LLM to identify missing capabilities
    - Creates new dynamic atoms
    - Stores atoms in semantic memory for future retrieval
    """

    def __init__(self):
        super().__init__()
        self.created_atoms = []
        self.llm_client = None

    @property
    def name(self) -> str:
        return "brainstorm"

    async def setup(self, event_bus, store, config: dict[str, Any]) -> None:
        """Initialize the brainstorm plugin."""
        await super().setup(event_bus, store, config)

        # Initialize LLM client for brainstorming using REST API
        try:
            import os

            self.api_key = os.getenv("GEMINI_API_KEY")
            if self.api_key:
                self.model_name = "gemini-1.5-flash"
                self.base_url = (
                    "https://generativelanguage.googleapis.com/v1beta/models"
                )
                logger.info("Gemini REST API configured for brainstorming")
                self.llm_client = True  # Flag to indicate LLM is available
            else:
                logger.warning("GEMINI_API_KEY not set - brainstorming will be limited")
                self.llm_client = None

        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None

        logger.info("BrainstormPlugin setup complete")

    async def start(self) -> None:
        """Start the brainstorm plugin."""
        await super().start()

        # Subscribe to gap analysis requests
        await self.subscribe("atom_gap_request", self._brainstorm)
        await self.subscribe("user_message", self._handle_brainstorm_commands)

        logger.info("BrainstormPlugin started - ready to identify capability gaps")

    async def shutdown(self) -> None:
        """Shutdown the brainstorm plugin."""
        logger.info(
            f"Shutting down BrainstormPlugin - created {len(self.created_atoms)} atoms total"
        )

    async def _handle_brainstorm_commands(self, event) -> None:
        """Handle chat commands for brainstorming"""
        try:
            # Parse the event data
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

            text = data.get("text", "")
            if not text.startswith("/brainstorm"):
                return

            # Parse command: /brainstorm gap="web scraping"
            if "gap=" in text:
                gap = text.split('gap="')[1].split('"')[0] if 'gap="' in text else ""
                if gap:
                    logger.info(f"Brainstorming atoms for gap: {gap}")

                    # Create gap request event
                    gap_event = AtomGapRequestEvent(source_plugin=self.name, task=gap)
                    await self._brainstorm(gap_event)

                    # Send response to chat
                    await self.event_bus._redis.publish(
                        "agent_reply",
                        json.dumps(
                            {
                                "text": f"🧠 Brainstorming new atoms for: {gap}... Check logs for results!"
                            }
                        ),
                    )

        except Exception as e:
            logger.error(f"Error handling brainstorm command: {e}")

    async def _brainstorm(self, event) -> None:
        """Core brainstorming logic - identifies gaps and creates new atoms"""
        try:
            task = getattr(event, "task", "general capability enhancement")
            logger.info(f"🧠 Starting brainstorm session for: {task}")

            # 1. List current atoms from memory
            current_atoms = await self._get_current_atoms()
            current_names = {atom.get("tool", "unknown") for atom in current_atoms}

            logger.info(
                f"📊 Found {len(current_atoms)} existing atoms: {list(current_names)[:10]}..."
            )

            # 2. Ask Gemini for gap analysis
            if not self.llm_client:
                logger.warning(
                    "LLM client not available - using fallback brainstorming"
                )
                new_atoms = await self._fallback_brainstorm(task, current_names)
            else:
                new_atoms = await self._llm_brainstorm(task, current_names)

            # 3. Register each new atom
            for atom_data in new_atoms:
                try:
                    atom = DynamicAtom(**atom_data)

                    # Store in semantic memory
                    memory_id = f"brain_{uuid4().hex}"
                    await self._store_atom(memory_id, atom)

                    # Track locally
                    self.created_atoms.append(atom)

                    # Emit ready event
                    ready_event = AtomReadyEvent(
                        source_plugin=self.name, atom=atom.model_dump()
                    )
                    await self.event_bus.publish(ready_event)

                    logger.info(
                        f"✅ BrainstormPlugin — created atom id={memory_id}, tool={atom.tool}"
                    )

                except Exception as e:
                    logger.error(f"Failed to create atom {atom_data}: {e}")

            logger.info(
                f"🎯 Brainstorm session complete - created {len(new_atoms)} new atoms"
            )

        except Exception as e:
            logger.error(f"Error in brainstorm session: {e}")

    async def _get_current_atoms(self) -> list[dict[str, Any]]:
        """Get current atoms from semantic memory"""
        try:
            if hasattr(self.store, "attention"):
                # Use attention mechanism to get top atoms
                zero_vec = np.zeros(1024, dtype=np.float32)
                memories = await self.store.attention(zero_vec, top_k=100)

                atoms = []
                for key, score in memories:
                    atom = self.store.get(key)
                    if atom and hasattr(atom, "value"):
                        try:
                            atom_data = atom.value
                            if (
                                isinstance(atom_data, dict)
                                and atom_data.get("type") == "dynamic_atom"
                            ):
                                atoms.append(atom_data)
                        except Exception:
                            continue

                return atoms
            logger.warning(
                "Attention mechanism not available - returning empty atom list"
            )
            return []

        except Exception as e:
            logger.error(f"Error getting current atoms: {e}")
            return []

    async def _llm_brainstorm(
        self, task: str, current_names: set
    ) -> list[dict[str, Any]]:
        """Use LLM to brainstorm missing atoms"""
        try:
            prompt = f"""You are a capability gap analyst. Analyze the current toolset and suggest missing atoms.

Current atoms: {list(current_names)[:20]}
Task context: {task}

Please suggest 3 missing atoms that would be useful for this task. Return ONLY a JSON list in this exact format:
[
  {{"tool": "tool_name", "code": "python_code_here", "description": "brief description"}},
  {{"tool": "tool_name2", "code": "python_code_here", "description": "brief description"}},
  {{"tool": "tool_name3", "code": "python_code_here", "description": "brief description"}}
]

Make the code practical and executable. Focus on filling genuine capability gaps."""

            response = await asyncio.wait_for(
                self._call_gemini_async(prompt), timeout=30.0
            )

            # Parse JSON response
            try:
                atoms = json.loads(response.strip())
                if isinstance(atoms, list) and len(atoms) > 0:
                    return atoms
                logger.warning("LLM returned invalid atom list format")
                return []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM brainstorm response: {e}")
                logger.debug(f"Raw response: {response[:200]}...")
                return []

        except Exception as e:
            logger.error(f"LLM brainstorming failed: {e}")
            return []

    async def _fallback_brainstorm(
        self, task: str, current_names: set
    ) -> list[dict[str, Any]]:
        """Fallback brainstorming when LLM is unavailable"""
        # Simple rule-based atom suggestions
        common_gaps = [
            {
                "tool": "file_reader",
                "code": "with open(filename, 'r') as f: return f.read()",
                "description": "Read file contents",
            },
            {
                "tool": "web_fetch",
                "code": "import requests; return requests.get(url).text",
                "description": "Fetch web content",
            },
            {
                "tool": "json_parser",
                "code": "import json; return json.loads(data)",
                "description": "Parse JSON data",
            },
        ]

        # Return gaps not already present
        missing_atoms = []
        for atom in common_gaps:
            if atom["tool"] not in current_names:
                missing_atoms.append(atom)

        return missing_atoms[:3]  # Return up to 3

    async def _store_atom(self, memory_id: str, atom: DynamicAtom) -> None:
        """Store atom in semantic memory"""
        try:
            # Check if store has semantic memory capability
            if hasattr(self.store, "upsert"):
                await self.store.upsert(
                    memory_id=memory_id,
                    content=atom.model_dump(),
                    hierarchy_path=["brainstormed", atom.category],
                )
            else:
                # Fallback: create neural atom directly
                from src.core.neural_atom import NeuralAtom

                # Generate embedding for searchability
                if hasattr(self.store, "embed_text"):
                    embedding_text = f"{atom.tool} {atom.description} {atom.code[:100]}"
                    embeddings = await self.store.embed_text([embedding_text])
                    vector = (
                        embeddings[0]
                        if embeddings
                        else np.random.rand(EMBEDDING_DIM).astype(np.float32)
                    )
                else:
                    vector = np.random.rand(1024).astype(np.float32)

                neural_atom = NeuralAtom(
                    key=memory_id, default_value=atom.model_dump(), vector=vector
                )

                self.store.add(neural_atom)
                logger.info(f"Stored atom {memory_id} directly in neural store")

        except Exception as e:
            logger.error(f"Failed to store atom {memory_id}: {e}")

    async def _call_gemini_async(self, prompt: str) -> str:
        """Async wrapper for Gemini API call"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.llm_client.generate_content(prompt)
        )
        return response.text

    async def get_stats(self) -> dict[str, Any]:
        """Get brainstorming statistics"""
        return {
            "atoms_created": len(self.created_atoms),
            "recent_atoms": [atom.tool for atom in self.created_atoms[-5:]],
            "llm_available": bool(self.llm_client),
        }
