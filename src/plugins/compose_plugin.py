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
import os
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

from src.core.config import EMBEDDING_DIM
from src.core.events import BaseEvent, ToolCallEvent
from src.core.plugin_interface import PluginInterface

"""
ComposePlugin - Dynamic Atom Composition
Stitches existing atoms together to create new combined capabilities
"""

logger = logging.getLogger(__name__)

# Use the same client library everywhere to avoid circular imports
try:
    import google.generativeai as genai

    _GENAI_OK = True
except Exception as e:
    genai = None
    _GENAI_OK = False
    logger.debug("compose: google-generativeai not available: %s", e)


class ComposeRequestEvent(BaseEvent):
    """Event requesting composition of atoms for a goal"""

    event_type: str = "compose_request"
    goal: str
    params: dict[str, Any] = {}


class DynamicAtom:
    """Dynamic atom structure for composed capabilities"""

    def __init__(
        self,
        tool: str = None,
        code: str = "",
        description: str = "",
        category: str = "composed",
    ):
        self.tool = tool or f"combo_{uuid4().hex[:8]}"
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


class ComposePlugin(PluginInterface):
    """
    Plugin that composes existing atoms into new combined capabilities.

    Features:
    - Finds relevant atoms using semantic similarity
    - Uses LLM to stitch atoms together intelligently
    - Creates new composed atoms
    - Executes composed tools
    """

    def __init__(self):
        super().__init__()
        self.composed_atoms = []
        self.llm_client = None

    @property
    def name(self) -> str:
        return "compose"

    async def setup(self, event_bus, store, config: dict[str, Any]) -> None:
        """Initialize the compose plugin."""
        await super().setup(event_bus, store, config)

        # Initialize LLM client for composition
        try:
            api_key = (
                (config.get("compose", {}) or {}).get("gemini_api_key")
                or config.get("gemini_api_key")
                or os.getenv("GEMINI_API_KEY", "")
            )
            if _GENAI_OK and api_key:
                genai.configure(api_key=api_key)
                model_name = (
                    (config.get("compose", {}) or {}).get("llm_model")
                    or config.get("llm_model")
                    or "gemini-1.5-flash"
                )
                self._model = genai.GenerativeModel(model_name=model_name)
                logger.info("ComposePlugin: Gemini client ready (%s)", model_name)
            else:
                self._model = None
                logger.info(
                    "ComposePlugin: Gemini not available - using template composition (fallback OK)"
                )
        except Exception as e:
            logger.error("Failed to initialize LLM client: %s", e)
            self._model = None

        logger.info("ComposePlugin setup complete")

    async def start(self) -> None:
        """Start the compose plugin."""
        await super().start()

        # Subscribe to composition requests
        await self.subscribe("compose_request", self._compose)
        await self.subscribe("user_message", self._handle_compose_commands)

        logger.info("ComposePlugin started - ready to compose atoms")

    async def shutdown(self) -> None:
        """Shutdown the compose plugin."""
        logger.info(
            f"Shutting down ComposePlugin - composed {len(self.composed_atoms)} atoms total"
        )

    async def _handle_compose_commands(self, event) -> None:
        """Handle chat commands for composition"""
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

            # Handle /compose command
            if text.startswith("/compose"):
                if "goal=" in text:
                    goal = (
                        text.split('goal="')[1].split('"')[0]
                        if 'goal="' in text
                        else ""
                    )
                    if goal:
                        logger.info(f"Composing atoms for goal: {goal}")

                        # Create compose request event
                        compose_event = ComposeRequestEvent(
                            source_plugin=self.name, goal=goal
                        )
                        await self._compose(compose_event)

                        # Send response to chat
                        await self.event_bus._redis.publish(
                            "agent_reply",
                            json.dumps(
                                {
                                    "text": f"🔧 Composing atoms for goal: {goal}... Check logs for results!"
                                }
                            ),
                        )

            # Handle /atoms listing command
            elif text.strip() == "/atoms":
                atoms = await self._list_atoms()
                atom_list = "\n".join(
                    [
                        f"• {atom.get('tool', 'unknown')}: {atom.get('description', 'no description')}"
                        for atom in atoms[:10]
                    ]
                )

                response = f"🧠 **Current Atoms** ({len(atoms)} total):\n{atom_list}"
                if len(atoms) > 10:
                    response += f"\n... and {len(atoms) - 10} more"

                await self.event_bus._redis.publish(
                    "agent_reply", json.dumps({"text": response})
                )

            # Handle /atom-create command
            elif text.startswith("/atom-create"):
                await self._handle_atom_create(text)

        except Exception as e:
            logger.error(f"Error handling compose command: {e}")

    async def _handle_atom_create(self, text: str) -> None:
        """Handle manual atom creation command"""
        try:
            # Parse: /atom-create tool="echo" code='print("hi")'
            tool = ""
            code = ""

            if 'tool="' in text:
                tool = text.split('tool="')[1].split('"')[0]

            if "code='" in text:
                code = text.split("code='")[1].split("'")[0]
            elif 'code="' in text:
                code = text.split('code="')[1].split('"')[0]

            if tool and code:
                # Create and store atom
                atom = DynamicAtom(
                    tool=tool,
                    code=code,
                    description=f"Manually created {tool} atom",
                    category="manual",
                )

                memory_id = f"manual_{uuid4().hex}"
                await self._store_atom(memory_id, atom)

                logger.info(f"✅ Manual atom created: {tool}")

                await self.event_bus._redis.publish(
                    "agent_reply",
                    json.dumps({"text": f"✅ Created atom: {tool} with code: {code}"}),
                )
            else:
                await self.event_bus._redis.publish(
                    "agent_reply",
                    json.dumps(
                        {
                            "text": "❌ Invalid format. Use: /atom-create tool=\"name\" code='code_here'"
                        }
                    ),
                )

        except Exception as e:
            logger.error(f"Error creating manual atom: {e}")

    async def _compose(self, event) -> None:
        """Core composition logic - finds and stitches atoms together"""
        try:
            goal = getattr(event, "goal", "general task")
            params = getattr(event, "params", {})

            logger.info(f"🔧 Starting composition for goal: {goal}")

            # 1. Find relevant atoms by embedding similarity
            relevant_atoms = await self._find_relevant_atoms(goal)

            if not relevant_atoms:
                logger.warning(f"No relevant atoms found for goal: {goal}")
                return

            logger.info(
                f"📊 Found {len(relevant_atoms)} relevant atoms for composition"
            )

            # 2. Ask Gemini to stitch them together
            if self._model:
                composed_code = await self._llm_compose(goal, relevant_atoms)
            else:
                composed_code = await self._fallback_compose(goal, relevant_atoms)

            if not composed_code:
                logger.warning("Failed to generate composed code")
                return

            # 3. Create and register composed atom
            combo_atom = DynamicAtom(
                code=composed_code,
                description=f"Composed atom for: {goal}",
                category="composed",
            )

            combo_id = f"combo_{uuid4().hex}"
            await self._store_atom(combo_id, combo_atom)

            self.composed_atoms.append(combo_atom)

            logger.info(f"✅ ComposePlugin — stitched atoms → new atom id={combo_id}")

            # 4. Execute the composed tool
            await self._execute_composed_tool(combo_atom, params)

            logger.info(f"🎯 Composition complete for goal: {goal}")

        except Exception as e:
            logger.error(f"Error in composition: {e}")

    async def _find_relevant_atoms(self, goal: str) -> list[dict[str, Any]]:
        """Find atoms relevant to the goal using embedding similarity"""
        try:
            if hasattr(self.store, "embed_text") and hasattr(self.store, "attention"):
                # Generate embedding for the goal
                goal_embeddings = await self.store.embed_text([goal])
                if not goal_embeddings:
                    logger.warning("Failed to generate goal embedding")
                    return []

                goal_vec = goal_embeddings[0]

                # Find similar atoms
                memories = await self.store.attention(goal_vec, top_k=5)

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
                                atom_data["similarity_score"] = float(score)
                                atoms.append(atom_data)
                        except Exception:
                            continue

                return atoms
            logger.warning("Embedding/attention not available - returning all atoms")
            return await self._list_atoms()

        except Exception as e:
            logger.error(f"Error finding relevant atoms: {e}")
            return []

    async def _list_atoms(self) -> list[dict[str, Any]]:
        """List all available atoms"""
        try:
            if hasattr(self.store, "attention"):
                zero_vec = np.zeros(1024, dtype=np.float32)
                memories = await self.store.attention(zero_vec, top_k=50)

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
            return []

        except Exception as e:
            logger.error(f"Error listing atoms: {e}")
            return []

    async def _llm_compose(self, goal: str, atoms: list[dict[str, Any]]) -> str:
        """Use LLM to compose atoms into unified code"""
        try:
            atom_descriptions = []
            for atom in atoms:
                desc = f"Tool: {atom.get('tool', 'unknown')}\n"
                desc += f"Code: {atom.get('code', '')}\n"
                desc += f"Description: {atom.get('description', '')}\n"
                atom_descriptions.append(desc)

            prompt = f"""You are a code composer. Combine the following atoms to achieve the goal.

Available Atoms:
{chr(10).join(atom_descriptions)}

Goal: {goal}

Return ONLY executable Python code that combines these atoms to achieve the goal.
The code should be self-contained and practical.
Do not include explanations or markdown formatting."""

            response = await asyncio.wait_for(
                self._call_gemini_async(prompt), timeout=30.0
            )

            # Clean up the response
            code = response.strip()
            if code.startswith("```python"):
                code = code.replace("```python", "").replace("```", "").strip()
            elif code.startswith("```"):
                code = code.replace("```", "").strip()

            logger.info(f"🧠 Generated composed code: {code[:100]}...")
            return code

        except Exception as e:
            logger.error(f"LLM composition failed: {e}")
            return ""

    async def _fallback_compose(self, goal: str, atoms: list[dict[str, Any]]) -> str:
        """Fallback composition when LLM is unavailable"""
        # Simple concatenation of atom codes
        code_parts = []
        for atom in atoms[:3]:  # Use first 3 atoms
            atom_code = atom.get("code", "")
            if atom_code:
                code_parts.append(f"# {atom.get('tool', 'unknown')}")
                code_parts.append(atom_code)

        return "\n".join(code_parts) if code_parts else 'print("No atoms to compose")'

    async def _execute_composed_tool(
        self, atom: DynamicAtom, params: dict[str, Any]
    ) -> None:
        """Execute the composed tool"""
        try:
            # Create tool call event with correct schema
            import time

            call_id = f"compose_{int(time.time() * 1000)}_{atom.tool}"
            tool_event = ToolCallEvent(
                source_plugin=self.name,
                tool_name=atom.tool,
                parameters=params,
                conversation_id=params.get("conversation_id", "unknown"),
                session_id=params.get("session_id", "unknown"),
                tool_call_id=call_id,
            )

            await self.event_bus.publish(tool_event)

            # Also try direct execution for demonstration
            try:
                # Simple execution - in production this should be sandboxed
                result = eval(atom.code) if atom.code else "No code to execute"
                logger.info(f"🔧 Tool executed → result: {str(result)[:100]}")
            except Exception as e:
                logger.info(f"🔧 Tool execution attempted → error: {str(e)[:100]}")

        except Exception as e:
            logger.error(f"Error executing composed tool: {e}")

    async def _store_atom(self, memory_id: str, atom: DynamicAtom) -> None:
        """Store composed atom in semantic memory"""
        try:
            if hasattr(self.store, "upsert"):
                await self.store.upsert(
                    memory_id=memory_id,
                    content=atom.model_dump(),
                    hierarchy_path=["composed", atom.category],
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
                logger.info(
                    f"Stored composed atom {memory_id} directly in neural store"
                )

        except Exception as e:
            logger.error(f"Failed to store composed atom {memory_id}: {e}")

    async def _call_gemini_async(self, prompt: str) -> str:
        """Async wrapper for Gemini API call"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.llm_client.generate_content(prompt)
        )
        return response.text

    async def get_stats(self) -> dict[str, Any]:
        """Get composition statistics"""
        return {
            "atoms_composed": len(self.composed_atoms),
            "recent_compositions": [atom.tool for atom in self.composed_atoms[-5:]],
            "llm_available": bool(self.llm_client),
        }
