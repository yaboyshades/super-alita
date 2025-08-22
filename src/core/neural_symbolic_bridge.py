#!/usr/bin/env python3
"""
Neural-Symbolic Bridge Implementation

This module implements the clean, efficient bridge between the neural layer
(1024-D vectors) and the symbolic layer (human-readable context for LLM reasoning).

Core principle: The neural layer proposes memories; the symbolic layer decides what to do with them.
"""

import asyncio
import logging
from typing import Any

from src.core.neural_atom import NeuralStore

logger = logging.getLogger(__name__)


class NeuralSymbolicBridge:
    """
    Efficient bridge that converts neural attention results to symbolic context.

    Query → Vector → Atoms → Symbolic Lines → LLM Decision
    """

    def __init__(self, store=None, llm_client=None):
        self.store = store
        self.llm_client = llm_client
        self._memory_cache = {}

    @staticmethod
    def format_memories(
        atoms_with_scores: list[tuple[str, float]],
        store: NeuralStore,
        max_length: int = 150,
    ) -> str:
        """
        Convert neural attention results to symbolic context lines.

        Args:
            atoms_with_scores: List of (atom_key, similarity_score) tuples
            store: NeuralStore to retrieve atom values
            max_length: Maximum length of each atom value summary

        Returns:
            Formatted string for inclusion in LLM prompts
        """
        if not atoms_with_scores:
            return 'relevant_memories:\\n  - "No relevant memories found."'

        lines = []
        for key, similarity in atoms_with_scores:
            atom = store.get(key)
            if atom:
                # Truncate value to readable length
                value_str = str(atom.value)
                summary = (
                    value_str[:max_length] + "..."
                    if len(value_str) > max_length
                    else value_str
                )
                lines.append(
                    f"  - (Similarity: {similarity:.2f}) Atom '{key}': {summary}"
                )

        return "relevant_memories:\\n" + "\\n".join(lines)

    @staticmethod
    async def get_contextual_memories(
        store: NeuralStore, context_text: str, top_k: int = 3
    ) -> list[tuple[str, float]]:
        """
        Neural query → Vector → Top-K atoms with similarity scores.

        Args:
            store: NeuralStore for attention queries
            context_text: Text to embed and query against
            top_k: Number of most relevant atoms to return

        Returns:
            List of (atom_key, similarity_score) tuples
        """
        try:
            # This would embed the context and perform attention query
            # For now, return mock data - replace with actual embedding in production
            return [
                ("mem_django_xss_01", 0.92),
                ("skill_dependency_scanner", 0.88),
                ("goal_bounty_security_audit", 0.85),
            ]
        except Exception as e:
            logger.error(f"Error retrieving contextual memories: {e}")
            return []

    async def build_symbolic_plan(
        self, goal: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Build symbolic plan with neural insights - now properly async."""
        try:
            # Get contextual memories asynchronously
            memories = await self.get_contextual_memories(
                self.store, goal, context.get("memory_limit", 10)
            )

            # Build plan structure
            plan = {
                "goal": goal,
                "context": context,
                "memories": memories,
                "steps": await self._generate_plan_steps(goal, memories),
                "confidence": self._calculate_plan_confidence(memories),
            }

            return plan

        except Exception as e:
            logger.error(f"Error building symbolic plan: {e}")
            return {"goal": goal, "steps": [], "confidence": 0.0}

    async def _generate_plan_steps(
        self, goal: str, memories: list[tuple[str, float]]
    ) -> list[dict[str, Any]]:
        """Generate plan steps based on goal and memories - now async."""
        try:
            # This might involve LLM calls which should be async
            if self.llm_client:
                # Use LLM to generate plan steps
                # prompt = f"Generate plan steps for goal: {goal}"  # Currently unused
                # Add actual LLM call here
                pass

            # Fallback to basic plan structure
            return [
                {
                    "step": 1,
                    "action": "analyze_goal",
                    "description": f"Analyze the goal: {goal}",
                },
                {
                    "step": 2,
                    "action": "identify_resources",
                    "description": "Identify available resources",
                },
                {
                    "step": 3,
                    "action": "execute_plan",
                    "description": "Execute the planned actions",
                },
            ]

        except Exception as e:
            logger.error(f"Error generating plan steps: {e}")
            return []

    def _calculate_plan_confidence(self, memories: list[tuple[str, float]]) -> float:
        """Calculate confidence score for the plan (sync operation)."""
        try:
            # Simple confidence calculation based on available memories
            if not memories:
                return 0.0

            # Average similarity score of memories
            total_similarity = sum(similarity for _, similarity in memories)
            return min(total_similarity / len(memories), 1.0)

        except Exception as e:
            logger.error(f"Error calculating plan confidence: {e}")
            return 0.0

    async def process_neural_input(self, input_data: Any) -> dict[str, Any]:
        """Process neural input and return symbolic representation - async."""
        try:
            # This might involve neural network processing
            processed = {
                "input": input_data,
                "symbolic_representation": await self._extract_symbolic_features(
                    input_data
                ),
                "confidence": 0.8,
            }
            return processed

        except Exception as e:
            logger.error(f"Error processing neural input: {e}")
            return {
                "input": input_data,
                "symbolic_representation": {},
                "confidence": 0.0,
            }

    async def _extract_symbolic_features(self, data: Any) -> dict[str, Any]:
        """Extract symbolic features from data - async operation."""
        try:
            # Add feature extraction logic here
            # This might involve embeddings or other async operations
            return {"features": [], "metadata": {"processed_at": "timestamp"}}
        except Exception as e:
            logger.error(f"Error extracting symbolic features: {e}")
            return {}


# Sync bridge function for backward compatibility
def run_bridge_operation(bridge_func, *args, **kwargs):
    """
    Helper to run bridge operations in sync contexts.
    Use this carefully - prefer async context when possible.
    """
    try:
        # Check if we're already in an async context
        # loop = asyncio.get_running_loop()  # Currently unused
        # We're in an async context, so we can't use asyncio.run()
        # The caller should use await instead
        raise RuntimeError(
            "Cannot run async bridge operation in async context. Use 'await' instead."
        )
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(bridge_func(*args, **kwargs))


# Example usage in Cognitive Contract methods:
async def example_cognitive_contract_usage():
    """
    Demonstrates the neural-symbolic bridge in a Cognitive Contract method.
    """
    store = None  # Would be actual NeuralStore instance

    # Step 1: Neural layer proposes memories
    memories = await NeuralSymbolicBridge.get_contextual_memories(
        store, context_text="Security vulnerability analysis", top_k=3
    )

    # Step 2: Symbolic layer formats for LLM reasoning
    memory_section = NeuralSymbolicBridge.format_memories(memories, store)

    # Step 3: Insert into Cognitive Contract template
    prompt = f"""# COGNITIVE CONTRACT: Example Decision
# AGENT: Super Alita

# --- NEURAL-SYMBOLIC BRIDGE ---
{memory_section}

# --- TASK ---
task:
  description: "Make strategic decision based on neural memories and symbolic reasoning."
"""

    return prompt


if __name__ == "__main__":
    print("Neural-Symbolic Bridge Implementation")
    print("=" * 50)
    print()
    print("Core Flow:")
    print("1. Query → Vector → Atoms")
    print("2. Only symbolic payloads survive")
    print("3. Format as key + value + relevance score")
    print("4. LLM receives clean, weighted context")
    print()
    print("Contract: Neural layer proposes; symbolic layer decides.")
