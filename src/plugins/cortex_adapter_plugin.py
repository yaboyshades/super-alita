"""
Cortex Adapter Plugin - Interface to external AI systems for bootstrapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol
import json
from datetime import datetime, timezone

from src.core.plugin_interface import PluginInterface
from src.core.event_bus import EventBus
from src.core.events import create_event
from src.core.temporal_graph import TemporalGraph, NeuralAtom
from src.core.navigation import NeuralNavigator, NavigationStrategy


class ExternalCortex(Protocol):
    """Protocol for external AI systems."""

    async def reason(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        ...

    async def analyze_problem(self, problem: str) -> List[Dict[str, Any]]:
        ...

    async def suggest_tools(self, intent: str) -> List[str]:
        ...


@dataclass
class CortexResponse:
    """Structured response from external cortex."""

    reasoning_steps: List[str]
    suggested_atoms: List[Dict[str, Any]]
    bonds_to_create: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any] | None = None


class GitHubCopilotCortex:
    """Integration with GitHub Copilot via API."""

    async def reason(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "reasoning_steps": ["Analyzed prompt."],
            "suggested_atoms": [
                {"content": "Rate limiter implementation", "type": "code", "metadata": {}}
            ],
            "bonds_to_create": [],
            "confidence": 0.85,
        }

    async def analyze_problem(self, problem: str) -> List[Dict[str, Any]]:
        return []

    async def suggest_tools(self, intent: str) -> List[str]:
        return []


class CortexAdapterPlugin(PluginInterface):
    """Plugin that adapts external AI systems as cognitive scaffolding."""

    def __init__(
        self, event_bus: EventBus, graph: TemporalGraph, navigator: NeuralNavigator
    ) -> None:
        self.event_bus = event_bus
        self.graph = graph
        self.navigator = navigator
        self.cortex_providers: Dict[str, ExternalCortex] = {}
        self.learning_history: List[Dict[str, Any]] = []

        # Register event handlers
        self.event_bus.subscribe("reasoning_request", self.handle_reasoning_request)
        self.event_bus.subscribe("knowledge_gap", self.handle_knowledge_gap)

    @property
    def name(self) -> str:  # type: ignore[override]
        return "cortex_adapter"

    async def setup(self, event_bus: Any, store: Any, config: Dict[str, Any]) -> None:  # type: ignore[override]
        self.event_bus = event_bus
        self.store = store
        self.config = config

    async def start(self) -> None:  # type: ignore[override]
        self.is_running = True

    def register_cortex(self, name: str, cortex: ExternalCortex) -> None:
        self.cortex_providers[name] = cortex

    async def handle_reasoning_request(self, event: Dict[str, Any]):
        """Handle requests for external reasoning assistance."""
        data = event.get("data", {}) if isinstance(event, dict) else {}
        prompt = data.get("prompt", "")
        context = dict(data.get("context", {}))

        current_knowledge = await self._build_context_from_graph(prompt)
        context.update(current_knowledge)

        for cortex_name, cortex in self.cortex_providers.items():
            try:
                raw = await cortex.reason(prompt, context)
                response = CortexResponse(
                    reasoning_steps=raw.get("reasoning_steps", []),
                    suggested_atoms=raw.get("suggested_atoms", []),
                    bonds_to_create=raw.get("bonds_to_create", []),
                    confidence=float(raw.get("confidence", 0.0)),
                    metadata=raw.get("metadata"),
                )

                # Learn from the response
                await self._learn_from_cortex_response(prompt, response, cortex_name)
                # Emit learned knowledge
                await self.event_bus.publish(
                    create_event(
                        "cortex_knowledge_learned",
                        event_version=1,
                        source_plugin=self.name,
                        prompt=prompt,
                        cortex_source=cortex_name,
                        atoms_created=len(response.suggested_atoms),
                        bonds_created=len(response.bonds_to_create),
                        confidence=response.confidence,
                    )
                )
            except Exception as e:  # pragma: no cover - defensive
                await self.event_bus.publish(
                    create_event(
                        "cortex_error",
                        event_version=1,
                        source_plugin=self.name,
                        cortex_name=cortex_name,
                        error=str(e),
                        prompt=prompt,
                    )
                )

    async def handle_knowledge_gap(self, event: Dict[str, Any]):
        """Handle detected knowledge gaps by querying cortex."""
        data = event.get("data", {}) if isinstance(event, dict) else {}
        gap_description = data.get("gap_description", "unspecified gap")
        context = dict(data.get("context", {}))

        gap_prompt = f"Help resolve knowledge gap: {gap_description}"
        await self.event_bus.publish(
            create_event(
                "reasoning_request",
                event_version=1,
                source_plugin=self.name,
                prompt=gap_prompt,
                context=context,
                source="knowledge_gap_detector",
            )
        )

    async def _build_context_from_graph(self, prompt: str) -> Dict[str, Any]:
        """Build context from current graph state."""
        relevant_atoms = await self.navigator.navigate(
            prompt,
            NavigationStrategy.SEMANTIC,
            depth=5,
            semantic_query=prompt,
        )

        return {
            "relevant_knowledge": [
                {
                    "content": atom.content,
                    "type": atom.atom_type,
                    "uuid": atom.uuid,
                }
                for atom in relevant_atoms[:10]
            ],
            "graph_stats": {
                "total_atoms": len(self.graph.atoms),
                "connection_count": sum(
                    len(a.bonds_out()) for a in self.graph.atoms.values()
                ),
            },
        }

    def _find_or_create_atom(self, content: str) -> Optional[NeuralAtom]:
        """Find existing atom or create new one."""
        content_lower = content.lower()
        for atom in self.graph.atoms.values():
            if content_lower in atom.content.lower():
                return atom
        return self.graph.create_atom(content, "concept", {})

    async def _learn_from_cortex_response(
        self, prompt: str, response: CortexResponse, cortex_name: str
    ) -> None:
        for atom_info in response.suggested_atoms:
            content = atom_info.get("content", "")
            atom_type = atom_info.get("type", "concept")
            metadata = atom_info.get("metadata", {})
            self._find_or_create_atom(content) or self.graph.create_atom(
                content, atom_type, metadata
            )
        for bond in response.bonds_to_create:
            src = bond.get("source")
            tgt = bond.get("target")
            if src and tgt:
                self.graph.create_bond(src, tgt, bond.get("metadata", {}))

        self.learning_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt,
                "cortex": cortex_name,
                "confidence": response.confidence,
            }
        )

    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about cortex-assisted learning."""
        if not self.learning_history:
            return {"total_sessions": 0}

        total_sessions = len(self.learning_history)
        avg_confidence = sum(
            s["confidence"] for s in self.learning_history
        ) / total_sessions
        cortex_sources = {s["cortex"] for s in self.learning_history}

        return {
            "total_sessions": total_sessions,
            "average_confidence": avg_confidence,
            "cortex_sources": list(cortex_sources),
            "recent_sessions": self.learning_history[-5:],
        }
