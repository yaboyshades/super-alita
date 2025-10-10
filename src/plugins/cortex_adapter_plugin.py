"""
Cortex Adapter Plugin - Interface to external AI systems for bootstrapping.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from src.core.events import create_event
from src.core.navigation import NavigationStrategy, NeuralNavigator
from src.core.plugin_interface import PluginInterface
from src.core.temporal_graph import NeuralAtom, TemporalGraph
from src.core.utils import (
    CircuitBreaker,
    blake2b_hexdigest,
    normalize_text,
    redact_prompt_and_context,
    sha256_json,
)


class ExternalCortex(Protocol):
    """Protocol for external AI systems."""

    async def reason(
        self, prompt: str, context: dict[str, Any]
    ) -> dict[str, Any]: ...

    async def analyze_problem(self, problem: str) -> list[dict[str, Any]]: ...

    async def suggest_tools(self, intent: str) -> list[str]: ...


@dataclass
class CortexResponse:
    """Structured response from external cortex."""

    reasoning_steps: list[str]
    suggested_atoms: list[dict[str, Any]]
    bonds_to_create: list[dict[str, Any]]
    confidence: float
    metadata: dict[str, Any] | None = None


class GitHubCopilotCortex:
    """Deterministic stub; replace with a real provider later."""

    async def reason(
        self, prompt: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "reasoning_steps": [
                "Analyze the problem context",
                "Identify key concepts and relationships",
                "Suggest solution approach",
            ],
            "suggested_atoms": [
                {
                    "content": f"Solution approach for: {prompt[:50]}",
                    "atom_type": "reasoning",
                    "metadata": {
                        "source": "github_copilot",
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                }
            ],
            "bonds_to_create": [
                {
                    "source_content": prompt[:30],
                    "target_content": f"Solution approach for: {prompt[:50]}",
                    "bond_type": "derives_solution",
                    "reason": "cortex_reasoning",
                    "context": context,
                }
            ],
            "confidence": 0.85,
        }


class CortexAdapterPlugin(PluginInterface):
    """Plugin that adapts external AI systems as cognitive scaffolding."""

    def __init__(self) -> None:
        self.graph: TemporalGraph | None = None
        self.navigator: NeuralNavigator | None = None
        self.cortex_providers: dict[str, ExternalCortex] = {}
        self.learning_history: list[dict[str, Any]] = []
        # Hardening
        self._dedup_index: dict[str, str] = {}
        self._quarantine_ttl_seconds = 7 * 24 * 3600
        self._circuit = CircuitBreaker(
            failure_threshold=5, recovery_timeout=30.0
        )
        self._budget_window_start = time.time()
        self._budget_calls = 0
        self._budget_max_per_min = 60

    @property
    def name(self) -> str:  # type: ignore[override]
        return "cortex_adapter"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:  # type: ignore[override]
        await super().setup(event_bus, store, config)
        self.graph = config.get("graph", self.graph)
        self.navigator = config.get("navigator", self.navigator)
        await self.event_bus.subscribe(
            "reasoning_request", self.handle_reasoning_request
        )
        await self.event_bus.subscribe(
            "knowledge_gap", self.handle_knowledge_gap
        )

    async def start(self) -> None:  # type: ignore[override]
        self.is_running = True

    def register_cortex(self, name: str, cortex: ExternalCortex) -> None:
        self.cortex_providers[name] = cortex

    async def handle_reasoning_request(self, event: dict[str, Any]):
        """Handle requests for external reasoning assistance."""
        data = event.get("data", {}) if isinstance(event, dict) else {}
        prompt = data.get("prompt", "")
        context = dict(data.get("context", {}))
        correlation_id = data.get("correlation_id") or str(uuid.uuid4())
        trace_id = data.get("trace_id") or correlation_id

        if not self._budget_allowed():
            await self.event_bus.publish(
                create_event(
                    "cortex_budget_exceeded",
                    event_version=1,
                    source_plugin=self.name,
                    correlation_id=correlation_id,
                    trace_id=trace_id,
                    reason="per-minute-limit",
                )
            )
            return

        if not self._circuit.allowed():
            await self.event_bus.publish(
                create_event(
                    "cortex_circuit_open",
                    event_version=1,
                    source_plugin=self.name,
                    correlation_id=correlation_id,
                    trace_id=trace_id,
                )
            )
            return

        current_knowledge = await self._build_context_from_graph(prompt)
        context.update(current_knowledge)

        red_prompt, red_context, red_report = redact_prompt_and_context(
            prompt, context
        )
        prompt_hash = sha256_json(red_prompt)
        context_hash = sha256_json(red_context)

        for cortex_name, cortex in self.cortex_providers.items():
            try:
                response_data = await cortex.reason(red_prompt, red_context)
                response = CortexResponse(**response_data)
                await self._learn_from_cortex_response(
                    prompt=red_prompt,
                    response=response,
                    source=cortex_name,
                    correlation_id=correlation_id,
                    trace_id=trace_id,
                    prompt_hash=prompt_hash,
                    context_hash=context_hash,
                )
                self._circuit.on_success()

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
                        correlation_id=correlation_id,
                        trace_id=trace_id,
                        prompt_hash=prompt_hash,
                        context_hash=context_hash,
                        redaction_summary=red_report,
                    )
                )
                break
            except Exception as e:  # pragma: no cover - defensive
                self._circuit.on_failure()
                await self.event_bus.publish(
                    create_event(
                        "cortex_error",
                        event_version=1,
                        source_plugin=self.name,
                        cortex_name=cortex_name,
                        error=str(e),
                        prompt=prompt,
                        correlation_id=correlation_id,
                        trace_id=trace_id,
                    )
                )

    async def handle_knowledge_gap(self, event: dict[str, Any]):
        """Handle detected knowledge gaps by querying cortex."""
        data = event.get("data", {}) if isinstance(event, dict) else {}
        gap_description = data.get("gap_description", "unspecified gap")
        context = dict(data.get("context", {}))
        gap_prompt = f"Fill knowledge gap: {gap_description}"
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

    async def _build_context_from_graph(self, prompt: str) -> dict[str, Any]:
        """Build context from current graph via semantic navigation."""
        relevant_atoms = await self.navigator.navigate(
            prompt,
            NavigationStrategy.SEMANTIC,
            depth=5,
            semantic_query=prompt,
        )
        return {
            "relevant_knowledge": [
                {"content": a.content, "type": a.atom_type, "uuid": a.uuid}
                for a in relevant_atoms[:10]
            ],
            "graph_stats": {
                "total_atoms": len(self.graph.atoms),
                "connection_count": sum(
                    len(a.bonds_out()) for a in self.graph.atoms.values()
                ),
            },
        }

    async def _learn_from_cortex_response(
        self,
        prompt: str,
        response: CortexResponse,
        source: str,
        *,
        correlation_id: str,
        trace_id: str,
        prompt_hash: str,
        context_hash: str,
    ) -> None:
        created_atoms: list[NeuralAtom] = []
        for atom_data in response.suggested_atoms:
            meta = {
                **atom_data.get("metadata", {}),
                "cortex_source": source,
                "learned_from": prompt,
                "confidence": response.confidence,
                "status": "quarantined",
                "provenance": {
                    "correlation_id": correlation_id,
                    "trace_id": trace_id,
                    "prompt_hash": prompt_hash,
                    "context_hash": context_hash,
                    "provider": source,
                },
            }
            atom = await self._create_or_get_atom(
                content=atom_data["content"],
                atom_type=atom_data.get("atom_type", "learned"),
                metadata=meta,
            )
            created_atoms.append(atom)
        for bond_data in response.bonds_to_create:
            src = self._find_or_create_atom(bond_data["source_content"])
            tgt = self._find_or_create_atom(bond_data["target_content"])
            if src and tgt:
                self.graph.create_bond(
                    src.uuid,
                    tgt.uuid,
                    {
                        **bond_data.get("context", {}),
                        "bond_type": bond_data.get(
                            "bond_type", "learned_relation"
                        ),
                        "reason": bond_data.get("reason", f"cortex_{source}"),
                        "cortex_source": source,
                        "confidence": response.confidence,
                        "status": "quarantined",
                        "provenance": {
                            "correlation_id": correlation_id,
                            "trace_id": trace_id,
                            "prompt_hash": prompt_hash,
                            "context_hash": context_hash,
                            "provider": source,
                        },
                    },
                )
        self.learning_history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "prompt": prompt,
                "cortex_source": source,
                "atoms_created": len(created_atoms),
                "confidence": response.confidence,
                "reasoning_steps": response.reasoning_steps,
            }
        )

    async def _create_or_get_atom(
        self, content: str, atom_type: str, metadata: dict[str, Any]
    ) -> NeuralAtom:
        key_norm = normalize_text(content or "")
        key = blake2b_hexdigest(f"{atom_type}|{key_norm}")
        existing_uuid = self._dedup_index.get(key)
        if existing_uuid:
            return self.graph.atoms[existing_uuid]
        atom = self.graph.create_atom(
            content=content, atom_type=atom_type, metadata=metadata
        )
        self._dedup_index[key] = atom.uuid
        return atom

    def _find_or_create_atom(self, content: str) -> Optional[NeuralAtom]:
        needle = (content or "").lower()
        for atom in self.graph.atoms.values():
            if needle and needle in (atom.content or "").lower():
                return atom
        return self.graph.create_atom(
            content=content,
            atom_type="auto_created",
            metadata={"source": "cortex_adapter"},
        )

    def _budget_allowed(self) -> bool:
        now = time.time()
        if (now - self._budget_window_start) >= 60.0:
            self._budget_window_start = now
            self._budget_calls = 0
        self._budget_calls += 1
        return self._budget_calls <= self._budget_max_per_min

    def maintenance_tick(self, now: Optional[float] = None) -> int:
        now = now or time.time()
        removed = 0
        # Quarantine GC hook
        return removed
