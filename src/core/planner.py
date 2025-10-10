"""Planner module for generating execution plans in Super-Alita.

This module implements a retrieval-augmented planner that uses semantic search
and graph expansion to propose a chain of operations (atoms) that can satisfy
an input query. It also validates that the input/output contracts between consecutive
atoms are compatible and caches successful plans for reuse.
"""

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()

# Type alias for atom identifier (could be name or unique ID of an atom/tool)
AtomId = str


class VectorIndex:
    """Placeholder for a semantic vector index that can find similar atoms."""

    def find_similar(
        self, query: str, top_k: int = 3
    ) -> list[tuple[AtomId, float]]:
        # In a real implementation, this would return a list of
        # (atom_id, similarity_score).
        # Here we return an empty list by default; in tests or integration,
        # this should be overridden.
        del query, top_k  # Avoid unused argument warnings
        return []


class GraphStore:
    """Placeholder for a graph store that holds relationships and contracts."""

    def __init__(self):
        # adjacency list: which atoms can follow a given atom in a plan
        self.adjacency: dict[AtomId, list[AtomId]] = {}
        # define input/output "contracts" for atoms (using required field names
        # for simplicity)
        self.input_contract: dict[AtomId, set[str]] = {}
        self.output_contract: dict[AtomId, set[str]] = {}

    def add_atom(
        self, atom_id: AtomId, input_fields: set[str], output_fields: set[str]
    ):
        """Register an atom with its input and output contract."""
        self.input_contract[atom_id] = input_fields
        self.output_contract[atom_id] = output_fields
        if atom_id not in self.adjacency:
            self.adjacency[atom_id] = []

    def add_edge(self, from_atom: AtomId, to_atom: AtomId):
        """Declare that 'to_atom' can follow 'from_atom' in a plan."""
        if from_atom not in self.adjacency:
            self.adjacency[from_atom] = []
        self.adjacency[from_atom].append(to_atom)

    def get_neighbors(self, atom_id: AtomId) -> list[AtomId]:
        """Get atoms that can follow the given atom (direct neighbors)."""
        return list(self.adjacency.get(atom_id, []))

    def is_contract_compatible(
        self, from_atom: AtomId, to_atom: AtomId
    ) -> bool:
        """Check if output of `from_atom` satisfies input of `to_atom`."""
        out_fields = self.output_contract.get(from_atom, set())
        in_fields = self.input_contract.get(to_atom, set())
        # All required input fields of to_atom must be present in from_atom's
        # output
        return in_fields.issubset(out_fields)


@dataclass
class PlanStep:
    atom_id: AtomId
    description: str | None = (
        None  # optional human-readable description of the atom
    )


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)
    rationale: str = ""  # explanation of why this plan was proposed

    def add_step(self, atom_id: AtomId, description: str = ""):
        self.steps.append(PlanStep(atom_id=atom_id, description=description))

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return len(self.steps)


@dataclass
class PlanProposedEvent:
    query: str
    plan: Plan

    def to_dict(self) -> dict[str, Any]:
        """Represent the event in dictionary form (for logging or publishing)."""
        return {
            "query": self.query,
            "plan_steps": [step.atom_id for step in self.plan.steps],
            "rationale": self.plan.rationale,
        }


class Planner:
    """Planner that proposes a plan (sequence of atoms) for a given input query."""

    def __init__(
        self,
        vector_index: VectorIndex,
        graph_store: GraphStore,
        max_steps: int = 3,
        cache_size: int = 50,
    ):
        self.vector_index = vector_index
        self.graph_store = graph_store
        self.max_steps = max_steps
        # Simple LRU cache for recently used plans (query -> Plan)
        self._plan_cache: dict[str, Plan] = {}
        self._cache_order: list[str] = []  # tracks query order for eviction
        self._cache_size = cache_size

    def propose_plan(self, query: str) -> PlanProposedEvent:
        logger.info("Planner invoked", query=query)
        # 1. Check cache: return cached plan if available
        if query in self._plan_cache:
            plan = self._plan_cache[query]
            logger.info(
                "Returning cached plan",
                query=query,
                plan_steps=[s.atom_id for s in plan.steps],
            )
            # Update cache recency order
            self._refresh_cache_key(query)
            return PlanProposedEvent(query=query, plan=plan)

        # 2. Retrieve candidate atoms via semantic similarity search
        candidates = self.vector_index.find_similar(query, top_k=5)
        if not candidates:
            # No relevant atoms found; return an empty plan event
            empty_plan = Plan(rationale="No relevant atoms found for query.")
            logger.warning(
                "No candidates from vector index for query", query=query
            )
            return PlanProposedEvent(query=query, plan=empty_plan)
        # Sort candidates by highest similarity score first
        candidates.sort(key=lambda x: x[1], reverse=True)

        chosen_plan: Plan | None = None
        rationale = ""
        # 3. Try composing a plan starting from each candidate in order
        for atom_id, score in candidates:
            # Start a new plan with this atom as the root
            plan = Plan()
            plan.add_step(
                atom_id, description=f"Root atom (similarity={score:.2f})"
            )
            logger.debug(
                "Trying atom as plan root",
                atom_id=atom_id,
                similarity=f"{score:.2f}",
            )
            # Expand the plan using graph connections (up to max_steps)
            self._expand_plan(plan)
            if len(plan) == 0:
                continue  # this candidate couldn't even form a valid single-step plan
            # We got a plan (at least the root). Record rationale and select it.
            chosen_plan = plan
            rationale = (
                f"Selected '{atom_id}' as starting atom (score {score:.2f})."
            )
            if len(plan) > 1:
                rationale += f" Expanded with {len(plan)-1} further step(s) via graph links."
            else:
                rationale += " No further expansion was possible."
            logger.info(
                "Composed plan",
                plan_steps=[step.atom_id for step in plan.steps],
                rationale=rationale,
            )
            break

        if not chosen_plan:
            # No plan could be composed from any candidate
            failed_plan = Plan(
                rationale="No valid plan could be composed from candidates."
            )
            logger.warning("Planner failed to compose a plan", query=query)
            self._add_to_cache(
                query, failed_plan
            )  # cache the failure (optional)
            return PlanProposedEvent(query=query, plan=failed_plan)

        # Attach the rationale to the chosen plan
        chosen_plan.rationale = rationale
        # Cache the successful plan for future reuse
        self._add_to_cache(query, chosen_plan)
        # Emit or log the plan proposed event
        event = PlanProposedEvent(query=query, plan=chosen_plan)
        logger.info("PlanProposedEvent", **event.to_dict())
        return event

    def _expand_plan(self, plan: Plan):
        """Expand the plan in-place by adding compatible neighbor atoms."""
        while len(plan) < self.max_steps:
            current_atom = plan.steps[-1].atom_id
            neighbors = self.graph_store.get_neighbors(current_atom)
            if not neighbors:
                logger.debug(
                    "No neighbors to expand from atom", atom_id=current_atom
                )
                break
            next_atom = None
            # Pick the first neighbor that has contract compatibility
            for candidate in neighbors:
                if self.graph_store.is_contract_compatible(
                    current_atom, candidate
                ):
                    next_atom = candidate
                    break
            if not next_atom:
                logger.debug(
                    "No contract-compatible neighbor found for atom",
                    atom_id=current_atom,
                )
                break
            # Add the compatible neighbor to the plan and continue expanding
            plan.add_step(
                next_atom, description="Chained from " + current_atom
            )
            logger.debug(
                "Added atom to plan chain",
                from_atom=current_atom,
                to_atom=next_atom,
            )
            # Loop will continue until max_steps or no more expansion possible

    def _add_to_cache(self, query: str, plan: Plan):
        """Add a plan to the cache, evicting the oldest entry if needed."""
        if query in self._plan_cache:
            # Update existing plan (though usually we won't cache twice for
            # same query in one run)
            self._plan_cache[query] = plan
            self._refresh_cache_key(query)
            return
        # Evict oldest if cache is full
        if len(self._cache_order) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            self._plan_cache.pop(oldest, None)
        # Add new plan to cache and record order
        self._plan_cache[query] = plan
        self._cache_order.append(query)

    def _refresh_cache_key(self, query: str):
        """Refresh the recency order of a cache entry when it's used."""
        if query in self._cache_order:
            self._cache_order.remove(query)
            self._cache_order.append(query)
