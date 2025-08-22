"""
The Darwin-Gödel cognitive lineage tracer for Super Alita.

This module provides the GenealogyTracer, a sophisticated observer that constructs
a verifiable, exportable history of the agent's cognitive evolution. It integrates
seamlessly with our Redis-backed event bus and neural atom system.

Enhanced with real-time analysis, fitness tracking, and comprehensive export
capabilities for scientific analysis of the agent's cognitive development.
"""

import asyncio
import json
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import networkx as nx

from src.core.events import BaseEvent
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


@dataclass
class LineageNode:
    """A node in the genealogy graph."""

    key: str
    node_type: str  # "atom", "skill", "memory", "goal", "event"
    birth_event: str | None = None
    birth_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    parent_keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    fitness_scores: list[float] = field(default_factory=list)
    is_active: bool = True

    def add_fitness_score(self, score: float) -> None:
        """Add a fitness score to this node."""
        self.fitness_scores.append(score)

    def get_average_fitness(self) -> float:
        """Get average fitness score."""
        if not self.fitness_scores:
            return 0.0
        return sum(self.fitness_scores) / len(self.fitness_scores)

    def get_age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.now(UTC) - self.birth_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "node_type": self.node_type,
            "birth_event": self.birth_event,
            "birth_time": self.birth_time.isoformat(),
            "parent_keys": self.parent_keys,
            "metadata": self.metadata,
            "fitness_scores": self.fitness_scores,
            "is_active": self.is_active,
            "average_fitness": self.get_average_fitness(),
            "age_seconds": self.get_age_seconds(),
        }


@dataclass
class LineageEdge:
    """An edge in the genealogy graph."""

    parent_key: str
    child_key: str
    edge_type: str = "parent_child"  # "parent_child", "influence", "mutation", "merge"
    strength: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parent_key": self.parent_key,
            "child_key": self.child_key,
            "edge_type": self.edge_type,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class GenealogyTracer(PluginInterface):
    """
    Enhanced genealogy tracer that integrates with the event bus system.

    Tracks and manages genealogy of all cognitive primitives with real-time
    event-driven updates. Provides full Darwin-Gödel style lineage tracking
    with export capabilities for analysis and visualization.

    Features:
    - Event-driven lineage tracking via Redis EventBus
    - Real-time cognitive pattern detection
    - Multi-format export (GraphML, JSON, analysis reports)
    - Fitness-based evolutionary analysis
    - Memory-efficient sliding window for large-scale operation
    - Comprehensive statistical analysis
    """

    def __init__(self):
        super().__init__()
        self.nodes: dict[str, LineageNode] = {}
        self.edges: list[LineageEdge] = []
        self.graph: nx.DiGraph = nx.DiGraph()
        self._generation_counter: dict[str, int] = {}

        # Enhanced tracking capabilities
        self._config: dict[str, Any] = {}
        self._last_event_id: str | None = None
        self._atom_genealogy: dict[str, dict[str, Any]] = {}
        self._skill_lineages: dict[str, list[str]] = {}
        self._fitness_history: dict[str, list[tuple[datetime, float]]] = {}
        self._generation_stats: dict[int, dict[str, Any]] = {}

        # Performance optimization for large-scale operation
        self._event_buffer: deque = deque(maxlen=10000)  # Sliding window
        self._analysis_cache: dict[str, Any] = {}
        self._last_analysis_time: datetime | None = None

        # Pattern detection
        self._cognitive_patterns: dict[str, int] = defaultdict(int)
        self._lineage_depth_distribution: dict[int, int] = defaultdict(int)

        logger.info(
            "GenealogyTracer initialized with enhanced event-driven capabilities"
        )

    @property
    def name(self) -> str:
        return "genealogy_tracer"

    async def setup(self, event_bus, store, config: dict[str, Any]):
        """Setup the genealogy tracer with configuration and dependencies."""
        await super().setup(event_bus, store, config)

        self._config = config.get(self.name, {})

        # Configuration defaults
        self._config.setdefault("export_on_shutdown", True)
        self._config.setdefault("export_path", "data/genealogy")
        self._config.setdefault("analysis_interval", 300)  # 5 minutes
        self._config.setdefault("max_depth_tracking", 10)
        self._config.setdefault("enable_pattern_detection", True)
        self._config.setdefault("enable_periodic_analysis", True)

        # Create export directory
        os.makedirs(self._config["export_path"], exist_ok=True)

        # Add genesis node to root the cognitive tree
        datetime.now(UTC)
        self.add_node(
            key="genesis",
            node_type="Genesis",
            birth_event="bootstrap",
            metadata={
                "description": "The initial state of the Super Alita agent",
                "generation": 0,
                "fitness_score": 1.0,
                "lineage_depth": 0,
            },
        )

        self._last_event_id = "genesis"

        logger.info("GenealogyTracer setup complete with genesis node established")

    async def start(self):
        """Subscribe to genealogy-relevant events and start analysis tasks."""
        await super().start()

        # Subscribe to key genealogy events
        await self.subscribe("atom_birth", self._on_atom_birth)
        await self.subscribe("atom_death", self._on_atom_death)
        await self.subscribe("skill_proposal", self._on_skill_proposal)
        await self.subscribe("skill_evaluation", self._on_skill_evaluation)
        await self.subscribe("mcts_operation", self._on_mcts_operation)
        await self.subscribe("pae_cycle", self._on_pae_cycle)
        await self.subscribe("neural_activity", self._on_neural_activity)

        # Catch-all for comprehensive tracking
        await self.subscribe("*", self._on_any_event)

        # Start periodic analysis task
        if self._config["enable_periodic_analysis"]:
            self.add_task(self._periodic_analysis())

        logger.info("GenealogyTracer started - monitoring cognitive evolution")

    async def shutdown(self):
        """Export final genealogy data and cleanup."""
        logger.info("GenealogyTracer shutting down...")

        if self._config["export_on_shutdown"]:
            await self._export_complete_genealogy()

        # Generate final analysis report
        final_stats = await self._generate_analysis_report()
        self._save_analysis_report(final_stats, "final_analysis")

        await super().shutdown()
        logger.info("GenealogyTracer shutdown complete")

    # Event handlers for real-time tracking
    async def _on_atom_birth(self, event: BaseEvent):
        """Track the birth of new neural atoms."""
        try:
            data = (
                event.model_dump() if hasattr(event, "model_dump") else event.__dict__
            )

            atom_key = data.get("atom_key")
            parent_keys = data.get("parent_keys", [])
            birth_context = data.get("birth_context", "unknown")
            lineage_metadata = data.get("lineage_metadata", {})
            genealogy_depth = data.get("genealogy_depth", 0)
            darwin_godel_signature = data.get("darwin_godel_signature", "")

            if not atom_key:
                return

            # Add node to graph
            self.add_node(
                key=atom_key,
                node_type="Atom",
                birth_event=birth_context,
                parent_keys=parent_keys,
                metadata={
                    "birth_context": birth_context,
                    "lineage_metadata": lineage_metadata,
                    "depth": genealogy_depth,
                    "signature": darwin_godel_signature,
                    "timestamp": event.timestamp.isoformat(),
                },
            )

            # Record detailed genealogy
            self._atom_genealogy[atom_key] = {
                "birth_time": event.timestamp,
                "parent_keys": parent_keys,
                "birth_context": birth_context,
                "lineage_metadata": lineage_metadata,
                "depth": genealogy_depth,
                "signature": darwin_godel_signature,
                "generation": self.get_generation(atom_key),
                "fitness_score": 0.0,
            }

            # Update statistics
            generation = self.get_generation(atom_key)
            if generation not in self._generation_stats:
                self._generation_stats[generation] = {
                    "count": 0,
                    "birth_contexts": defaultdict(int),
                    "average_fitness": 0.0,
                }
            self._generation_stats[generation]["count"] += 1
            self._generation_stats[generation]["birth_contexts"][birth_context] += 1

            # Pattern detection
            if self._config["enable_pattern_detection"]:
                self._detect_cognitive_patterns(atom_key, parent_keys, birth_context)

            logger.debug(
                f"Tracked atom birth: {atom_key} (gen: {generation}, depth: {genealogy_depth})"
            )

        except Exception as e:
            logger.error(f"Error tracking atom birth: {e}", exc_info=True)

    async def _on_atom_death(self, event: BaseEvent):
        """Track the death/dissolution of neural atoms."""
        try:
            data = (
                event.model_dump() if hasattr(event, "model_dump") else event.__dict__
            )

            atom_key = data.get("atom_key")
            death_reason = data.get("death_reason", "unknown")
            contribution_score = data.get("contribution_score", 0.0)

            if atom_key and atom_key in self.nodes:
                # Update node
                self.nodes[atom_key].is_active = False
                self.nodes[atom_key].metadata.update(
                    {
                        "death_time": event.timestamp.isoformat(),
                        "death_reason": death_reason,
                        "final_contribution": contribution_score,
                    }
                )

                # Update detailed genealogy
                if atom_key in self._atom_genealogy:
                    self._atom_genealogy[atom_key].update(
                        {
                            "death_time": event.timestamp,
                            "death_reason": death_reason,
                            "final_contribution": contribution_score,
                            "lifespan": (
                                event.timestamp
                                - self._atom_genealogy[atom_key]["birth_time"]
                            ).total_seconds(),
                        }
                    )

                logger.debug(f"Tracked atom death: {atom_key} (reason: {death_reason})")

        except Exception as e:
            logger.error(f"Error tracking atom death: {e}", exc_info=True)

    async def _on_skill_proposal(self, event: BaseEvent):
        """Track skill creation and evolution."""
        try:
            data = (
                event.model_dump() if hasattr(event, "model_dump") else event.__dict__
            )

            skill_id = data.get("proposal_id") or data.get("skill_id")
            skill_name = data.get("skill_name", data.get("name", "unknown"))
            parent_skills = data.get("parent_skills", [])
            proposer = data.get("proposer", "unknown")
            confidence = data.get("confidence", 0.5)

            if skill_id:
                # Track skill lineage
                self._skill_lineages[skill_id] = parent_skills

                # Add to graph
                self.add_node(
                    key=skill_id,
                    node_type="Skill",
                    birth_event="skill_proposal",
                    parent_keys=parent_skills,
                    metadata={
                        "skill_name": skill_name,
                        "proposer": proposer,
                        "confidence": confidence,
                        "timestamp": event.timestamp.isoformat(),
                    },
                )

                logger.debug(
                    f"Tracked skill proposal: {skill_name} (parents: {len(parent_skills)})"
                )

        except Exception as e:
            logger.error(f"Error tracking skill proposal: {e}", exc_info=True)

    async def _on_skill_evaluation(self, event: BaseEvent):
        """Track skill fitness and evaluation results."""
        try:
            data = (
                event.model_dump() if hasattr(event, "model_dump") else event.__dict__
            )

            skill_id = data.get("skill_id")
            performance_score = data.get("performance_score", 0.0)
            success = data.get("success", False)

            if skill_id:
                # Record fitness history
                if skill_id not in self._fitness_history:
                    self._fitness_history[skill_id] = []

                self._fitness_history[skill_id].append(
                    (event.timestamp, performance_score)
                )

                # Update node fitness
                if skill_id in self.nodes:
                    self.update_fitness(skill_id, performance_score)
                    self.nodes[skill_id].metadata.update(
                        {
                            "latest_fitness": performance_score,
                            "success_status": success,
                            "evaluation_count": len(self._fitness_history[skill_id]),
                        }
                    )

                logger.debug(
                    f"Tracked skill evaluation: {skill_id} (fitness: {performance_score})"
                )

        except Exception as e:
            logger.error(f"Error tracking skill evaluation: {e}", exc_info=True)

    async def _on_mcts_operation(self, event: BaseEvent):
        """Track MCTS tree operations for skill evolution."""
        try:
            data = (
                event.model_dump() if hasattr(event, "model_dump") else event.__dict__
            )

            node_id = data.get("node_id")
            operation = data.get("operation")
            value = data.get("value", 0.0)
            visit_count = data.get("visit_count", 0)
            depth = data.get("depth", 0)

            if node_id and operation:
                mcts_node_key = f"mcts_{node_id}_{operation}"
                self.add_node(
                    key=mcts_node_key,
                    node_type="MCTS_Operation",
                    birth_event="mcts_operation",
                    metadata={
                        "operation": operation,
                        "value": value,
                        "visit_count": visit_count,
                        "depth": depth,
                        "timestamp": event.timestamp.isoformat(),
                    },
                )

        except Exception as e:
            logger.error(f"Error tracking MCTS operation: {e}", exc_info=True)

    async def _on_pae_cycle(self, event: BaseEvent):
        """Track Perceive-Act-Evolve cycles."""
        try:
            data = (
                event.model_dump() if hasattr(event, "model_dump") else event.__dict__
            )

            cycle_id = data.get("cycle_id")
            phase = data.get("phase")
            fitness_score = data.get("fitness_score", 0.0)

            if cycle_id and phase:
                pae_node_key = f"pae_{cycle_id}_{phase}"

                # Find parent phase for linking
                parent_key = None
                if phase == "act":
                    parent_key = f"pae_{cycle_id}_perceive"
                elif phase == "evolve":
                    parent_key = f"pae_{cycle_id}_act"

                self.add_node(
                    key=pae_node_key,
                    node_type="PAE_Cycle",
                    birth_event="pae_cycle",
                    parent_keys=(
                        [parent_key] if parent_key and parent_key in self.nodes else []
                    ),
                    metadata={
                        "cycle_id": cycle_id,
                        "phase": phase,
                        "fitness_score": fitness_score,
                        "timestamp": event.timestamp.isoformat(),
                    },
                )

        except Exception as e:
            logger.error(f"Error tracking PAE cycle: {e}", exc_info=True)

    async def _on_neural_activity(self, event: BaseEvent):
        """Track neural activity patterns."""
        try:
            data = (
                event.model_dump() if hasattr(event, "model_dump") else event.__dict__
            )

            activity_pattern = data.get("activity_pattern", [])
            attention_focus = data.get("attention_focus", [])

            if activity_pattern or attention_focus:
                activity_node_key = f"neural_activity_{event.event_id}"
                self.add_node(
                    key=activity_node_key,
                    node_type="Neural_Activity",
                    birth_event="neural_activity",
                    metadata={
                        "pattern_size": len(activity_pattern),
                        "attention_targets": (
                            len(attention_focus) if attention_focus else 0
                        ),
                        "timestamp": event.timestamp.isoformat(),
                    },
                )

        except Exception as e:
            logger.error(f"Error tracking neural activity: {e}", exc_info=True)

    async def _on_any_event(self, event: BaseEvent):
        """Catch-all handler for comprehensive event tracking."""
        try:
            # Add to event buffer for analysis
            self._event_buffer.append(
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "source_plugin": event.source_plugin,
                    "timestamp": event.timestamp,
                }
            )

        except Exception as e:
            logger.debug(f"Minor error in catch-all event handler: {e}")

    def _detect_cognitive_patterns(
        self, atom_key: str, parent_keys: list[str], birth_context: str
    ):
        """Detect recurring cognitive patterns in atom creation."""
        # Pattern: Single-parent inheritance
        if len(parent_keys) == 1:
            self._cognitive_patterns["single_inheritance"] += 1

        # Pattern: Multi-parent combination
        elif len(parent_keys) > 1:
            self._cognitive_patterns["multi_inheritance"] += 1
            if len(parent_keys) > 3:
                self._cognitive_patterns["complex_combination"] += 1

        # Pattern: Spontaneous generation
        elif len(parent_keys) == 0:
            self._cognitive_patterns["spontaneous_generation"] += 1

        # Context-based patterns
        context_patterns = {
            "skill_creation": "skill_discovery_pattern",
            "memory_formation": "learning_pattern",
            "goal_derivation": "planning_pattern",
        }

        for context_key, pattern_name in context_patterns.items():
            if context_key in birth_context:
                self._cognitive_patterns[pattern_name] += 1

    async def _periodic_analysis(self):
        """Perform periodic analysis of cognitive evolution."""
        while self.is_running:
            try:
                await asyncio.sleep(self._config["analysis_interval"])

                if not self.is_running:
                    break

                # Generate analysis report
                analysis_data = await self._generate_analysis_report()

                # Save periodic analysis
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                self._save_analysis_report(analysis_data, f"periodic_{timestamp}")

                # Clear old cache
                self._analysis_cache.clear()
                self._last_analysis_time = datetime.now(UTC)

                logger.info(
                    f"Periodic genealogy analysis completed: {len(self.nodes)} nodes, {len(self.edges)} edges"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic analysis: {e}", exc_info=True)

    async def _generate_analysis_report(self) -> dict[str, Any]:
        """Generate comprehensive analysis of cognitive evolution."""
        try:
            analysis = {
                "timestamp": datetime.now(UTC).isoformat(),
                "graph_statistics": {
                    "total_nodes": len(self.nodes),
                    "total_edges": len(self.edges),
                    "active_nodes": sum(
                        1 for node in self.nodes.values() if node.is_active
                    ),
                    "node_types": self._get_node_type_distribution(),
                    "max_generation": (
                        max(self._generation_counter.values())
                        if self._generation_counter
                        else 0
                    ),
                },
                "genealogy_statistics": {
                    "total_atoms": len(self._atom_genealogy),
                    "generation_distribution": dict(self._generation_stats),
                    "average_generation": self._calculate_average_generation(),
                    "cognitive_patterns": dict(self._cognitive_patterns),
                },
                "fitness_analysis": self._analyze_fitness_trends(),
                "skill_evolution": {
                    "total_skills": len(self._skill_lineages),
                    "evolution_chains": self._analyze_skill_evolution_chains(),
                },
                "temporal_analysis": {
                    "event_rate": len(self._event_buffer)
                    / max(1, self._config["analysis_interval"]),
                    "recent_activity": self._analyze_recent_activity(),
                },
            }

            return analysis

        except Exception as e:
            logger.error(f"Error generating analysis report: {e}", exc_info=True)
            return {"error": str(e)}

    def _get_node_type_distribution(self) -> dict[str, int]:
        """Get distribution of node types in the graph."""
        type_counts = defaultdict(int)
        for node in self.nodes.values():
            type_counts[node.node_type] += 1
        return dict(type_counts)

    def _calculate_average_generation(self) -> float:
        """Calculate average generation across all atoms."""
        if not self._atom_genealogy:
            return 0.0

        total_generations = sum(
            atom_data.get("generation", 0)
            for atom_data in self._atom_genealogy.values()
        )
        return total_generations / len(self._atom_genealogy)

    def _analyze_fitness_trends(self) -> dict[str, Any]:
        """Analyze fitness trends across all tracked entities."""
        if not self._fitness_history:
            return {"total_entities": 0}

        all_fitness_scores = []
        for entity_history in self._fitness_history.values():
            all_fitness_scores.extend([score for _, score in entity_history])

        if not all_fitness_scores:
            return {"total_entities": len(self._fitness_history)}

        try:
            import statistics

            return {
                "total_entities": len(self._fitness_history),
                "average_fitness": statistics.mean(all_fitness_scores),
                "fitness_std": (
                    statistics.stdev(all_fitness_scores)
                    if len(all_fitness_scores) > 1
                    else 0
                ),
                "max_fitness": max(all_fitness_scores),
                "min_fitness": min(all_fitness_scores),
                "improving_entities": self._count_improving_entities(),
            }
        except ImportError:
            # Fallback without statistics module
            return {
                "total_entities": len(self._fitness_history),
                "average_fitness": sum(all_fitness_scores) / len(all_fitness_scores),
                "max_fitness": max(all_fitness_scores),
                "min_fitness": min(all_fitness_scores),
            }

    def _count_improving_entities(self) -> int:
        """Count entities with improving fitness trends."""
        improving_count = 0
        for entity_history in self._fitness_history.values():
            if len(entity_history) >= 2:
                first_score = entity_history[0][1]
                last_score = entity_history[-1][1]
                if last_score > first_score:
                    improving_count += 1
        return improving_count

    def _analyze_skill_evolution_chains(self) -> dict[str, Any]:
        """Analyze skill evolution chains and patterns."""
        chains = []
        max_chain_length = 0

        for skill_id, parents in self._skill_lineages.items():
            chain_length = self._calculate_skill_chain_length(skill_id, set())
            max_chain_length = max(max_chain_length, chain_length)

            if chain_length > 2:  # Interesting chains
                chains.append(
                    {
                        "skill_id": skill_id,
                        "chain_length": chain_length,
                        "parents": parents,
                    }
                )

        return {
            "total_chains": len(chains),
            "max_chain_length": max_chain_length,
            "complex_chains": [c for c in chains if c["chain_length"] > 3],
        }

    def _calculate_skill_chain_length(self, skill_id: str, visited: set[str]) -> int:
        """Calculate the length of a skill evolution chain."""
        if skill_id in visited or skill_id not in self._skill_lineages:
            return 0

        visited.add(skill_id)
        parents = self._skill_lineages[skill_id]

        if not parents:
            return 1

        max_parent_length = max(
            self._calculate_skill_chain_length(parent, visited.copy())
            for parent in parents
        )

        return max_parent_length + 1

    def _analyze_recent_activity(self) -> dict[str, Any]:
        """Analyze recent activity patterns."""
        if not self._event_buffer:
            return {"recent_events": 0}

        recent_cutoff = datetime.now(UTC) - timedelta(minutes=10)
        recent_events = [
            event for event in self._event_buffer if event["timestamp"] > recent_cutoff
        ]

        event_types = defaultdict(int)
        source_plugins = defaultdict(int)

        for event in recent_events:
            event_types[event["event_type"]] += 1
            source_plugins[event["source_plugin"]] += 1

        return {
            "recent_events": len(recent_events),
            "event_types": dict(event_types),
            "active_plugins": dict(source_plugins),
        }

    def _save_analysis_report(
        self, analysis_data: dict[str, Any], filename_prefix: str
    ):
        """Save analysis report to file."""
        try:
            filename = f"{filename_prefix}_analysis.json"
            filepath = os.path.join(self._config["export_path"], filename)

            with open(filepath, "w") as f:
                json.dump(analysis_data, f, indent=2, default=str)

            logger.debug(f"Analysis report saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving analysis report: {e}", exc_info=True)

    async def _export_complete_genealogy(self):
        """Export complete genealogy in multiple formats."""
        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

            # Export GraphML for visualization
            self.export_to_graphml(
                os.path.join(
                    self._config["export_path"], f"genealogy_{timestamp}.graphml"
                )
            )

            # Export JSON for programmatic analysis
            self.export_to_json(
                os.path.join(self._config["export_path"], f"genealogy_{timestamp}.json")
            )

            # Export analysis summary
            analysis_data = await self._generate_analysis_report()
            self._save_analysis_report(analysis_data, f"final_{timestamp}")

            logger.info(f"Complete genealogy exported with timestamp {timestamp}")

        except Exception as e:
            logger.error(f"Error exporting complete genealogy: {e}", exc_info=True)

    def get_genealogy_summary(self) -> dict[str, Any]:
        """Get a quick summary of the current genealogy state."""
        return {
            "total_atoms": len(self._atom_genealogy),
            "total_skills": len(self._skill_lineages),
            "graph_nodes": len(self.nodes),
            "graph_edges": len(self.edges),
            "generations": len(self._generation_stats),
            "cognitive_patterns": len(self._cognitive_patterns),
            "recent_events": len(self._event_buffer),
            "active_nodes": sum(1 for node in self.nodes.values() if node.is_active),
        }

    def add_node(
        self,
        key: str,
        node_type: str,
        birth_event: str | None = None,
        parent_keys: list[str] = None,
        metadata: dict[str, Any] = None,
    ) -> LineageNode:
        """Add a new node to the genealogy."""

        node = LineageNode(
            key=key,
            node_type=node_type,
            birth_event=birth_event,
            parent_keys=parent_keys or [],
            metadata=metadata or {},
        )

        self.nodes[key] = node
        self.graph.add_node(key, **node.to_dict())

        # Add edges to parents
        for parent_key in node.parent_keys:
            self.add_edge(parent_key, key, "parent_child")

        # Track generation
        if parent_keys:
            max_parent_gen = max(
                self._generation_counter.get(pk, 0) for pk in parent_keys
            )
            self._generation_counter[key] = max_parent_gen + 1
        else:
            self._generation_counter[key] = 0

        return node

    def add_edge(
        self,
        parent_key: str,
        child_key: str,
        edge_type: str = "parent_child",
        strength: float = 1.0,
        metadata: dict[str, Any] = None,
    ) -> LineageEdge:
        """Add an edge between nodes."""

        edge = LineageEdge(
            parent_key=parent_key,
            child_key=child_key,
            edge_type=edge_type,
            strength=strength,
            metadata=metadata or {},
        )

        self.edges.append(edge)
        self.graph.add_edge(parent_key, child_key, **edge.to_dict())

        return edge

    def update_fitness(self, key: str, fitness_score: float) -> None:
        """Update fitness score for a node."""
        if key in self.nodes:
            self.nodes[key].add_fitness_score(fitness_score)
            # Update graph node attributes
            self.graph.nodes[key]["fitness_scores"] = self.nodes[key].fitness_scores
            self.graph.nodes[key]["average_fitness"] = self.nodes[
                key
            ].get_average_fitness()

    def deactivate_node(self, key: str) -> None:
        """Mark a node as inactive (pruned)."""
        if key in self.nodes:
            self.nodes[key].is_active = False
            self.graph.nodes[key]["is_active"] = False

    def get_ancestors(self, key: str, max_depth: int | None = None) -> set[str]:
        """Get all ancestors of a node."""
        ancestors = set()
        to_visit = [(key, 0)]

        while to_visit:
            current_key, depth = to_visit.pop()

            if max_depth is not None and depth >= max_depth:
                continue

            if current_key in self.nodes:
                for parent_key in self.nodes[current_key].parent_keys:
                    if parent_key not in ancestors:
                        ancestors.add(parent_key)
                        to_visit.append((parent_key, depth + 1))

        return ancestors

    def get_descendants(self, key: str, max_depth: int | None = None) -> set[str]:
        """Get all descendants of a node."""
        descendants = set()
        to_visit = [(key, 0)]

        while to_visit:
            current_key, depth = to_visit.pop()

            if max_depth is not None and depth >= max_depth:
                continue

            # Find children
            children = [
                edge.child_key for edge in self.edges if edge.parent_key == current_key
            ]

            for child_key in children:
                if child_key not in descendants:
                    descendants.add(child_key)
                    to_visit.append((child_key, depth + 1))

        return descendants

    def get_generation(self, key: str) -> int:
        """Get generation number of a node."""
        return self._generation_counter.get(key, 0)

    def get_lineage_path(self, key: str) -> list[str]:
        """Get the primary lineage path to root for a node."""
        path = [key]
        current = key

        while current in self.nodes and self.nodes[current].parent_keys:
            # Choose the parent with highest fitness as primary lineage
            parents = self.nodes[current].parent_keys
            if len(parents) == 1:
                current = parents[0]
            else:
                # Select parent with highest average fitness
                best_parent = max(
                    parents,
                    key=lambda p: (
                        self.nodes[p].get_average_fitness() if p in self.nodes else 0
                    ),
                )
                current = best_parent

            path.append(current)

        return path

    def analyze_fitness_evolution(self, node_type: str | None = None) -> dict[str, Any]:
        """Analyze fitness evolution over generations."""

        nodes_to_analyze = [
            node
            for node in self.nodes.values()
            if node_type is None or node.node_type == node_type
        ]

        # Group by generation
        generation_fitness: dict[int, list[float]] = {}
        for node in nodes_to_analyze:
            gen = self.get_generation(node.key)
            if gen not in generation_fitness:
                generation_fitness[gen] = []

            avg_fitness = node.get_average_fitness()
            if avg_fitness > 0:  # Only include nodes with fitness scores
                generation_fitness[gen].append(avg_fitness)

        # Calculate statistics per generation
        generation_stats = {}
        for gen, fitnesses in generation_fitness.items():
            if fitnesses:
                generation_stats[gen] = {
                    "count": len(fitnesses),
                    "mean_fitness": sum(fitnesses) / len(fitnesses),
                    "max_fitness": max(fitnesses),
                    "min_fitness": min(fitnesses),
                }

        return {
            "total_generations": (
                max(generation_fitness.keys()) + 1 if generation_fitness else 0
            ),
            "generation_stats": generation_stats,
            "total_nodes_analyzed": len(nodes_to_analyze),
        }

    def find_evolutionary_branches(self) -> list[dict[str, Any]]:
        """Find major evolutionary branches in the genealogy."""

        branches = []

        # Find nodes with multiple children (branch points)
        for node_key, node in self.nodes.items():
            children = [
                edge.child_key for edge in self.edges if edge.parent_key == node_key
            ]

            if len(children) > 1:
                branch_info = {
                    "branch_point": node_key,
                    "generation": self.get_generation(node_key),
                    "children": children,
                    "child_fitness": {
                        child: self.nodes[child].get_average_fitness()
                        for child in children
                        if child in self.nodes
                    },
                }
                branches.append(branch_info)

        # Sort by generation
        branches.sort(key=lambda x: x["generation"])
        return branches

    def export_to_graphml(self, filepath: str) -> None:
        """Export genealogy to GraphML format for visualization."""

        # Create a clean graph for export
        export_graph = nx.DiGraph()

        # Add nodes with all attributes
        for key, node in self.nodes.items():
            node_attrs = node.to_dict()
            # Convert datetime and other non-serializable types
            for attr_key, attr_value in node_attrs.items():
                if isinstance(attr_value, list | dict):
                    node_attrs[attr_key] = json.dumps(attr_value)
                elif not isinstance(attr_value, str | int | float | bool):
                    node_attrs[attr_key] = str(attr_value)

            export_graph.add_node(key, **node_attrs)

        # Add edges with attributes
        for edge in self.edges:
            edge_attrs = edge.to_dict()
            # Convert non-serializable types
            for attr_key, attr_value in edge_attrs.items():
                if isinstance(attr_value, list | dict):
                    edge_attrs[attr_key] = json.dumps(attr_value)
                elif not isinstance(attr_value, str | int | float | bool):
                    edge_attrs[attr_key] = str(attr_value)

            export_graph.add_edge(edge.parent_key, edge.child_key, **edge_attrs)

        # Write to GraphML
        nx.write_graphml(export_graph, filepath)

    def export_to_json(self, filepath: str) -> None:
        """Export genealogy to JSON format."""

        export_data = {
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "generation_counter": self._generation_counter,
            "export_timestamp": datetime.now(UTC).isoformat(),
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

    def import_from_json(self, filepath: str) -> None:
        """Import genealogy from JSON format."""

        with open(filepath) as f:
            data = json.load(f)

        # Clear existing data
        self.nodes.clear()
        self.edges.clear()
        self.graph.clear()

        # Import nodes
        for key, node_data in data["nodes"].items():
            node_data["birth_time"] = datetime.fromisoformat(node_data["birth_time"])
            node = LineageNode(**node_data)
            self.nodes[key] = node
            self.graph.add_node(key, **node.to_dict())

        # Import edges
        for edge_data in data["edges"]:
            edge_data["created_at"] = datetime.fromisoformat(edge_data["created_at"])
            edge = LineageEdge(**edge_data)
            self.edges.append(edge)
            self.graph.add_edge(edge.parent_key, edge.child_key, **edge.to_dict())

        # Import generation counter
        self._generation_counter = data.get("generation_counter", {})

    def get_statistics(self) -> dict[str, Any]:
        """Get genealogy statistics."""

        active_nodes = sum(1 for node in self.nodes.values() if node.is_active)
        node_types = {}
        fitness_stats = []

        for node in self.nodes.values():
            # Count node types
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

            # Collect fitness scores
            if node.fitness_scores:
                fitness_stats.extend(node.fitness_scores)

        return {
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "inactive_nodes": len(self.nodes) - active_nodes,
            "total_edges": len(self.edges),
            "node_types": node_types,
            "max_generation": (
                max(self._generation_counter.values())
                if self._generation_counter
                else 0
            ),
            "fitness_stats": {
                "total_scores": len(fitness_stats),
                "mean_fitness": (
                    sum(fitness_stats) / len(fitness_stats) if fitness_stats else 0
                ),
                "max_fitness": max(fitness_stats) if fitness_stats else 0,
                "min_fitness": min(fitness_stats) if fitness_stats else 0,
            },
        }

    def prune_lineage(
        self, fitness_threshold: float = 0.1, age_threshold_hours: float = 168
    ) -> list[str]:
        """
        Prune lineage based on fitness and age thresholds.

        Args:
            fitness_threshold: Minimum fitness to keep
            age_threshold_hours: Maximum age in hours to keep low-fitness nodes

        Returns:
            List of pruned node keys
        """

        pruned_keys = []
        current_time = datetime.now(UTC)

        for key, node in list(self.nodes.items()):
            should_prune = False

            # Check fitness
            avg_fitness = node.get_average_fitness()
            if avg_fitness < fitness_threshold:
                # Check age
                age_hours = (current_time - node.birth_time).total_seconds() / 3600
                if age_hours > age_threshold_hours:
                    should_prune = True

            if should_prune:
                self.deactivate_node(key)
                pruned_keys.append(key)

        return pruned_keys


# Global genealogy tracer instance
_global_tracer: GenealogyTracer | None = None


def get_global_tracer() -> GenealogyTracer | None:
    """Get the global genealogy tracer instance."""
    return _global_tracer


def set_global_tracer(tracer: GenealogyTracer):
    """Set the global genealogy tracer instance."""
    global _global_tracer
    _global_tracer = tracer


def trace_birth(
    key: str,
    node_type: str,
    birth_event: str | None = None,
    parent_keys: list[str] = None,
    metadata: dict[str, Any] = None,
) -> LineageNode | None:
    """Convenience function to trace birth of a new entity."""

    tracer = get_global_tracer()
    if tracer:
        return tracer.add_node(
            key=key,
            node_type=node_type,
            birth_event=birth_event,
            parent_keys=parent_keys,
            metadata=metadata,
        )
    return None


def trace_fitness(key: str, fitness_score: float) -> None:
    """Convenience function to trace fitness score."""

    tracer = get_global_tracer()
    if tracer:
        tracer.update_fitness(key, fitness_score)


# Utility functions for genealogy analysis
def trace_atom_birth(
    atom_key: str,
    parent_keys: list[str],
    birth_context: str,
    lineage_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Utility function to create standardized atom birth trace data."""
    return {
        "atom_key": atom_key,
        "parent_keys": parent_keys,
        "birth_context": birth_context,
        "lineage_metadata": lineage_metadata or {},
        "timestamp": datetime.now(UTC).isoformat(),
    }


def trace_skill_fitness(
    entity_id: str, fitness_score: float, evaluation_context: str | None = None
) -> dict[str, Any]:
    """Utility function to create standardized fitness trace data."""
    return {
        "entity_id": entity_id,
        "fitness_score": fitness_score,
        "evaluation_context": evaluation_context,
        "timestamp": datetime.now(UTC).isoformat(),
    }
