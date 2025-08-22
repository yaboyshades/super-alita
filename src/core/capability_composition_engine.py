#!/usr/bin/env python3
"""
🔗 CAPABILITY COMPOSITION ENGINE - PURE "HOW TO DO" ARCHITECTURE
Advanced capability composition and execution pattern discovery

AGENT DEV MODE (Copilot read this):
- Pure procedural encoding: NO task names, only execution patterns
- Capability graphs encode HOW abilities compose, not WHAT tasks to do
- Dynamic pattern discovery learns HOW to solve novel problems
- Meta-execution strategies evolve HOW the agent thinks
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from uuid import uuid4

import networkx as nx

logger = logging.getLogger(__name__)


class ExecutionPattern(Enum):
    """Pure execution patterns - HOW capabilities work, not WHAT they do"""

    SEQUENTIAL = auto()  # HOW: Execute one after another
    PARALLEL = auto()  # HOW: Execute simultaneously
    CONDITIONAL = auto()  # HOW: Execute based on conditions
    ITERATIVE = auto()  # HOW: Execute repeatedly until condition
    RECURSIVE = auto()  # HOW: Execute self-referentially
    ADAPTIVE = auto()  # HOW: Modify execution based on feedback
    COMPOSITIONAL = auto()  # HOW: Combine multiple patterns


@dataclass
class ExecutionNode:
    """A node in the capability graph representing HOW to execute"""

    node_id: str
    execution_signature: str  # HOW this capability executes
    input_patterns: list[str]  # HOW input is structured
    output_patterns: list[str]  # HOW output is structured
    success_conditions: dict[str, Any]  # HOW to measure success
    failure_patterns: list[str]  # HOW failures manifest
    performance_metrics: dict[str, float] = field(default_factory=dict)
    adaptation_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecutionEdge:
    """An edge representing HOW capabilities connect"""

    source_id: str
    target_id: str
    composition_pattern: ExecutionPattern
    data_flow_pattern: str  # HOW data flows between capabilities
    success_rate: float = 0.0
    avg_latency: float = 0.0
    conditions: dict[str, Any] = field(default_factory=dict)


class CapabilityCompositionEngine:
    """
    🔗 Pure "HOW TO DO" capability composition engine

    Key Principles:
    - NO task-specific knowledge: only execution patterns
    - Learns HOW capabilities compose effectively
    - Discovers HOW to solve novel problems
    - Evolves HOW the agent approaches challenges
    """

    def __init__(self):
        self.capability_graph = nx.DiGraph()
        self.execution_patterns: dict[str, ExecutionPattern] = {}
        self.composition_history: list[dict[str, Any]] = []
        self.meta_patterns: dict[str, dict[str, Any]] = {}

        # Learning parameters
        self.pattern_discovery_threshold = 0.7
        self.composition_success_threshold = 0.8
        self.adaptation_rate = 0.1

        logger.info("🔗 CapabilityCompositionEngine initialized")

    async def discover_execution_path(
        self,
        input_signature: str,
        output_signature: str,
        constraints: dict[str, Any] | None = None,
    ) -> list[ExecutionNode]:
        """
        Discover HOW to achieve output from input using capability composition
        Pure procedural: no task knowledge, only execution pattern matching
        """
        constraints = constraints or {}

        # Find nodes that can handle the input signature
        capable_start_nodes = [
            node_id
            for node_id, node_data in self.capability_graph.nodes(data=True)
            if self._can_handle_input_pattern(node_data["node"], input_signature)
        ]

        if not capable_start_nodes:
            logger.warning(
                f"🔗 No capabilities found for input pattern: {input_signature}"
            )
            return []

        # Find nodes that can produce the output signature
        capable_end_nodes = [
            node_id
            for node_id, node_data in self.capability_graph.nodes(data=True)
            if self._can_produce_output_pattern(node_data["node"], output_signature)
        ]

        if not capable_end_nodes:
            logger.warning(
                f"🔗 No capabilities found for output pattern: {output_signature}"
            )
            return []

        # Find optimal composition paths
        best_paths = []

        for start_node in capable_start_nodes:
            for end_node in capable_end_nodes:
                try:
                    paths = list(
                        nx.all_simple_paths(
                            self.capability_graph,
                            start_node,
                            end_node,
                            cutoff=constraints.get("max_path_length", 5),
                        )
                    )

                    for path in paths:
                        path_score = await self._evaluate_execution_path(
                            path, constraints
                        )
                        best_paths.append((path, path_score))

                except nx.NetworkXNoPath:
                    continue

        if not best_paths:
            # Attempt to discover new composition patterns
            return await self._discover_novel_composition(
                input_signature, output_signature, constraints
            )

        # Return best path as ExecutionNodes
        best_path, _ = max(best_paths, key=lambda x: x[1])
        execution_nodes = [
            self.capability_graph.nodes[node_id]["node"] for node_id in best_path
        ]

        logger.info(f"🔗 Discovered execution path: {len(execution_nodes)} nodes")
        return execution_nodes

    async def learn_composition_pattern(
        self,
        execution_trace: list[dict[str, Any]],
        success: bool,
        performance_metrics: dict[str, float],
    ) -> None:
        """
        Learn HOW capability compositions work from execution traces
        Pure meta-learning: improves HOW patterns are discovered and composed
        """
        if len(execution_trace) < 2:
            return

        # Extract composition pattern
        pattern_signature = self._extract_pattern_signature(execution_trace)

        # Update composition history
        composition_record = {
            "pattern_signature": pattern_signature,
            "execution_trace": execution_trace,
            "success": success,
            "performance_metrics": performance_metrics,
            "timestamp": logger.name,  # Use as timestamp placeholder
        }
        self.composition_history.append(composition_record)

        # Learn from successful patterns
        if (
            success
            and performance_metrics.get("efficiency", 0)
            > self.composition_success_threshold
        ):
            await self._reinforce_pattern(pattern_signature, performance_metrics)

        # Learn from failures
        if not success:
            await self._analyze_failure_pattern(pattern_signature, execution_trace)

        # Discover meta-patterns
        if len(self.composition_history) % 10 == 0:  # Every 10 compositions
            await self._discover_meta_patterns()

        logger.debug(
            f"🔗 Learned from composition: {pattern_signature} (success: {success})"
        )

    async def evolve_execution_strategy(
        self, current_strategy: dict[str, Any], feedback: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evolve HOW the agent approaches execution based on feedback
        Meta-level adaptation: changes HOW strategies are formed
        """
        evolved_strategy = current_strategy.copy()

        # Analyze feedback patterns
        success_rate = feedback.get("success_rate", 0.0)
        efficiency = feedback.get("efficiency", 0.0)
        adaptability = feedback.get("adaptability", 0.0)

        # Evolve based on performance
        if success_rate < 0.6:
            # Poor success rate: increase conservatism
            evolved_strategy["risk_tolerance"] = max(
                0.1, evolved_strategy.get("risk_tolerance", 0.5) - self.adaptation_rate
            )
            evolved_strategy["exploration_rate"] = max(
                0.1,
                evolved_strategy.get("exploration_rate", 0.3) - self.adaptation_rate,
            )

        elif success_rate > 0.8 and efficiency < 0.6:
            # Good success but poor efficiency: increase optimization
            evolved_strategy["optimization_focus"] = min(
                1.0,
                evolved_strategy.get("optimization_focus", 0.5) + self.adaptation_rate,
            )

        elif success_rate > 0.8 and efficiency > 0.7:
            # High performance: increase exploration
            evolved_strategy["exploration_rate"] = min(
                0.9,
                evolved_strategy.get("exploration_rate", 0.3) + self.adaptation_rate,
            )

        # Adapt composition preferences
        if adaptability < 0.5:
            # Poor adaptability: favor simpler patterns
            evolved_strategy["complexity_preference"] = max(
                0.1,
                evolved_strategy.get("complexity_preference", 0.5)
                - self.adaptation_rate,
            )

        logger.info(f"🔗 Evolved execution strategy: {evolved_strategy}")
        return evolved_strategy

    def register_capability(
        self,
        execution_node: ExecutionNode,
        connections: list[ExecutionEdge] | None = None,
    ) -> None:
        """
        Register a new capability in the composition graph
        Pure capability registration: only HOW patterns, no task specifics
        """
        connections = connections or []

        # Add node to graph
        self.capability_graph.add_node(execution_node.node_id, node=execution_node)

        # Add edges for connections
        for edge in connections:
            if (
                edge.source_id in self.capability_graph
                and edge.target_id in self.capability_graph
            ):
                self.capability_graph.add_edge(
                    edge.source_id, edge.target_id, edge_data=edge
                )

        logger.info(f"🔗 Registered capability: {execution_node.execution_signature}")

    def _can_handle_input_pattern(
        self, node: ExecutionNode, input_signature: str
    ) -> bool:
        """Check if node can handle input pattern"""
        return any(
            self._pattern_matches(pattern, input_signature)
            for pattern in node.input_patterns
        )

    def _can_produce_output_pattern(
        self, node: ExecutionNode, output_signature: str
    ) -> bool:
        """Check if node can produce output pattern"""
        return any(
            self._pattern_matches(pattern, output_signature)
            for pattern in node.output_patterns
        )

    def _pattern_matches(self, pattern: str, signature: str) -> bool:
        """Simple pattern matching - can be enhanced with regex/ML"""
        return (
            pattern.lower() in signature.lower() or signature.lower() in pattern.lower()
        )

    async def _evaluate_execution_path(
        self, path: list[str], constraints: dict[str, Any]
    ) -> float:
        """Evaluate how good an execution path is based on learned patterns"""
        if len(path) < 2:
            return 0.5

        score = 1.0

        # Penalize long paths unless efficiency is proven
        if len(path) > 3:
            score *= 0.9 ** (len(path) - 3)

        # Reward paths with good historical performance
        for i in range(len(path) - 1):
            edge_key = f"{path[i]}->{path[i + 1]}"
            if edge_key in self.meta_patterns:
                pattern_data = self.meta_patterns[edge_key]
                score *= pattern_data.get("success_rate", 0.5)

        return score

    async def _discover_novel_composition(
        self, input_signature: str, output_signature: str, constraints: dict[str, Any]
    ) -> list[ExecutionNode]:
        """
        Discover novel ways to compose capabilities for new problems
        Meta-discovery: HOW to find new HOWs
        """
        # This is where true AI creativity happens
        # For now, return empty - can be enhanced with ML/reasoning
        logger.info(
            f"🔗 Attempting novel composition discovery: {input_signature} -> {output_signature}"
        )
        return []

    def _extract_pattern_signature(self, execution_trace: list[dict[str, Any]]) -> str:
        """Extract a signature representing the execution pattern"""
        if not execution_trace:
            return "empty_pattern"

        # Create signature from execution characteristics
        pattern_elements = []

        for step in execution_trace:
            step_type = step.get("type", "unknown")
            duration = step.get("duration", 0)
            success = step.get("success", False)

            # Encode HOW this step executed
            if duration < 0.1:
                speed = "fast"
            elif duration < 1.0:
                speed = "medium"
            else:
                speed = "slow"

            element = f"{step_type}_{speed}_{'success' if success else 'failure'}"
            pattern_elements.append(element)

        return "->".join(pattern_elements)

    async def _reinforce_pattern(
        self, pattern_signature: str, performance_metrics: dict[str, float]
    ) -> None:
        """Reinforce successful execution patterns"""
        if pattern_signature not in self.meta_patterns:
            self.meta_patterns[pattern_signature] = {
                "success_rate": 0.0,
                "avg_efficiency": 0.0,
                "usage_count": 0,
            }

        pattern = self.meta_patterns[pattern_signature]
        pattern["usage_count"] += 1
        pattern["success_rate"] = (
            pattern["success_rate"] * (pattern["usage_count"] - 1) + 1.0
        ) / pattern["usage_count"]

        efficiency = performance_metrics.get("efficiency", 0.5)
        pattern["avg_efficiency"] = (
            pattern["avg_efficiency"] * (pattern["usage_count"] - 1) + efficiency
        ) / pattern["usage_count"]

    async def _analyze_failure_pattern(
        self, pattern_signature: str, execution_trace: list[dict[str, Any]]
    ) -> None:
        """Analyze why execution patterns fail"""
        # Learn what causes failures in execution patterns
        failure_key = f"failure_{pattern_signature}"

        if failure_key not in self.meta_patterns:
            self.meta_patterns[failure_key] = {
                "failure_rate": 0.0,
                "common_causes": [],
                "usage_count": 0,
            }

        pattern = self.meta_patterns[failure_key]
        pattern["usage_count"] += 1
        pattern["failure_rate"] = (
            pattern["failure_rate"] * (pattern["usage_count"] - 1) + 1.0
        ) / pattern["usage_count"]

        # Extract failure causes
        for step in execution_trace:
            if not step.get("success", True):
                error_type = step.get("error_type", "unknown")
                if error_type not in pattern["common_causes"]:
                    pattern["common_causes"].append(error_type)

    async def _discover_meta_patterns(self) -> None:
        """
        Discover meta-patterns: HOW successful patterns relate to each other
        This is where the agent learns HOW to learn better
        """
        # Analyze composition history for higher-order patterns
        successful_compositions = [
            comp
            for comp in self.composition_history
            if comp["success"]
            and comp["performance_metrics"].get("efficiency", 0) > 0.7
        ]

        if len(successful_compositions) < 5:
            return

        # Look for patterns in successful patterns
        pattern_clusters = {}

        for comp in successful_compositions:
            signature = comp["pattern_signature"]
            signature_parts = signature.split("->")

            for part in signature_parts:
                if part not in pattern_clusters:
                    pattern_clusters[part] = []
                pattern_clusters[part].append(comp)

        # Identify frequently successful sub-patterns
        for pattern_part, compositions in pattern_clusters.items():
            if len(compositions) >= 3:  # Found a meta-pattern
                meta_pattern_key = f"meta_{pattern_part}"

                avg_efficiency = sum(
                    comp["performance_metrics"].get("efficiency", 0)
                    for comp in compositions
                ) / len(compositions)

                self.meta_patterns[meta_pattern_key] = {
                    "meta_type": "frequent_success",
                    "avg_efficiency": avg_efficiency,
                    "frequency": len(compositions),
                    "discovery_timestamp": logger.name,  # Placeholder
                }

        logger.info(f"🔗 Discovered {len(pattern_clusters)} meta-patterns")

    def get_composition_insights(self) -> dict[str, Any]:
        """Get insights about HOW capabilities compose and evolve"""
        total_compositions = len(self.composition_history)
        successful_compositions = sum(
            1 for comp in self.composition_history if comp["success"]
        )

        return {
            "total_compositions": total_compositions,
            "success_rate": successful_compositions / max(total_compositions, 1),
            "meta_patterns_discovered": len(
                [p for p in self.meta_patterns if p.startswith("meta_")]
            ),
            "capability_count": self.capability_graph.number_of_nodes(),
            "connection_count": self.capability_graph.number_of_edges(),
            "top_patterns": sorted(
                [
                    (pattern, data.get("avg_efficiency", 0))
                    for pattern, data in self.meta_patterns.items()
                    if not pattern.startswith("failure_")
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }


# Example usage demonstrating pure "HOW TO DO" architecture
async def demonstrate_pure_procedural_encoding():
    """
    Demonstrate how this architecture encodes pure execution patterns
    without any task-specific knowledge
    """
    engine = CapabilityCompositionEngine()

    # Register capabilities by HOW they execute, not WHAT they do
    text_processor = ExecutionNode(
        node_id=str(uuid4()),
        execution_signature="input:text_stream->processing:pattern_matching->output:structured_data",
        input_patterns=["text_stream", "string_data", "unstructured_input"],
        output_patterns=["structured_data", "parsed_content", "key_value_pairs"],
        success_conditions={"parsing_accuracy": 0.9, "processing_time": 2.0},
        failure_patterns=["malformed_input", "encoding_error", "timeout"],
    )

    data_transformer = ExecutionNode(
        node_id=str(uuid4()),
        execution_signature="input:structured_data->processing:transformation->output:formatted_result",
        input_patterns=["structured_data", "key_value_pairs", "json_data"],
        output_patterns=["formatted_result", "display_ready", "standardized_format"],
        success_conditions={"transformation_accuracy": 0.95, "processing_time": 1.0},
        failure_patterns=["invalid_schema", "transformation_error", "memory_limit"],
    )

    # Register capabilities
    engine.register_capability(text_processor)
    engine.register_capability(data_transformer)

    # Discover execution path based on pure procedural patterns
    execution_path = await engine.discover_execution_path(
        input_signature="text_stream",
        output_signature="formatted_result",
        constraints={"max_path_length": 3, "efficiency_threshold": 0.8},
    )

    logger.info(f"🔗 Discovered execution path with {len(execution_path)} steps")

    # Simulate learning from execution
    execution_trace = [
        {
            "type": "text_processing",
            "duration": 0.5,
            "success": True,
            "input_pattern": "text_stream",
            "output_pattern": "structured_data",
        },
        {
            "type": "data_transformation",
            "duration": 0.3,
            "success": True,
            "input_pattern": "structured_data",
            "output_pattern": "formatted_result",
        },
    ]

    await engine.learn_composition_pattern(
        execution_trace=execution_trace,
        success=True,
        performance_metrics={"efficiency": 0.85, "accuracy": 0.92},
    )

    # Evolve execution strategy
    current_strategy = {
        "risk_tolerance": 0.5,
        "exploration_rate": 0.3,
        "optimization_focus": 0.6,
        "complexity_preference": 0.4,
    }

    feedback = {"success_rate": 0.9, "efficiency": 0.75, "adaptability": 0.6}

    evolved_strategy = await engine.evolve_execution_strategy(
        current_strategy, feedback
    )

    # Get composition insights
    insights = engine.get_composition_insights()

    logger.info("🔗 Pure procedural architecture demonstration complete")
    logger.info(f"🔗 Composition insights: {insights}")

    return {
        "execution_path": execution_path,
        "evolved_strategy": evolved_strategy,
        "insights": insights,
    }


if __name__ == "__main__":
    import asyncio

    asyncio.run(demonstrate_pure_procedural_encoding())
