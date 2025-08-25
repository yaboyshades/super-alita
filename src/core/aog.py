# src/core/aog.py
"""
Defines the data structures for the And-Or Graph (AOG) used by the
LADDER reasoning engine.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

# Define the possible types for a node in the graph.
AOGNodeType = Literal["AND", "OR", "LEAF"]


class AOGNode(BaseModel):
    """
    Represents a single node in the And-Or Graph.

    Each node is a symbolic representation of a part of a plan or a goal.
    These nodes will be stored as the 'value' of a NeuralAtom in the NeuralStore.
    """

    node_id: str = Field(..., description="A unique, human-readable ID for this node.")
    description: str = Field(
        ...,
        description="A natural language description of what this node represents (e.g., 'Analyze bounty requirements'). This will be embedded.",
    )
    node_type: AOGNodeType = Field(..., alias="type")

    # Children are represented by their node_ids, which correspond to NeuralAtom keys.
    children: list[str] = Field(default_factory=list)

    # For LEAF nodes, this template defines the tool to be executed.
    tool_template: dict | None = Field(
        None, description="Template for the tool call if this is a LEAF node."
    )

    # Additional metadata for enhanced reasoning
    priority: float = Field(1.0, description="Priority weight for this node")
    cost_estimate: float = Field(1.0, description="Estimated cost/effort for this node")
    success_probability: float = Field(
        0.8, description="Estimated probability of success"
    )
    preconditions: list[str] = Field(
        default_factory=list, description="Required preconditions"
    )
    postconditions: list[str] = Field(
        default_factory=list, description="Expected outcomes"
    )

    # Execution tracking
    execution_count: int = Field(
        0, description="Number of times this node has been executed"
    )
    success_count: int = Field(0, description="Number of successful executions")
    last_executed: str | None = Field(
        None, description="ISO timestamp of last execution"
    )

    class Config:
        # Allow 'type' to be used as a field name, which is a keyword in Python.
        populate_by_name = True

    def get_success_rate(self) -> float:
        """Calculate the success rate for this node."""
        if self.execution_count == 0:
            return self.success_probability
        return self.success_count / self.execution_count

    def update_execution_stats(self, success: bool, timestamp: str | None = None):
        """Update execution statistics."""
        self.execution_count += 1
        if success:
            self.success_count += 1
        self.last_executed = timestamp or datetime.now(UTC).isoformat()

    def is_terminal(self) -> bool:
        """Check if this is a terminal (executable) node."""
        return self.node_type == "LEAF"

    def is_and(self) -> bool:
        """Check if this is an AND node."""
        return self.node_type == "AND"

    def is_or(self) -> bool:
        """Check if this is an OR node."""
        return self.node_type == "OR"


class AOGPlan(BaseModel):
    """
    Represents a complete execution plan generated from the AOG.
    """

    plan_id: str = Field(..., description="Unique identifier for this plan")
    goal_node_id: str = Field(..., description="The root goal node this plan addresses")
    steps: list[str] = Field(
        ..., description="Ordered list of LEAF node IDs to execute"
    )
    path_taken: list[str] = Field(
        ..., description="All node IDs traversed during planning"
    )

    # Plan metadata
    estimated_cost: float = Field(0.0, description="Total estimated cost of the plan")
    estimated_probability: float = Field(0.0, description="Overall success probability")
    created_at: str = Field(..., description="ISO timestamp when plan was created")

    # Execution tracking
    status: str = Field(
        "created", description="Plan status: created, executing, completed, failed"
    )
    current_step: int = Field(0, description="Index of current step being executed")
    execution_log: list[dict] = Field(
        default_factory=list, description="Log of step executions"
    )

    def add_execution_log(
        self,
        step_index: int,
        node_id: str,
        success: bool,
        duration: float,
        output: str = "",
        error: str = "",
    ):
        """Add an execution log entry."""
        self.execution_log.append(
            {
                "step_index": step_index,
                "node_id": node_id,
                "success": success,
                "duration": duration,
                "output": output,
                "error": error,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def get_completion_rate(self) -> float:
        """Get the completion rate of the plan."""
        if not self.steps:
            return 0.0
        return self.current_step / len(self.steps)


class AOGAnalysis(BaseModel):
    """
    Represents analysis results from the AOG reasoning system.
    """

    analysis_id: str = Field(..., description="Unique identifier for this analysis")
    analysis_type: str = Field(
        ..., description="Type of analysis: forward_plan, backward_diagnosis, etc."
    )

    # Analysis results
    nodes_analyzed: list[str] = Field(..., description="Node IDs that were analyzed")
    critical_path: list[str] = Field(
        default_factory=list, description="Most critical nodes in the analysis"
    )
    bottlenecks: list[str] = Field(
        default_factory=list, description="Identified bottleneck nodes"
    )
    alternatives: list[list[str]] = Field(
        default_factory=list, description="Alternative paths found"
    )

    # Metrics
    confidence_score: float = Field(
        0.0, description="Confidence in the analysis results"
    )
    complexity_score: float = Field(
        0.0, description="Complexity metric for the analyzed graph"
    )

    # Recommendations
    recommendations: list[str] = Field(
        default_factory=list, description="Strategic recommendations"
    )
    risk_factors: list[str] = Field(
        default_factory=list, description="Identified risk factors"
    )

    created_at: str = Field(
        ..., description="ISO timestamp when analysis was performed"
    )
