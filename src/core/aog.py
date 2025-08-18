# src/core/aog.py
"""
And-Or Graph (AOG) node definitions for LADDER reasoning.

AOG nodes represent hierarchical decision structures:
- AND nodes: All children must be satisfied
- OR nodes: One child must be satisfied (attention-based selection)
- TERMINAL nodes: Executable actions (MCP tool calls)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class AOGNodeType(Enum):
    """Types of AOG nodes."""
    AND = "AND"
    OR = "OR"
    TERMINAL = "TERMINAL"


@dataclass
class AOGNode:
    """
    A node in the And-Or Graph for LADDER reasoning.
    
    Each node represents a step in goal decomposition or action execution.
    """
    
    node_id: str
    type: str  # "AND", "OR", or "TERMINAL"
    description: str
    children: List[str] = field(default_factory=list)
    tool_template: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate node configuration."""
        if self.type not in ["AND", "OR", "TERMINAL"]:
            raise ValueError(f"Invalid node type: {self.type}")
        
        if self.type == "TERMINAL" and not self.tool_template:
            raise ValueError("TERMINAL nodes must have a tool_template")
        
        if self.type in ["AND", "OR"] and not self.children:
            raise ValueError(f"{self.type} nodes must have children")
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal (executable) node."""
        return self.type == "TERMINAL"
    
    def is_and(self) -> bool:
        """Check if this is an AND node."""
        return self.type == "AND"
    
    def is_or(self) -> bool:
        """Check if this is an OR node."""
        return self.type == "OR"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "type": self.type,
            "description": self.description,
            "children": self.children,
            "tool_template": self.tool_template,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AOGNode":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            type=data["type"],
            description=data["description"],
            children=data.get("children", []),
            tool_template=data.get("tool_template"),
            metadata=data.get("metadata", {})
        )
    
    def __repr__(self) -> str:
        return f"AOGNode(id='{self.node_id}', type='{self.type}', children={len(self.children)})"
