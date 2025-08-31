# Neural Atom/Bond System - Agent Instructions

## Overview
The `src/neural/` directory implements the cognitive fabric of Super Alita:
- **Neural Atoms** - Atomic units of cognition with deterministic UUIDs
- **Neural Bonds** - Relationships and connections between atoms
- **Memory Store** - Persistent graph storage for cognitive artifacts
- **MCP Integration** - Model Context Protocol server for neural operations

## Key Files & Components

### Core Neural System
- `atom.py` - Neural atom implementation and core logic
- `bond.py` - Neural bond relationships and graph connections
- `store.py` - Persistent storage backend for atoms and bonds
- `mcp_server.py` - MCP server for neural operations

### Neural Atom Architecture
```python
# Neural atoms are deterministic cognitive artifacts
class NeuralAtom:
    uuid: str          # Deterministic UUID from content hash
    content: Any       # The actual data/content
    atom_type: str     # Type classification (tool_output, memory, etc.)
    title: str         # Human-readable title
    metadata: Dict     # Additional context and provenance
    timestamp: datetime # Creation timestamp
    bonds: List[Bond]  # Connections to other atoms
```

## Deterministic Identity System

### UUID Generation
```python
from src.core.neural_atom import create_atom

# UUIDs are deterministic based on content
atom1 = create_atom(
    content={"key": "value"},
    atom_type="tool_output",
    title="Example Atom"
)

atom2 = create_atom(
    content={"key": "value"},  # Same content
    atom_type="tool_output",   # Same type
    title="Example Atom"      # Same title
)

assert atom1.uuid == atom2.uuid  # Same UUID for same content
```

### Content Normalization
```python
def normalize_content(content: Any) -> str:
    """Normalize content for deterministic UUID generation"""

    if isinstance(content, dict):
        # Sort keys for consistency
        normalized = json.dumps(content, sort_keys=True, default=str)
    elif isinstance(content, list):
        # Handle lists with potential ordering issues
        normalized = json.dumps(sorted(content), default=str)
    else:
        # Convert to string representation
        normalized = str(content)

    return normalized

def generate_atom_uuid(content: Any, atom_type: str, title: str) -> str:
    """Generate deterministic UUID for atom"""
    normalized_content = normalize_content(content)
    seed_string = f"{normalized_content}|{atom_type}|{title}"

    # Use UUIDv5 with namespace for determinism
    return str(uuid.uuid5(NEURAL_NAMESPACE, seed_string))
```

## Neural Bond System

### Bond Types
```python
class BondType(Enum):
    CAUSAL = "causal"          # A caused B
    TEMPORAL = "temporal"      # A happened before B
    SEMANTIC = "semantic"      # A is semantically related to B
    HIERARCHICAL = "hierarchical"  # A contains/is parent of B
    REFERENCE = "reference"    # A references B
    DERIVATION = "derivation"  # A was derived from B
```

### Creating Bonds
```python
from src.neural.bond import create_bond

# Create relationship between atoms
bond = create_bond(
    source_atom=atom1,
    target_atom=atom2,
    bond_type=BondType.CAUSAL,
    strength=0.95,  # Confidence in relationship
    metadata={
        "context": "tool execution",
        "timestamp": datetime.now(timezone.utc)
    }
)
```

## Memory Store Operations

### Storing Atoms
```python
from src.neural.store import NeuralStore

store = NeuralStore()

# Store single atom
await store.store_atom(atom)

# Store with bonds
await store.store_atom_with_bonds(atom, bonds)

# Batch storage
await store.store_atoms_batch([atom1, atom2, atom3])
```

### Querying Atoms
```python
# Query by type
tool_outputs = await store.query_atoms(atom_type="tool_output")

# Query by content similarity
similar_atoms = await store.find_similar_atoms(
    reference_atom=query_atom,
    similarity_threshold=0.8,
    max_results=10
)

# Query by bonds
related_atoms = await store.get_connected_atoms(
    atom_uuid=source_uuid,
    bond_types=[BondType.CAUSAL, BondType.TEMPORAL],
    max_depth=2
)
```

### Graph Traversal
```python
# Find path between atoms
path = await store.find_path(
    start_atom_uuid=start_uuid,
    end_atom_uuid=end_uuid,
    max_depth=5
)

# Get neighborhood
neighborhood = await store.get_neighborhood(
    center_atom_uuid=center_uuid,
    radius=2,
    bond_types=[BondType.SEMANTIC]
)
```

## Development Guidelines

### Creating Atoms
```python
# Best practices for atom creation
def create_tool_output_atom(
    tool_name: str,
    input_data: Dict,
    output_data: Any,
    execution_context: Dict
) -> NeuralAtom:
    """Create atom for tool execution output"""

    content = {
        "tool_name": tool_name,
        "input": input_data,
        "output": output_data,
        "execution_time": execution_context.get("duration"),
        "success": execution_context.get("success", True)
    }

    title = f"{tool_name} execution result"

    metadata = {
        "provenance": {
            "source": "tool_executor",
            "activity": "tool_execution",
            "timestamp": datetime.now(timezone.utc),
            "context": execution_context
        }
    }

    return create_atom(
        content=content,
        atom_type="tool_output",
        title=title,
        metadata=metadata
    )
```

### Bond Creation Patterns
```python
# Create causal bonds for tool chains
async def create_tool_chain_bonds(
    input_atom: NeuralAtom,
    tool_atoms: List[NeuralAtom],
    output_atom: NeuralAtom
) -> List[Bond]:
    """Create bonds representing tool execution chain"""

    bonds = []

    # Input caused first tool
    bonds.append(create_bond(
        source_atom=input_atom,
        target_atom=tool_atoms[0],
        bond_type=BondType.CAUSAL,
        strength=1.0
    ))

    # Chain tool executions
    for i in range(len(tool_atoms) - 1):
        bonds.append(create_bond(
            source_atom=tool_atoms[i],
            target_atom=tool_atoms[i + 1],
            bond_type=BondType.TEMPORAL,
            strength=1.0
        ))

    # Last tool caused output
    bonds.append(create_bond(
        source_atom=tool_atoms[-1],
        target_atom=output_atom,
        bond_type=BondType.DERIVATION,
        strength=1.0
    ))

    return bonds
```

## MCP Integration

### Neural MCP Server
```python
# The neural system exposes MCP tools for external access
class NeuralMCPServer:
    """MCP server for neural operations"""

    def __init__(self, store: NeuralStore):
        self.store = store

    async def create_atom_tool(self, content: Dict, atom_type: str, title: str) -> Dict:
        """MCP tool for creating atoms"""
        atom = create_atom(content, atom_type, title)
        await self.store.store_atom(atom)

        return {
            "success": True,
            "atom_uuid": atom.uuid,
            "message": f"Created atom: {title}"
        }

    async def query_atoms_tool(self, query_params: Dict) -> Dict:
        """MCP tool for querying atoms"""
        atoms = await self.store.query_atoms(**query_params)

        return {
            "success": True,
            "atoms": [atom.to_dict() for atom in atoms],
            "count": len(atoms)
        }
```

## Testing Guidelines

### Atom Testing
```python
import pytest
from src.neural.atom import NeuralAtom
from src.core.neural_atom import create_atom

def test_atom_deterministic_uuid():
    """Test that atoms have deterministic UUIDs"""
    content = {"test": "data"}

    atom1 = create_atom(content, "test_type", "Test Atom")
    atom2 = create_atom(content, "test_type", "Test Atom")

    assert atom1.uuid == atom2.uuid

def test_atom_different_content_different_uuid():
    """Test that different content produces different UUIDs"""
    atom1 = create_atom({"test": "data1"}, "test_type", "Test")
    atom2 = create_atom({"test": "data2"}, "test_type", "Test")

    assert atom1.uuid != atom2.uuid

@pytest.mark.asyncio
async def test_atom_storage_retrieval():
    """Test storing and retrieving atoms"""
    store = NeuralStore()

    atom = create_atom({"test": "data"}, "test_type", "Test Atom")
    await store.store_atom(atom)

    retrieved = await store.get_atom(atom.uuid)
    assert retrieved.uuid == atom.uuid
    assert retrieved.content == atom.content
```

### Bond Testing
```python
from src.neural.bond import create_bond, BondType

def test_bond_creation():
    """Test creating bonds between atoms"""
    atom1 = create_atom({"input": "data"}, "input", "Input")
    atom2 = create_atom({"output": "result"}, "output", "Output")

    bond = create_bond(
        source_atom=atom1,
        target_atom=atom2,
        bond_type=BondType.CAUSAL,
        strength=0.9
    )

    assert bond.source_uuid == atom1.uuid
    assert bond.target_uuid == atom2.uuid
    assert bond.bond_type == BondType.CAUSAL
    assert bond.strength == 0.9

@pytest.mark.asyncio
async def test_graph_traversal():
    """Test graph traversal operations"""
    store = NeuralStore()

    # Create connected atoms
    atom1 = create_atom({"step": 1}, "process", "Step 1")
    atom2 = create_atom({"step": 2}, "process", "Step 2")
    atom3 = create_atom({"step": 3}, "process", "Step 3")

    bond1 = create_bond(atom1, atom2, BondType.TEMPORAL, 1.0)
    bond2 = create_bond(atom2, atom3, BondType.TEMPORAL, 1.0)

    # Store in graph
    await store.store_atom_with_bonds(atom1, [bond1])
    await store.store_atom_with_bonds(atom2, [bond2])
    await store.store_atom(atom3)

    # Test path finding
    path = await store.find_path(atom1.uuid, atom3.uuid)
    assert len(path) == 3
    assert path[0].uuid == atom1.uuid
    assert path[-1].uuid == atom3.uuid
```

## Performance Guidelines

### Efficient Storage
```python
# Batch operations for better performance
async def store_tool_execution_atoms(
    executions: List[ToolExecution]
) -> None:
    """Efficiently store multiple tool executions"""

    atoms = []
    bonds = []

    for execution in executions:
        atom = create_tool_output_atom(execution)
        atoms.append(atom)

        # Create bonds to previous executions
        if len(atoms) > 1:
            bond = create_bond(
                atoms[-2], atoms[-1],
                BondType.TEMPORAL, 1.0
            )
            bonds.append(bond)

    # Batch store everything
    await store.store_atoms_batch(atoms)
    await store.store_bonds_batch(bonds)
```

### Caching Strategies
```python
from functools import lru_cache
from src.core.cache import TTLCache

class CachedNeuralStore:
    """Neural store with caching for performance"""

    def __init__(self, store: NeuralStore):
        self.store = store
        self.atom_cache = TTLCache(maxsize=1000, ttl=300)  # 5 min TTL

    async def get_atom(self, uuid: str) -> NeuralAtom:
        """Get atom with caching"""
        if uuid in self.atom_cache:
            return self.atom_cache[uuid]

        atom = await self.store.get_atom(uuid)
        self.atom_cache[uuid] = atom
        return atom

    @lru_cache(maxsize=100)
    def get_similar_atoms_cached(
        self,
        content_hash: str,
        threshold: float
    ) -> List[NeuralAtom]:
        """Cache similarity queries"""
        # Implementation would hash content for cache key
        pass
```

## Common Patterns

### Atom Lineage Tracking
```python
class AtomLineage:
    """Track atom creation and derivation lineage"""

    @staticmethod
    async def create_derived_atom(
        source_atoms: List[NeuralAtom],
        transformation: str,
        result_content: Any,
        result_title: str
    ) -> NeuralAtom:
        """Create atom derived from other atoms"""

        # Create result atom
        result_atom = create_atom(
            content=result_content,
            atom_type="derived_result",
            title=result_title,
            metadata={
                "transformation": transformation,
                "source_uuids": [atom.uuid for atom in source_atoms]
            }
        )

        # Create derivation bonds
        bonds = [
            create_bond(
                source_atom=source_atom,
                target_atom=result_atom,
                bond_type=BondType.DERIVATION,
                strength=1.0
            )
            for source_atom in source_atoms
        ]

        # Store with lineage
        await store.store_atom_with_bonds(result_atom, bonds)

        return result_atom
```

### Semantic Clustering
```python
async def cluster_atoms_by_similarity(
    atoms: List[NeuralAtom],
    similarity_threshold: float = 0.8
) -> List[List[NeuralAtom]]:
    """Cluster atoms by semantic similarity"""

    clusters = []
    processed = set()

    for atom in atoms:
        if atom.uuid in processed:
            continue

        # Find similar atoms
        similar = await store.find_similar_atoms(
            reference_atom=atom,
            similarity_threshold=similarity_threshold
        )

        # Create cluster
        cluster = [atom] + [a for a in similar if a.uuid not in processed]
        clusters.append(cluster)

        # Mark as processed
        for cluster_atom in cluster:
            processed.add(cluster_atom.uuid)

    return clusters
```

## Debugging & Monitoring

### Neural System Health
```python
async def check_neural_system_health() -> Dict:
    """Check health of neural system components"""

    health = {
        "store_accessible": False,
        "atom_count": 0,
        "bond_count": 0,
        "recent_activity": False
    }

    try:
        # Test store access
        await store.ping()
        health["store_accessible"] = True

        # Get counts
        health["atom_count"] = await store.count_atoms()
        health["bond_count"] = await store.count_bonds()

        # Check recent activity
        recent_atoms = await store.query_atoms(
            created_after=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        health["recent_activity"] = len(recent_atoms) > 0

    except Exception as e:
        health["error"] = str(e)

    return health
```

### Atom Metrics
```python
from src.core.metrics import MetricsCollector

# Track atom operations
MetricsCollector.increment("neural.atom.created")
MetricsCollector.increment("neural.bond.created")
MetricsCollector.gauge("neural.store.atom_count", await store.count_atoms())

# Track performance
with MetricsCollector.timer("neural.query.similarity"):
    similar_atoms = await store.find_similar_atoms(atom)
```
