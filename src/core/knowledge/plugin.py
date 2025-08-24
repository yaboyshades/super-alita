"""
Knowledge Graph Plugin for Super Alita
"""

from typing import Optional, Dict, Any
from pathlib import Path

from ..plugin_interface import PluginInterface
from .store import KnowledgeStore
from .handlers import KnowledgeGraphEventHandlers


class KnowledgeGraphPlugin(PluginInterface):
    """
    Plugin that maintains a deterministic knowledge graph
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("knowledge_graph.db")
        self.knowledge_store: Optional[KnowledgeStore] = None
        self.handlers: Optional[KnowledgeGraphEventHandlers] = None
        self.event_bus = None
        
    @property
    def name(self) -> str:
        return "knowledge_graph"
    
    async def setup(self, event_bus=None, **kwargs):
        """Initialize the knowledge graph plugin"""
        self.event_bus = event_bus
        
        # Initialize knowledge store
        self.knowledge_store = KnowledgeStore(self.db_path)
        
        # Initialize event handlers
        self.handlers = KnowledgeGraphEventHandlers(self.knowledge_store)
        
        # Subscribe to all events if event bus is available
        if self.event_bus:
            await self.event_bus.subscribe("*", self.handlers.handle_event)
            print(f"📊 Knowledge Graph Plugin initialized (DB: {self.db_path})")
        else:
            print(f"📊 Knowledge Graph Plugin initialized without event bus (DB: {self.db_path})")
    
    async def start(self):
        """Start the knowledge graph plugin"""
        if not self.knowledge_store:
            await self.setup()
    
    async def shutdown(self):
        """Shutdown the knowledge graph plugin"""
        if self.knowledge_store:
            self.knowledge_store.close()
            self.knowledge_store = None
        
        self.handlers = None
        print("📊 Knowledge Graph Plugin shutdown complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        if not self.knowledge_store:
            return {"error": "Knowledge store not initialized"}
        
        return self.knowledge_store.get_statistics()
    
    def search_atoms(self, search_term: str, limit: int = 50):
        """Search atoms by content"""
        if not self.knowledge_store:
            return []
        
        return self.knowledge_store.search_atoms_by_content(search_term, limit)
    
    def get_atom(self, atom_id: str):
        """Get atom by ID"""
        if not self.knowledge_store:
            return None
        
        return self.knowledge_store.get_atom(atom_id)
    
    def get_related_atoms(self, atom_id: str):
        """Get atoms related to the given atom"""
        if not self.knowledge_store:
            return []
        
        # Get outgoing bonds
        outgoing_bonds = self.knowledge_store.get_bonds_from_atom(atom_id)
        # Get incoming bonds
        incoming_bonds = self.knowledge_store.get_bonds_to_atom(atom_id)
        
        related_atom_ids = set()
        for bond in outgoing_bonds:
            related_atom_ids.add(bond.to_atom_id)
        for bond in incoming_bonds:
            related_atom_ids.add(bond.from_atom_id)
        
        # Retrieve the actual atoms
        related_atoms = []
        for related_id in related_atom_ids:
            atom = self.knowledge_store.get_atom(related_id)
            if atom:
                related_atoms.append(atom)
        
        return related_atoms
    
    def create_manual_concept(self, name: str, description: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """Create a concept atom manually"""
        if not self.handlers:
            raise RuntimeError("Knowledge graph not initialized")
        
        return self.handlers.create_concept_atom(name, description, properties)
    
    def create_manual_entity(self, name: str, entity_type: str, attributes: Optional[Dict[str, Any]] = None) -> str:
        """Create an entity atom manually"""
        if not self.handlers:
            raise RuntimeError("Knowledge graph not initialized")
        
        return self.handlers.create_entity_atom(name, entity_type, attributes)
    
    def link_concepts(
        self, 
        from_atom_id: str, 
        to_atom_id: str, 
        relationship: str, 
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a relationship between atoms"""
        if not self.handlers:
            raise RuntimeError("Knowledge graph not initialized")
        
        return self.handlers.link_atoms(from_atom_id, to_atom_id, relationship, strength, metadata)
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export the entire knowledge graph for analysis"""
        if not self.knowledge_store:
            return {"error": "Knowledge store not initialized"}
        
        # Get all atoms by type
        from .store import AtomType, BondType
        
        all_atoms = {}
        for atom_type in AtomType:
            atoms = self.knowledge_store.get_atoms_by_type(atom_type, limit=1000)
            all_atoms[atom_type.value] = [atom.to_dict() for atom in atoms]
        
        # Get all bonds
        cursor = self.knowledge_store.connection.execute("SELECT * FROM bonds")
        all_bonds = []
        for row in cursor:
            all_bonds.append({
                "bond_id": row["bond_id"],
                "from_atom_id": row["from_atom_id"],
                "to_atom_id": row["to_atom_id"],
                "bond_type": row["bond_type"],
                "strength": row["strength"],
                "metadata": row["metadata"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            })
        
        return {
            "atoms": all_atoms,
            "bonds": all_bonds,
            "statistics": self.get_statistics()
        }