"""
SQLite-backed Knowledge Graph Store with deterministic atom/bond management
"""

import sqlite3
import hashlib
import json
import uuid
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


class AtomType(Enum):
    """Types of atoms in the knowledge graph"""
    CONCEPT = "concept"
    ENTITY = "entity"
    EVENT = "event"
    RELATIONSHIP = "relationship"
    COGNITIVE_TURN = "cognitive_turn"
    CORTEX_RESULT = "cortex_result"
    TELEMETRY_MARKER = "telemetry_marker"


class BondType(Enum):
    """Types of bonds between atoms"""
    RELATES_TO = "relates_to"
    CAUSED_BY = "caused_by"
    CONTAINS = "contains"
    FOLLOWS = "follows"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    DERIVED_FROM = "derived_from"
    TRIGGERS = "triggers"


@dataclass
class Atom:
    """An atom in the knowledge graph"""
    atom_id: str  # Deterministic UUID based on content
    atom_type: AtomType
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    hash_signature: str  # Content hash for idempotency
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert atom to dictionary"""
        return {
            "atom_id": self.atom_id,
            "atom_type": self.atom_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "hash_signature": self.hash_signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Atom":
        """Create atom from dictionary"""
        return cls(
            atom_id=data["atom_id"],
            atom_type=AtomType(data["atom_type"]),
            content=data["content"],
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            hash_signature=data["hash_signature"]
        )


@dataclass
class Bond:
    """A bond between atoms in the knowledge graph"""
    bond_id: str  # Deterministic UUID based on atom IDs and bond type
    from_atom_id: str
    to_atom_id: str
    bond_type: BondType
    strength: float  # 0.0 to 1.0
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bond to dictionary"""
        return {
            "bond_id": self.bond_id,
            "from_atom_id": self.from_atom_id,
            "to_atom_id": self.to_atom_id,
            "bond_type": self.bond_type.value,
            "strength": self.strength,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bond":
        """Create bond from dictionary"""
        return cls(
            bond_id=data["bond_id"],
            from_atom_id=data["from_atom_id"],
            to_atom_id=data["to_atom_id"],
            bond_type=BondType(data["bond_type"]),
            strength=data["strength"],
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )


class KnowledgeStore:
    """
    SQLite-backed knowledge graph store with deterministic atom/bond management
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("knowledge_graph.db")
        self.connection: Optional[sqlite3.Connection] = None
        self._setup_database()
    
    def _setup_database(self):
        """Initialize SQLite database with required tables"""
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row  # Enable dict-like access
        
        # Create atoms table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS atoms (
                atom_id TEXT PRIMARY KEY,
                atom_type TEXT NOT NULL,
                content TEXT NOT NULL,  -- JSON
                metadata TEXT NOT NULL, -- JSON
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                hash_signature TEXT NOT NULL UNIQUE
            )
        """)
        
        # Create bonds table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS bonds (
                bond_id TEXT PRIMARY KEY,
                from_atom_id TEXT NOT NULL,
                to_atom_id TEXT NOT NULL,
                bond_type TEXT NOT NULL,
                strength REAL NOT NULL,
                metadata TEXT NOT NULL, -- JSON
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (from_atom_id) REFERENCES atoms (atom_id),
                FOREIGN KEY (to_atom_id) REFERENCES atoms (atom_id)
            )
        """)
        
        # Create indexes for performance
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_atoms_type ON atoms (atom_type)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_atoms_hash ON atoms (hash_signature)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_bonds_from ON bonds (from_atom_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_bonds_to ON bonds (to_atom_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_bonds_type ON bonds (bond_type)")
        
        self.connection.commit()
    
    def _generate_content_hash(self, content: Dict[str, Any]) -> str:
        """Generate deterministic hash from content"""
        # Sort keys for consistent hashing
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _generate_atom_id(self, atom_type: AtomType, content: Dict[str, Any]) -> str:
        """Generate deterministic atom ID"""
        # Create a deterministic hash from atom type and content
        content_str = json.dumps(content, sort_keys=True)
        combined_data = f"{atom_type.value}:{content_str}"
        hash_digest = hashlib.sha256(combined_data.encode()).hexdigest()
        
        # Convert to UUID format for consistency
        uuid_str = f"{hash_digest[:8]}-{hash_digest[8:12]}-{hash_digest[12:16]}-{hash_digest[16:20]}-{hash_digest[20:32]}"
        return uuid_str
    
    def _generate_bond_id(self, from_atom_id: str, to_atom_id: str, bond_type: BondType) -> str:
        """Generate deterministic bond ID"""
        bond_data = f"{from_atom_id}:{to_atom_id}:{bond_type.value}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, bond_data))
    
    def create_atom(
        self, 
        atom_type: AtomType, 
        content: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Atom:
        """Create or retrieve existing atom (idempotent)"""
        if metadata is None:
            metadata = {}
        
        # Generate deterministic IDs
        atom_id = self._generate_atom_id(atom_type, content)
        hash_signature = self._generate_content_hash(content)
        
        # Check if atom already exists
        cursor = self.connection.execute(
            "SELECT * FROM atoms WHERE hash_signature = ?",
            (hash_signature,)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Return existing atom
            return Atom.from_dict({
                "atom_id": existing["atom_id"],
                "atom_type": existing["atom_type"],
                "content": json.loads(existing["content"]),
                "metadata": json.loads(existing["metadata"]),
                "created_at": existing["created_at"],
                "updated_at": existing["updated_at"],
                "hash_signature": existing["hash_signature"]
            })
        
        # Create new atom
        now = datetime.now(UTC)
        atom = Atom(
            atom_id=atom_id,
            atom_type=atom_type,
            content=content,
            metadata=metadata,
            created_at=now,
            updated_at=now,
            hash_signature=hash_signature
        )
        
        # Insert into database
        self.connection.execute("""
            INSERT INTO atoms (
                atom_id, atom_type, content, metadata, 
                created_at, updated_at, hash_signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            atom.atom_id,
            atom.atom_type.value,
            json.dumps(atom.content),
            json.dumps(atom.metadata),
            atom.created_at.isoformat(),
            atom.updated_at.isoformat(),
            atom.hash_signature
        ))
        self.connection.commit()
        
        return atom
    
    def create_bond(
        self,
        from_atom_id: str,
        to_atom_id: str,
        bond_type: BondType,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Bond:
        """Create or update bond between atoms"""
        if metadata is None:
            metadata = {}
        
        # Validate strength
        strength = max(0.0, min(1.0, strength))
        
        # Generate deterministic bond ID
        bond_id = self._generate_bond_id(from_atom_id, to_atom_id, bond_type)
        
        # Check if bond already exists
        cursor = self.connection.execute(
            "SELECT * FROM bonds WHERE bond_id = ?",
            (bond_id,)
        )
        existing = cursor.fetchone()
        
        now = datetime.now(UTC)
        
        if existing:
            # Update existing bond
            self.connection.execute("""
                UPDATE bonds SET 
                    strength = ?, 
                    metadata = ?, 
                    updated_at = ?
                WHERE bond_id = ?
            """, (
                strength,
                json.dumps(metadata),
                now.isoformat(),
                bond_id
            ))
            
            return Bond(
                bond_id=bond_id,
                from_atom_id=from_atom_id,
                to_atom_id=to_atom_id,
                bond_type=bond_type,
                strength=strength,
                metadata=metadata,
                created_at=datetime.fromisoformat(existing["created_at"]),
                updated_at=now
            )
        else:
            # Create new bond
            bond = Bond(
                bond_id=bond_id,
                from_atom_id=from_atom_id,
                to_atom_id=to_atom_id,
                bond_type=bond_type,
                strength=strength,
                metadata=metadata,
                created_at=now,
                updated_at=now
            )
            
            self.connection.execute("""
                INSERT INTO bonds (
                    bond_id, from_atom_id, to_atom_id, bond_type,
                    strength, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bond.bond_id,
                bond.from_atom_id,
                bond.to_atom_id,
                bond.bond_type.value,
                bond.strength,
                json.dumps(bond.metadata),
                bond.created_at.isoformat(),
                bond.updated_at.isoformat()
            ))
        
        self.connection.commit()
        return bond
    
    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Retrieve atom by ID"""
        cursor = self.connection.execute(
            "SELECT * FROM atoms WHERE atom_id = ?",
            (atom_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return Atom.from_dict({
            "atom_id": row["atom_id"],
            "atom_type": row["atom_type"],
            "content": json.loads(row["content"]),
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "hash_signature": row["hash_signature"]
        })
    
    def get_atoms_by_type(self, atom_type: AtomType, limit: int = 100) -> List[Atom]:
        """Retrieve atoms by type"""
        cursor = self.connection.execute(
            "SELECT * FROM atoms WHERE atom_type = ? ORDER BY created_at DESC LIMIT ?",
            (atom_type.value, limit)
        )
        
        atoms = []
        for row in cursor:
            atoms.append(Atom.from_dict({
                "atom_id": row["atom_id"],
                "atom_type": row["atom_type"],
                "content": json.loads(row["content"]),
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "hash_signature": row["hash_signature"]
            }))
        
        return atoms
    
    def get_bonds_from_atom(self, atom_id: str) -> List[Bond]:
        """Get all bonds originating from an atom"""
        cursor = self.connection.execute(
            "SELECT * FROM bonds WHERE from_atom_id = ?",
            (atom_id,)
        )
        
        bonds = []
        for row in cursor:
            bonds.append(Bond.from_dict({
                "bond_id": row["bond_id"],
                "from_atom_id": row["from_atom_id"],
                "to_atom_id": row["to_atom_id"],
                "bond_type": row["bond_type"],
                "strength": row["strength"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }))
        
        return bonds
    
    def get_bonds_to_atom(self, atom_id: str) -> List[Bond]:
        """Get all bonds pointing to an atom"""
        cursor = self.connection.execute(
            "SELECT * FROM bonds WHERE to_atom_id = ?",
            (atom_id,)
        )
        
        bonds = []
        for row in cursor:
            bonds.append(Bond.from_dict({
                "bond_id": row["bond_id"],
                "from_atom_id": row["from_atom_id"],
                "to_atom_id": row["to_atom_id"],
                "bond_type": row["bond_type"],
                "strength": row["strength"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }))
        
        return bonds
    
    def search_atoms_by_content(self, search_term: str, limit: int = 50) -> List[Atom]:
        """Search atoms by content (simple text search)"""
        cursor = self.connection.execute("""
            SELECT * FROM atoms 
            WHERE content LIKE ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (f"%{search_term}%", limit))
        
        atoms = []
        for row in cursor:
            atoms.append(Atom.from_dict({
                "atom_id": row["atom_id"],
                "atom_type": row["atom_type"],
                "content": json.loads(row["content"]),
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "hash_signature": row["hash_signature"]
            }))
        
        return atoms
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        # Count atoms by type
        atom_counts = {}
        for atom_type in AtomType:
            cursor = self.connection.execute(
                "SELECT COUNT(*) as count FROM atoms WHERE atom_type = ?",
                (atom_type.value,)
            )
            atom_counts[atom_type.value] = cursor.fetchone()["count"]
        
        # Count bonds by type
        bond_counts = {}
        for bond_type in BondType:
            cursor = self.connection.execute(
                "SELECT COUNT(*) as count FROM bonds WHERE bond_type = ?",
                (bond_type.value,)
            )
            bond_counts[bond_type.value] = cursor.fetchone()["count"]
        
        # Total counts
        total_atoms = sum(atom_counts.values())
        total_bonds = sum(bond_counts.values())
        
        return {
            "total_atoms": total_atoms,
            "total_bonds": total_bonds,
            "atoms_by_type": atom_counts,
            "bonds_by_type": bond_counts,
            "database_path": str(self.db_path)
        }
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()