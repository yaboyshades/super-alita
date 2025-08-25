"""
Event handlers for knowledge graph integration
"""

import asyncio
from typing import Any, Dict, Optional
from datetime import datetime, UTC

from ..events import BaseEvent
from .store import KnowledgeStore, AtomType, BondType


class KnowledgeGraphEventHandlers:
    """
    Event handlers that convert system events into knowledge graph atoms/bonds
    """
    
    def __init__(self, knowledge_store: KnowledgeStore):
        self.knowledge_store = knowledge_store
    
    async def handle_cortex_cycle_event(self, event: BaseEvent) -> None:
        """Handle Cortex cycle completion events"""
        try:
            if event.event_type == "cortex_cycle_complete":
                # Extract cycle data from metadata
                cycle_id = event.metadata.get("cycle_id")
                success = event.metadata.get("success", False)
                duration_ms = event.metadata.get("duration_ms", 0.0)
                error = event.metadata.get("error")
                
                if not cycle_id:
                    return
                
                # Create atom for the cortex cycle result
                cortex_atom = self.knowledge_store.create_atom(
                    atom_type=AtomType.CORTEX_RESULT,
                    content={
                        "cycle_id": cycle_id,
                        "success": success,
                        "duration_ms": duration_ms,
                        "error": error,
                        "timestamp": event.timestamp.isoformat()
                    },
                    metadata={
                        "event_id": event.event_id,
                        "source_plugin": event.source_plugin
                    }
                )
                
                # Create atom for the event itself
                event_atom = self.knowledge_store.create_atom(
                    atom_type=AtomType.EVENT,
                    content={
                        "event_type": event.event_type,
                        "event_id": event.event_id,
                        "timestamp": event.timestamp.isoformat(),
                        "source_plugin": event.source_plugin
                    },
                    metadata=event.metadata
                )
                
                # Create bond between event and cortex result
                self.knowledge_store.create_bond(
                    from_atom_id=event_atom.atom_id,
                    to_atom_id=cortex_atom.atom_id,
                    bond_type=BondType.CAUSED_BY,
                    strength=1.0,
                    metadata={"event_type": "cortex_cycle_complete"}
                )
                
                print(f"📊 Knowledge: Created atoms for Cortex cycle {cycle_id}")
                
        except Exception as e:
            print(f"⚠️ Error handling Cortex cycle event: {e}")
    
    async def handle_telemetry_event(self, event: BaseEvent) -> None:
        """Handle telemetry events"""
        try:
            # Create atom for the telemetry event
            telemetry_atom = self.knowledge_store.create_atom(
                atom_type=AtomType.TELEMETRY_MARKER,
                content={
                    "event_type": event.event_type,
                    "event_id": event.event_id,
                    "source_plugin": event.source_plugin,
                    "timestamp": event.timestamp.isoformat(),
                    "metadata": event.metadata
                },
                metadata={
                    "captured_at": datetime.now(UTC).isoformat()
                }
            )
            
            # If there's a cycle_id in metadata, create relationship
            cycle_id = event.metadata.get("cycle_id")
            if cycle_id:
                # Find the cortex result atom for this cycle
                cortex_atoms = self.knowledge_store.search_atoms_by_content(cycle_id)
                for atom in cortex_atoms:
                    if atom.atom_type == AtomType.CORTEX_RESULT:
                        # Create bond between telemetry and cortex result
                        self.knowledge_store.create_bond(
                            from_atom_id=telemetry_atom.atom_id,
                            to_atom_id=atom.atom_id,
                            bond_type=BondType.RELATES_TO,
                            strength=0.8,
                            metadata={"relationship": "telemetry_data"}
                        )
                        break
            
        except Exception as e:
            print(f"⚠️ Error handling telemetry event: {e}")
    
    async def handle_cognitive_turn_event(self, event: BaseEvent) -> None:
        """Handle cognitive turn events from DTA system"""
        try:
            if hasattr(event, 'turn_data') or 'turn_data' in event.metadata:
                turn_data = getattr(event, 'turn_data', event.metadata.get('turn_data', {}))
                
                # Create atom for the cognitive turn
                turn_atom = self.knowledge_store.create_atom(
                    atom_type=AtomType.COGNITIVE_TURN,
                    content={
                        "turn_id": turn_data.get("turn_id", event.event_id),
                        "user_input": turn_data.get("user_input", ""),
                        "agent_response": turn_data.get("agent_response", ""),
                        "confidence": turn_data.get("confidence", 0.0),
                        "timestamp": event.timestamp.isoformat()
                    },
                    metadata={
                        "event_id": event.event_id,
                        "source_plugin": event.source_plugin,
                        **turn_data
                    }
                )
                
                print(f"🧠 Knowledge: Created atom for cognitive turn {turn_data.get('turn_id', event.event_id)}")
                
        except Exception as e:
            print(f"⚠️ Error handling cognitive turn event: {e}")
    
    async def handle_generic_event(self, event: BaseEvent) -> None:
        """Handle any other event types"""
        try:
            # Create a generic event atom
            event_atom = self.knowledge_store.create_atom(
                atom_type=AtomType.EVENT,
                content={
                    "event_type": event.event_type,
                    "event_id": event.event_id,
                    "source_plugin": event.source_plugin,
                    "timestamp": event.timestamp.isoformat()
                },
                metadata=event.metadata
            )
            
            # Check for relationships in metadata
            if "related_to" in event.metadata:
                related_ids = event.metadata["related_to"]
                if isinstance(related_ids, str):
                    related_ids = [related_ids]
                
                for related_id in related_ids:
                    # Try to find related atoms
                    related_atoms = self.knowledge_store.search_atoms_by_content(related_id)
                    for related_atom in related_atoms:
                        self.knowledge_store.create_bond(
                            from_atom_id=event_atom.atom_id,
                            to_atom_id=related_atom.atom_id,
                            bond_type=BondType.RELATES_TO,
                            strength=0.6,
                            metadata={"inferred_relationship": True}
                        )
            
        except Exception as e:
            print(f"⚠️ Error handling generic event: {e}")
    
    async def handle_event(self, event: BaseEvent) -> None:
        """Main event handler that routes to specific handlers"""
        try:
            # Route based on event type
            if event.event_type == "cortex_cycle_complete":
                await self.handle_cortex_cycle_event(event)
            elif event.event_type == "cognitive_turn":
                await self.handle_cognitive_turn_event(event)
            elif "telemetry" in event.event_type.lower():
                await self.handle_telemetry_event(event)
            else:
                await self.handle_generic_event(event)
                
        except Exception as e:
            print(f"⚠️ Error in knowledge graph event handler: {e}")
    
    def create_concept_atom(
        self, 
        concept_name: str, 
        description: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a concept atom and return its ID"""
        if properties is None:
            properties = {}
        
        atom = self.knowledge_store.create_atom(
            atom_type=AtomType.CONCEPT,
            content={
                "name": concept_name,
                "description": description,
                "properties": properties
            },
            metadata={
                "created_by": "knowledge_graph_handlers",
                "manual_creation": True
            }
        )
        
        return atom.atom_id
    
    def create_entity_atom(
        self,
        entity_name: str,
        entity_type: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create an entity atom and return its ID"""
        if attributes is None:
            attributes = {}
        
        atom = self.knowledge_store.create_atom(
            atom_type=AtomType.ENTITY,
            content={
                "name": entity_name,
                "type": entity_type,
                "attributes": attributes
            },
            metadata={
                "created_by": "knowledge_graph_handlers",
                "manual_creation": True
            }
        )
        
        return atom.atom_id
    
    def link_atoms(
        self,
        from_atom_id: str,
        to_atom_id: str,
        relationship_type: str,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a bond between atoms and return bond ID"""
        if metadata is None:
            metadata = {}
        
        # Map relationship type to BondType
        bond_type_mapping = {
            "relates_to": BondType.RELATES_TO,
            "caused_by": BondType.CAUSED_BY,
            "contains": BondType.CONTAINS,
            "follows": BondType.FOLLOWS,
            "part_of": BondType.PART_OF,
            "similar_to": BondType.SIMILAR_TO,
            "derived_from": BondType.DERIVED_FROM,
            "triggers": BondType.TRIGGERS
        }
        
        bond_type = bond_type_mapping.get(relationship_type, BondType.RELATES_TO)
        
        bond = self.knowledge_store.create_bond(
            from_atom_id=from_atom_id,
            to_atom_id=to_atom_id,
            bond_type=bond_type,
            strength=strength,
            metadata={
                **metadata,
                "created_by": "knowledge_graph_handlers",
                "manual_creation": True
            }
        )
        
        return bond.bond_id