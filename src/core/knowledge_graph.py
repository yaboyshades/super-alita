"""
Knowledge Graph Enrichment Architecture for Super Alita

Provides dynamic knowledge graph enrichment from event streams with temporal versioning
and provenance annotations for enhanced reasoning and memory management.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in the knowledge graph"""

    USER = "user"
    TOOL = "tool"
    CONCEPT = "concept"
    ACTION = "action"
    RESULT = "result"
    CONTEXT = "context"
    DYNAMIC_TOOL = "dynamic_tool"


class RelationType(Enum):
    """Types of relationships between entities"""

    USES = "uses"
    CREATES = "creates"
    DEPENDS_ON = "depends_on"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    RESULTED_IN = "resulted_in"
    TRIGGERED_BY = "triggered_by"
    ENHANCES = "enhances"


@dataclass
class ProvenanceAnnotation:
    """Provenance information for knowledge graph entities"""

    source_event_id: str
    session_id: str
    timestamp: datetime
    confidence_score: float  # 0.0 to 1.0
    evidence: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_event_id": self.source_event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "confidence_score": self.confidence_score,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
        }


@dataclass
class KGEntity:
    """Knowledge graph entity with temporal versioning"""

    id: str
    type: EntityType
    properties: dict[str, Any]
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    provenance: list[ProvenanceAnnotation] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())

    def update_properties(
        self, new_properties: dict[str, Any], provenance: ProvenanceAnnotation
    ):
        """Update entity properties with versioning"""
        self.properties.update(new_properties)
        self.version += 1
        self.updated_at = datetime.now(UTC)
        self.provenance.append(provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "properties": self.properties,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "provenance": [p.to_dict() for p in self.provenance],
        }


@dataclass
class KGRelationship:
    """Knowledge graph relationship with temporal versioning"""

    id: str
    from_entity_id: str
    to_entity_id: str
    relationship_type: RelationType
    properties: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Relationship strength
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    provenance: list[ProvenanceAnnotation] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())

    def update_weight(self, new_weight: float, provenance: ProvenanceAnnotation):
        """Update relationship weight with provenance"""
        self.weight = new_weight
        self.version += 1
        self.updated_at = datetime.now(UTC)
        self.provenance.append(provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from_entity_id": self.from_entity_id,
            "to_entity_id": self.to_entity_id,
            "relationship_type": self.relationship_type.value,
            "properties": self.properties,
            "weight": self.weight,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "provenance": [p.to_dict() for p in self.provenance],
        }


class EventProcessor:
    """Processes events to extract knowledge graph insights"""

    def __init__(self):
        self.entity_extractors = {
            "tool_call": self._extract_tool_entities,
            "dynamic_tool_created": self._extract_dynamic_tool_entities,
            "user_input": self._extract_user_entities,
            "response_generated": self._extract_response_entities,
        }

        self.relationship_extractors = {
            "tool_call": self._extract_tool_relationships,
            "dynamic_tool_created": self._extract_dynamic_tool_relationships,
            "state_transition": self._extract_state_relationships,
        }

    async def process_event(
        self, event: dict[str, Any]
    ) -> tuple[list[KGEntity], list[KGRelationship]]:
        """Process an event to extract entities and relationships"""
        event_type = event.get("type", "unknown")
        entities = []
        relationships = []

        # Extract entities
        entity_extractor = self.entity_extractors.get(event_type)
        if entity_extractor:
            entities = await entity_extractor(event)

        # Extract relationships
        relationship_extractor = self.relationship_extractors.get(event_type)
        if relationship_extractor:
            relationships = await relationship_extractor(event, entities)

        return entities, relationships

    async def _extract_tool_entities(self, event: dict[str, Any]) -> list[KGEntity]:
        """Extract entities from tool call events"""
        entities = []

        tool_name = event.get("tool_name")
        if tool_name:
            provenance = ProvenanceAnnotation(
                source_event_id=event.get("id", "unknown"),
                session_id=event.get("session_id", "unknown"),
                timestamp=datetime.fromisoformat(
                    event.get("timestamp", datetime.now(UTC).isoformat())
                ),
                confidence_score=0.9,
                evidence={"event_type": "tool_call", "tool_name": tool_name},
            )

            tool_entity = KGEntity(
                id=f"tool_{tool_name}",
                type=EntityType.TOOL,
                properties={
                    "name": tool_name,
                    "usage_count": 1,
                    "last_used": datetime.now(UTC).isoformat(),
                },
                provenance=[provenance],
            )
            entities.append(tool_entity)

        return entities

    async def _extract_dynamic_tool_entities(
        self, event: dict[str, Any]
    ) -> list[KGEntity]:
        """Extract entities from dynamic tool creation events"""
        entities = []

        tool_data = event.get("tool_data", {})
        tool_name = tool_data.get("name")

        if tool_name:
            provenance = ProvenanceAnnotation(
                source_event_id=event.get("id", "unknown"),
                session_id=event.get("session_id", "unknown"),
                timestamp=datetime.fromisoformat(
                    event.get("timestamp", datetime.now(UTC).isoformat())
                ),
                confidence_score=1.0,
                evidence={"event_type": "dynamic_tool_created", "tool_data": tool_data},
            )

            dynamic_tool_entity = KGEntity(
                id=f"dynamic_tool_{tool_name}",
                type=EntityType.DYNAMIC_TOOL,
                properties={
                    "name": tool_name,
                    "description": tool_data.get("description", ""),
                    "parameters": tool_data.get("parameters", []),
                    "created_at": datetime.now(UTC).isoformat(),
                    "creator": "super_alita",
                },
                provenance=[provenance],
            )
            entities.append(dynamic_tool_entity)

        return entities

    async def _extract_user_entities(self, event: dict[str, Any]) -> list[KGEntity]:
        """Extract entities from user input events"""
        entities = []

        user_id = event.get("user_id", "anonymous")
        user_input = event.get("content", "")

        provenance = ProvenanceAnnotation(
            source_event_id=event.get("id", "unknown"),
            session_id=event.get("session_id", "unknown"),
            timestamp=datetime.fromisoformat(
                event.get("timestamp", datetime.now(UTC).isoformat())
            ),
            confidence_score=1.0,
            evidence={"event_type": "user_input", "input_length": len(user_input)},
        )

        user_entity = KGEntity(
            id=f"user_{user_id}",
            type=EntityType.USER,
            properties={
                "user_id": user_id,
                "interaction_count": 1,
                "last_interaction": datetime.now(UTC).isoformat(),
            },
            provenance=[provenance],
        )
        entities.append(user_entity)

        return entities

    async def _extract_response_entities(self, event: dict[str, Any]) -> list[KGEntity]:
        """Extract entities from response generation events"""
        entities = []

        response_content = event.get("content", "")
        if response_content:
            provenance = ProvenanceAnnotation(
                source_event_id=event.get("id", "unknown"),
                session_id=event.get("session_id", "unknown"),
                timestamp=datetime.fromisoformat(
                    event.get("timestamp", datetime.now(UTC).isoformat())
                ),
                confidence_score=0.8,
                evidence={
                    "event_type": "response_generated",
                    "content_length": len(response_content),
                },
            )

            response_entity = KGEntity(
                id=f"response_{event.get('id', uuid4())}",
                type=EntityType.RESULT,
                properties={
                    "content": response_content[:500],  # Truncate for storage
                    "length": len(response_content),
                    "generated_at": datetime.now(UTC).isoformat(),
                },
                provenance=[provenance],
            )
            entities.append(response_entity)

        return entities

    async def _extract_tool_relationships(
        self, event: dict[str, Any], _entities: list[KGEntity]
    ) -> list[KGRelationship]:
        """Extract relationships from tool call events"""
        relationships = []

        user_id = event.get("user_id", "anonymous")
        tool_name = event.get("tool_name")

        if tool_name:
            provenance = ProvenanceAnnotation(
                source_event_id=event.get("id", "unknown"),
                session_id=event.get("session_id", "unknown"),
                timestamp=datetime.fromisoformat(
                    event.get("timestamp", datetime.now(UTC).isoformat())
                ),
                confidence_score=0.9,
                evidence={"event_type": "tool_call"},
            )

            # User uses tool relationship
            uses_relationship = KGRelationship(
                id=f"uses_{user_id}_{tool_name}",
                from_entity_id=f"user_{user_id}",
                to_entity_id=f"tool_{tool_name}",
                relationship_type=RelationType.USES,
                weight=1.0,
                provenance=[provenance],
            )
            relationships.append(uses_relationship)

        return relationships

    async def _extract_dynamic_tool_relationships(
        self, event: dict[str, Any], _entities: list[KGEntity]
    ) -> list[KGRelationship]:
        """Extract relationships from dynamic tool creation events"""
        relationships = []

        tool_data = event.get("tool_data", {})
        tool_name = tool_data.get("name")
        user_id = event.get("user_id", "anonymous")

        if tool_name:
            provenance = ProvenanceAnnotation(
                source_event_id=event.get("id", "unknown"),
                session_id=event.get("session_id", "unknown"),
                timestamp=datetime.fromisoformat(
                    event.get("timestamp", datetime.now(UTC).isoformat())
                ),
                confidence_score=1.0,
                evidence={"event_type": "dynamic_tool_created"},
            )

            # User creates dynamic tool relationship
            creates_relationship = KGRelationship(
                id=f"creates_{user_id}_{tool_name}",
                from_entity_id=f"user_{user_id}",
                to_entity_id=f"dynamic_tool_{tool_name}",
                relationship_type=RelationType.CREATES,
                weight=1.0,
                provenance=[provenance],
            )
            relationships.append(creates_relationship)

        return relationships

    async def _extract_state_relationships(
        self, event: dict[str, Any], _entities: list[KGEntity]
    ) -> list[KGRelationship]:
        """Extract relationships from state transition events"""
        relationships = []

        from_state = event.get("from_state")
        to_state = event.get("to_state")
        trigger = event.get("trigger")

        if from_state and to_state and trigger:
            provenance = ProvenanceAnnotation(
                source_event_id=event.get("id", "unknown"),
                session_id=event.get("session_id", "unknown"),
                timestamp=datetime.fromisoformat(
                    event.get("timestamp", datetime.now(UTC).isoformat())
                ),
                confidence_score=0.9,
                evidence={"event_type": "state_transition", "trigger": trigger},
            )

            # State triggers state relationship
            triggered_relationship = KGRelationship(
                id=f"triggers_{from_state}_{to_state}",
                from_entity_id=f"state_{from_state}",
                to_entity_id=f"state_{to_state}",
                relationship_type=RelationType.TRIGGERED_BY,
                properties={"trigger": trigger},
                weight=1.0,
                provenance=[provenance],
            )
            relationships.append(triggered_relationship)

        return relationships


class KnowledgeGraphEnricher:
    """Main knowledge graph enrichment coordinator"""

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.event_processor = EventProcessor()
        self.entities: dict[str, KGEntity] = {}
        self.relationships: dict[str, KGRelationship] = {}
        self.enrichment_log: list[dict[str, Any]] = []

    async def start_enrichment(self):
        """Start listening to events for knowledge graph enrichment"""
        if self.event_bus:
            await self.event_bus.subscribe("*", self._process_event_for_kg)
        logger.info("Knowledge graph enrichment started")

    async def _process_event_for_kg(self, event: dict[str, Any]):
        """Process event for knowledge graph enrichment"""
        try:
            entities, relationships = await self.event_processor.process_event(event)

            # Add or update entities
            for entity in entities:
                await self._add_or_update_entity(entity)

            # Add or update relationships
            for relationship in relationships:
                await self._add_or_update_relationship(relationship)

            # Log enrichment
            self.enrichment_log.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "event_id": event.get("id", "unknown"),
                    "entities_added": len(entities),
                    "relationships_added": len(relationships),
                }
            )

            logger.debug(
                "Enriched KG with %d entities and %d relationships",
                len(entities),
                len(relationships),
            )

        except Exception as e:
            logger.error("Error processing event for KG enrichment: %s", str(e))

    async def _add_or_update_entity(self, entity: KGEntity):
        """Add or update entity in knowledge graph"""
        if entity.id in self.entities:
            # Update existing entity
            existing = self.entities[entity.id]
            existing.update_properties(
                entity.properties, entity.provenance[0] if entity.provenance else None
            )
        else:
            # Add new entity
            self.entities[entity.id] = entity

    async def _add_or_update_relationship(self, relationship: KGRelationship):
        """Add or update relationship in knowledge graph"""
        if relationship.id in self.relationships:
            # Update existing relationship weight
            existing = self.relationships[relationship.id]
            new_weight = min(existing.weight + 0.1, 2.0)  # Cap at 2.0
            existing.update_weight(
                new_weight,
                relationship.provenance[0] if relationship.provenance else None,
            )
        else:
            # Add new relationship
            self.relationships[relationship.id] = relationship

    def get_enrichment_stats(self) -> dict[str, Any]:
        """Get knowledge graph enrichment statistics"""
        entity_types = {}
        relationship_types = {}

        for entity in self.entities.values():
            entity_type = entity.type.value
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        for relationship in self.relationships.values():
            rel_type = relationship.relationship_type.value
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "enrichment_events": len(self.enrichment_log),
            "latest_enrichment": self.enrichment_log[-1]
            if self.enrichment_log
            else None,
        }

    def export_graph(self) -> dict[str, Any]:
        """Export the knowledge graph in JSON format"""
        return {
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "relationships": [rel.to_dict() for rel in self.relationships.values()],
            "metadata": {
                "exported_at": datetime.now(UTC).isoformat(),
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
            },
        }


# Global enricher instance
kg_enricher = KnowledgeGraphEnricher()


async def example_usage():
    """Example of knowledge graph enrichment"""

    # Sample events for testing
    sample_events = [
        {
            "id": "event_1",
            "type": "user_input",
            "user_id": "user123",
            "session_id": "session_abc",
            "content": "Create a calculator tool",
            "timestamp": datetime.now(UTC).isoformat(),
        },
        {
            "id": "event_2",
            "type": "dynamic_tool_created",
            "user_id": "user123",
            "session_id": "session_abc",
            "tool_data": {
                "name": "calculator",
                "description": "A simple calculator tool",
                "parameters": ["a", "b"],
            },
            "timestamp": datetime.now(UTC).isoformat(),
        },
        {
            "id": "event_3",
            "type": "tool_call",
            "user_id": "user123",
            "session_id": "session_abc",
            "tool_name": "calculator",
            "timestamp": datetime.now(UTC).isoformat(),
        },
    ]

    # Process events
    for event in sample_events:
        await kg_enricher._process_event_for_kg(event)

    # Print statistics
    stats = kg_enricher.get_enrichment_stats()
    print(f"KG Enrichment Stats: {json.dumps(stats, indent=2)}")

    # Export graph
    graph_export = kg_enricher.export_graph()
    print(
        f"Exported {len(graph_export['entities'])} entities and {len(graph_export['relationships'])} relationships"
    )


if __name__ == "__main__":
    asyncio.run(example_usage())
