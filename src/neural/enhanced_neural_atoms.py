#!/usr/bin/env python3
"""
Enhanced Neural Atoms with EOS Integration and Mangle Reasoning

This module extends the base Neural Atom system with:
- EOS LADDER orchestration for knowledge processing
- Mangle deductive reasoning for atom validation
- Advanced relationship mapping and inference
- Constitutional compliance validation
- Knowledge graph integration
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

try:
    from prometheus_client import Counter, Histogram

    # Prometheus metrics
    atom_operations = Counter(
        "enhanced_neural_atom_operations_total",
        "Enhanced Neural Atom operations",
        ["operation", "atom_type"]
    )
    atom_processing_time = Histogram(
        "enhanced_neural_atom_processing_seconds", 
        "Enhanced Neural Atom processing time",
        ["operation"]
    )
    prometheus_available = True
except ImportError:
    prometheus_available = False

    class MockCounter:
        def labels(self, **kwargs: Any) -> "MockCounter":
            return self
        
        def inc(self) -> None:
            pass

    class MockHistogram:
        def labels(self, **kwargs: Any) -> "MockHistogram":
            return self
        
        def time(self):
            return self
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass

    atom_operations = MockCounter()
    atom_processing_time = MockHistogram()

# Import base Atom class
try:
    from .atom import ATOM_NS, Atom
except ImportError:
    # Fallback if import fails
    ATOM_NS = uuid.UUID("d6e2a8b1-4c7f-4e0a-8b9c-1d2e3f4a5b6c")
    
    @dataclass
    class Atom:
        atom_type: str
        title: str
        content: str
        meta: dict[str, Any]
        atom_id: str | None = None

logger = logging.getLogger(__name__)


class AtomProcessingStage(Enum):
    """Stages of atom processing through EOS LADDER"""
    LIFT = "lift"           # Discover and extract knowledge
    DECOMPOSE = "decompose"  # Break down complex atoms
    SYNTHESIZE = "synthesize"  # Combine related atoms
    DESCEND = "descend"      # Apply and materialize knowledge


class AtomRelationType(Enum):
    """Types of relationships between atoms"""
    DEPENDS_ON = "depends_on"
    DERIVES_FROM = "derives_from"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    REFINES = "refines"
    CONTAINS = "contains"
    SIMILAR_TO = "similar_to"


class ValidationLevel(Enum):
    """Levels of atom validation"""
    BASIC = "basic"         # Structure and format validation
    CONTENT = "content"     # Content quality and coherence
    CONSTITUTIONAL = "constitutional"  # Constitutional compliance
    MANGLE = "mangle"       # Mangle reasoning validation


@dataclass
class AtomRelationship:
    """Relationship between two atoms"""
    source_atom_id: str
    target_atom_id: str
    relation_type: AtomRelationType
    confidence: float
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


@dataclass
class AtomValidationResult:
    """Result of atom validation"""
    is_valid: bool
    validation_level: ValidationLevel
    confidence: float
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EOSProcessingContext:
    """Context for EOS processing of atoms"""
    current_stage: AtomProcessingStage
    domain: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    goals: list[str] = field(default_factory=list)
    processing_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EnhancedAtom(Atom):
    """Enhanced Neural Atom with advanced capabilities"""
    
    # EOS processing information
    processing_stage: AtomProcessingStage = AtomProcessingStage.LIFT
    eos_context: EOSProcessingContext | None = None
    
    # Validation and quality metrics
    validation_results: list[AtomValidationResult] = field(
        default_factory=list
    )
    quality_score: float = 0.0
    constitutional_score: float = 0.0
    
    # Relationship information
    relationships: list[AtomRelationship] = field(default_factory=list)
    inferred_properties: dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    last_processed: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    processing_count: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if not self.eos_context:
            self.eos_context = EOSProcessingContext(
                current_stage=self.processing_stage
            )
        
        # Update metrics
        if prometheus_available:
            atom_operations.labels(
                operation="created", atom_type=self.atom_type
            ).inc()


class MangleReasoningEngine:
    """Mangle-based reasoning engine for atom validation and inference"""
    
    def __init__(self):
        self.reasoning_rules = self._initialize_reasoning_rules()
        self.inference_cache = {}
    
    def _initialize_reasoning_rules(self) -> dict[str, Any]:
        """Initialize mangle reasoning rules"""
        return {
            "consistency_rules": [
                {
                    "name": "no_contradiction",
                    "description": "Atoms should not contradict each other",
                    "pattern": "atom_a.content != negation(atom_b.content)",
                    "severity": "error"
                },
                {
                    "name": "dependency_coherence",
                    "description": "Dependencies should be logically coherent",
                    "pattern": "depends_on(A, B) → compatible(A, B)",
                    "severity": "warning"
                }
            ],
            "quality_rules": [
                {
                    "name": "content_completeness",
                    "description": (
                        "Atoms should have complete and meaningful content"
                    ),
                    "pattern": "length(content) > min_threshold AND has_semantic_meaning(content)",
                    "severity": "warning"
                },
                {
                    "name": "metadata_richness",
                    "description": "Atoms should have rich metadata for context",
                    "pattern": "metadata.keys().size() >= 3",
                    "severity": "info"
                }
            ]
        }
    
    async def validate_atom(self, atom: EnhancedAtom) -> AtomValidationResult:
        """Validate atom using mangle reasoning"""
        with atom_processing_time.labels(operation="mangle_validation").time():
            issues = []
            recommendations = []
            confidence = 1.0
            
            # Apply consistency rules
            for rule in self.reasoning_rules["consistency_rules"]:
                violation = await self._check_rule_violation(atom, rule)
                if violation:
                    issues.append(f"{rule['name']}: {violation}")
                    confidence *= 0.8
                    if rule['severity'] == 'error':
                        confidence *= 0.5
            
            # Apply quality rules
            for rule in self.reasoning_rules["quality_rules"]:
                violation = await self._check_rule_violation(atom, rule)
                if violation:
                    issues.append(f"{rule['name']}: {violation}")
                    recommendations.append(f"Consider improving: {rule['description']}")
                    confidence *= 0.9
            
            is_valid = confidence > 0.5 and not any("error" in issue for issue in issues)
            
            return AtomValidationResult(
                is_valid=is_valid,
                validation_level=ValidationLevel.MANGLE,
                confidence=confidence,
                issues=issues,
                recommendations=recommendations,
                metadata={
                    "rules_applied": len(self.reasoning_rules["consistency_rules"]) + len(self.reasoning_rules["quality_rules"]),
                    "processing_time": time.time()
                }
            )
    
    async def _check_rule_violation(self, atom: EnhancedAtom, rule: dict[str, Any]) -> str | None:
        """Check if atom violates a specific rule"""
        try:
            # Simplified rule checking - in practice would use more sophisticated logic
            rule_name = rule["name"]
            
            if rule_name == "content_completeness":
                if len(atom.content.strip()) < 10:
                    return "Content too short or empty"
                if not any(char.isalpha() for char in atom.content):
                    return "Content lacks meaningful text"
            
            elif rule_name == "metadata_richness":
                if len(atom.meta.keys()) < 3:
                    return "Insufficient metadata fields"
            
            elif rule_name == "no_contradiction":
                # Would check against other atoms in context
                pass
            
            elif rule_name == "dependency_coherence":
                # Would check relationship consistency
                pass
            
            return None
        except Exception as e:
            logger.warning(f"Rule check failed for {rule_name}: {e}")
            return None
    
    async def infer_relationships(self, atom: EnhancedAtom, context_atoms: list[EnhancedAtom]) -> list[AtomRelationship]:
        """Infer relationships between atoms using mangle reasoning"""
        relationships = []
        
        for context_atom in context_atoms:
            if context_atom.atom_id == atom.atom_id:
                continue
            
            # Semantic similarity check
            similarity = await self._calculate_semantic_similarity(atom, context_atom)
            if similarity > 0.7:
                relationships.append(AtomRelationship(
                    source_atom_id=atom.atom_id,
                    target_atom_id=context_atom.atom_id,
                    relation_type=AtomRelationType.SIMILAR_TO,
                    confidence=similarity,
                    evidence=[f"High semantic similarity: {similarity:.2f}"]
                ))
            
            # Content dependency check
            if await self._check_content_dependency(atom, context_atom):
                relationships.append(AtomRelationship(
                    source_atom_id=atom.atom_id,
                    target_atom_id=context_atom.atom_id,
                    relation_type=AtomRelationType.DEPENDS_ON,
                    confidence=0.8,
                    evidence=["Content references detected"]
                ))
        
        return relationships
    
    async def _calculate_semantic_similarity(self, atom1: EnhancedAtom, atom2: EnhancedAtom) -> float:
        """Calculate semantic similarity between atoms"""
        # Simplified similarity calculation
        words1 = set(atom1.content.lower().split())
        words2 = set(atom2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _check_content_dependency(self, atom1: EnhancedAtom, atom2: EnhancedAtom) -> bool:
        """Check if atom1 depends on atom2 based on content"""
        # Check if atom1 references atom2's title or key concepts
        atom2_concepts = [atom2.title.lower()]
        atom2_concepts.extend(atom2.meta.get("keywords", []))
        
        atom1_content_lower = atom1.content.lower()
        
        return any(concept in atom1_content_lower for concept in atom2_concepts)


class EOSAtomOrchestrator:
    """EOS-based orchestrator for atom processing using LADDER methodology"""
    
    def __init__(self):
        self.mangle_engine = MangleReasoningEngine()
        self.processing_stats = {
            "atoms_processed": 0,
            "relationships_inferred": 0,
            "validations_performed": 0,
            "processing_time": 0.0
        }
    
    async def process_atom_through_ladder(self, atom: EnhancedAtom, context: EOSProcessingContext | None = None) -> EnhancedAtom:
        """Process atom through EOS LADDER methodology"""
        if context:
            atom.eos_context = context
        
        start_time = time.time()
        
        try:
            # LIFT: Discover and extract knowledge
            atom = await self._lift_atom(atom)
            
            # DECOMPOSE: Break down complex atoms if needed
            atom = await self._decompose_atom(atom)
            
            # SYNTHESIZE: Combine with related knowledge
            atom = await self._synthesize_atom(atom)
            
            # DESCEND: Apply and materialize knowledge
            atom = await self._descend_atom(atom)
            
            # Update processing metadata
            atom.processing_count += 1
            atom.last_processed = datetime.now(UTC).isoformat()
            
            processing_time = time.time() - start_time
            self.processing_stats["processing_time"] += processing_time
            self.processing_stats["atoms_processed"] += 1
            
            if prometheus_available:
                atom_operations.labels(operation="ladder_processed", atom_type=atom.atom_type).inc()
            
            return atom
            
        except Exception as e:
            logger.error(f"LADDER processing failed for atom {atom.atom_id}: {e}")
            raise
    
    async def _lift_atom(self, atom: EnhancedAtom) -> EnhancedAtom:
        """LIFT: Discover and extract knowledge from atom"""
        atom.processing_stage = AtomProcessingStage.LIFT
        
        # Extract key concepts and entities
        extracted_concepts = await self._extract_concepts(atom.content)
        atom.inferred_properties["concepts"] = extracted_concepts
        
        # Validate basic structure
        validation_result = await self._basic_validation(atom)
        atom.validation_results.append(validation_result)
        
        # Update context
        if atom.eos_context:
            atom.eos_context.processing_history.append({
                "stage": "lift",
                "timestamp": datetime.now(UTC).isoformat(),
                "concepts_extracted": len(extracted_concepts),
                "validation_passed": validation_result.is_valid
            })
        
        return atom
    
    async def _decompose_atom(self, atom: EnhancedAtom) -> EnhancedAtom:
        """DECOMPOSE: Break down complex atoms into simpler components"""
        atom.processing_stage = AtomProcessingStage.DECOMPOSE
        
        # Check if atom needs decomposition
        needs_decomposition = len(atom.content.split()) > 100 or atom.content.count('\n') > 10
        
        if needs_decomposition:
            # Identify decomposition points
            decomposition_points = await self._identify_decomposition_points(atom)
            atom.inferred_properties["decomposition_points"] = decomposition_points
            
            # Mark for potential splitting
            atom.meta["decomposition_suggested"] = True
            atom.meta["decomposition_complexity"] = len(decomposition_points)
        
        return atom
    
    async def _synthesize_atom(self, atom: EnhancedAtom) -> EnhancedAtom:
        """SYNTHESIZE: Combine atom with related knowledge and infer relationships"""
        atom.processing_stage = AtomProcessingStage.SYNTHESIZE
        
        # This would typically involve accessing a knowledge base
        # For now, we'll simulate synthesis
        
        # Perform mangle reasoning validation
        mangle_validation = await self.mangle_engine.validate_atom(atom)
        atom.validation_results.append(mangle_validation)
        
        # Update quality score based on validations
        atom.quality_score = sum(v.confidence for v in atom.validation_results) / len(atom.validation_results)
        
        self.processing_stats["validations_performed"] += 1
        
        return atom
    
    async def _descend_atom(self, atom: EnhancedAtom) -> EnhancedAtom:
        """DESCEND: Apply and materialize knowledge"""
        atom.processing_stage = AtomProcessingStage.DESCEND
        
        # Calculate final scores
        atom.quality_score = max(atom.quality_score, 0.5)  # Ensure minimum quality
        
        # Apply constitutional compliance check if content suggests it
        if any(keyword in atom.content.lower() for keyword in ["code", "policy", "rule", "compliance"]):
            constitutional_result = await self._constitutional_validation(atom)
            atom.validation_results.append(constitutional_result)
            atom.constitutional_score = constitutional_result.confidence
        
        # Finalize atom state
        atom.meta["processing_complete"] = True
        atom.meta["final_quality_score"] = atom.quality_score
        atom.meta["validation_count"] = len(atom.validation_results)
        
        return atom
    
    async def _extract_concepts(self, content: str) -> list[str]:
        """Extract key concepts from content"""
        # Simplified concept extraction
        words = content.lower().split()
        
        # Filter for meaningful concepts (longer words, proper nouns, etc.)
        concepts = []
        for word in words:
            clean_word = word.strip('.,!?";:')
            if len(clean_word) > 4 and clean_word.isalpha():
                concepts.append(clean_word)
        
        # Remove duplicates and return top concepts
        unique_concepts = list(set(concepts))
        return unique_concepts[:10]  # Top 10 concepts
    
    async def _basic_validation(self, atom: EnhancedAtom) -> AtomValidationResult:
        """Perform basic validation of atom structure"""
        issues = []
        
        if not atom.title.strip():
            issues.append("Missing or empty title")
        
        if not atom.content.strip():
            issues.append("Missing or empty content")
        
        if not atom.atom_type.strip():
            issues.append("Missing or empty atom type")
        
        confidence = 1.0 - (len(issues) * 0.3)
        
        return AtomValidationResult(
            is_valid=len(issues) == 0,
            validation_level=ValidationLevel.BASIC,
            confidence=max(confidence, 0.0),
            issues=issues,
            recommendations=["Ensure all required fields are populated"] if issues else []
        )
    
    async def _identify_decomposition_points(self, atom: EnhancedAtom) -> list[dict[str, Any]]:
        """Identify points where atom could be decomposed"""
        points = []
        
        # Look for natural breaking points
        paragraphs = atom.content.split('\n\n')
        if len(paragraphs) > 3:
            for i, paragraph in enumerate(paragraphs[1:], 1):  # Skip first paragraph
                if len(paragraph.strip()) > 50:  # Substantial paragraph
                    points.append({
                        "type": "paragraph_break",
                        "position": i,
                        "preview": paragraph[:50] + "..." if len(paragraph) > 50 else paragraph
                    })
        
        # Look for list structures
        lines = atom.content.split('\n')
        list_items = [i for i, line in enumerate(lines) if line.strip().startswith(('-', '*', '1.', '2.'))]
        if len(list_items) > 2:
            points.append({
                "type": "list_structure",
                "position": list_items[0],
                "item_count": len(list_items)
            })
        
        return points
    
    async def _constitutional_validation(self, atom: EnhancedAtom) -> AtomValidationResult:
        """Validate atom for constitutional compliance"""
        # Simplified constitutional validation
        issues = []
        
        content_lower = atom.content.lower()
        
        # Check for potentially problematic content
        problematic_terms = ["discriminatory", "biased", "harmful", "illegal"]
        for term in problematic_terms:
            if term in content_lower:
                issues.append(f"Potentially problematic content detected: {term}")
        
        # Check for privacy concerns
        if any(pattern in content_lower for pattern in ["personal data", "private information", "confidential"]):
            issues.append("Potential privacy concerns detected")
        
        confidence = 1.0 - (len(issues) * 0.2)
        
        return AtomValidationResult(
            is_valid=len(issues) == 0,
            validation_level=ValidationLevel.CONSTITUTIONAL,
            confidence=max(confidence, 0.0),
            issues=issues,
            recommendations=["Review content for constitutional compliance"] if issues else []
        )
    
    def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            "avg_processing_time": self.processing_stats["processing_time"] / max(1, self.processing_stats["atoms_processed"]),
            "validation_rate": self.processing_stats["validations_performed"] / max(1, self.processing_stats["atoms_processed"])
        }


class EnhancedNeuralStore:
    """Enhanced neural store for managing enhanced atoms with relationships"""
    
    def __init__(self):
        self.atoms: dict[str, EnhancedAtom] = {}
        self.relationships: dict[str, list[AtomRelationship]] = {}
        self.orchestrator = EOSAtomOrchestrator()
        self.atom_index = {}  # For efficient searching
    
    async def add_atom(self, atom: EnhancedAtom, process_through_ladder: bool = True) -> str:
        """Add atom to store with optional LADDER processing"""
        if process_through_ladder:
            atom = await self.orchestrator.process_atom_through_ladder(atom)
        
        self.atoms[atom.atom_id] = atom
        
        # Update search index
        await self._update_search_index(atom)
        
        # Infer relationships with existing atoms if we have context
        if len(self.atoms) > 1:
            context_atoms = list(self.atoms.values())
            relationships = await self.orchestrator.mangle_engine.infer_relationships(atom, context_atoms)
            
            for relationship in relationships:
                await self.add_relationship(relationship)
        
        if prometheus_available:
            atom_operations.labels(operation="stored", atom_type=atom.atom_type).inc()
        
        return atom.atom_id
    
    async def add_relationship(self, relationship: AtomRelationship) -> None:
        """Add relationship between atoms"""
        source_id = relationship.source_atom_id
        
        if source_id not in self.relationships:
            self.relationships[source_id] = []
        
        self.relationships[source_id].append(relationship)
        self.orchestrator.processing_stats["relationships_inferred"] += 1
    
    async def get_atom(self, atom_id: str) -> EnhancedAtom | None:
        """Get atom by ID"""
        return self.atoms.get(atom_id)
    
    async def get_related_atoms(self, atom_id: str, relation_type: AtomRelationType | None = None) -> list[tuple[EnhancedAtom, AtomRelationship]]:
        """Get atoms related to the specified atom"""
        related = []
        relationships = self.relationships.get(atom_id, [])
        
        for relationship in relationships:
            if relation_type is None or relationship.relation_type == relation_type:
                target_atom = self.atoms.get(relationship.target_atom_id)
                if target_atom:
                    related.append((target_atom, relationship))
        
        return related
    
    async def search_atoms(self, query: str, atom_type: str | None = None) -> list[EnhancedAtom]:
        """Search atoms by content and metadata"""
        results = []
        query_lower = query.lower()
        
        for atom in self.atoms.values():
            if atom_type and atom.atom_type != atom_type:
                continue
            
            # Search in title, content, and concepts
            if (query_lower in atom.title.lower() or 
                query_lower in atom.content.lower() or
                any(query_lower in concept for concept in atom.inferred_properties.get("concepts", []))):
                results.append(atom)
        
        # Sort by quality score
        results.sort(key=lambda a: a.quality_score, reverse=True)
        return results
    
    async def _update_search_index(self, atom: EnhancedAtom) -> None:
        """Update search index for efficient querying"""
        concepts = atom.inferred_properties.get("concepts", [])
        keywords = [atom.title.lower(), atom.atom_type.lower()] + concepts
        
        for keyword in keywords:
            if keyword not in self.atom_index:
                self.atom_index[keyword] = []
            self.atom_index[keyword].append(atom.atom_id)
    
    def get_stats(self) -> dict[str, Any]:
        """Get store statistics"""
        total_validations = sum(len(atom.validation_results) for atom in self.atoms.values())
        avg_quality = sum(atom.quality_score for atom in self.atoms.values()) / max(1, len(self.atoms))
        
        return {
            "total_atoms": len(self.atoms),
            "total_relationships": sum(len(rels) for rels in self.relationships.values()),
            "total_validations": total_validations,
            "average_quality_score": avg_quality,
            "processing_stats": self.orchestrator.get_processing_stats(),
            "atom_types": list(set(atom.atom_type for atom in self.atoms.values())),
            "index_size": len(self.atom_index)
        }


# Factory functions for creating enhanced atoms
async def create_enhanced_atom(
    atom_type: str,
    title: str,
    content: str,
    meta: dict[str, Any] | None = None,
    domain: str | None = None
) -> EnhancedAtom:
    """Factory function to create enhanced atom"""
    return EnhancedAtom(
        atom_type=atom_type,
        title=title,
        content=content,
        meta=meta or {},
        eos_context=EOSProcessingContext(
            current_stage=AtomProcessingStage.LIFT,
            domain=domain
        )
    )


# Export main classes
__all__ = [
    "EnhancedAtom",
    "EOSAtomOrchestrator", 
    "EnhancedNeuralStore",
    "MangleReasoningEngine",
    "AtomRelationship",
    "AtomValidationResult",
    "AtomProcessingStage",
    "AtomRelationType",
    "ValidationLevel",
    "create_enhanced_atom"
]