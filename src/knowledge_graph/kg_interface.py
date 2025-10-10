"""Core Knowledge Graph interface for LADDER planning enhancement."""

import time
from collections import defaultdict
from typing import Any

from .models import (
    EntityType,
    KnowledgeEntity,
    KnowledgeQuery,
    KnowledgeQueryResult,
    KnowledgeRelation,
    PlanningContext,
    PlanningPattern,
)


class KnowledgeGraphInterface:
    """
    Core Knowledge Graph interface for storing and querying planning knowledge.

    This interface provides:
    - Entity and relationship storage
    - Pattern recognition and storage
    - Semantic querying for planning context
    - Learning from planning outcomes
    """

    def __init__(self):
        """Initialize the knowledge graph."""
        # Core storage
        self.entities: dict[str, KnowledgeEntity] = {}
        self.relations: dict[str, KnowledgeRelation] = {}
        self.patterns: dict[str, PlanningPattern] = {}
        self.contexts: dict[str, PlanningContext] = {}

        # Indexes for efficient querying
        self.entity_by_type: dict[EntityType, set[str]] = defaultdict(set)
        self.relations_by_source: dict[str, set[str]] = defaultdict(set)
        self.relations_by_target: dict[str, set[str]] = defaultdict(set)
        self.patterns_by_domain: dict[str, set[str]] = defaultdict(set)

        # Initialize with some basic patterns
        self._initialize_base_patterns()

    def _initialize_base_patterns(self) -> None:
        """Initialize the KG with basic planning patterns."""
        # Code development pattern
        code_pattern = PlanningPattern(
            pattern_name="code_development",
            goal_template="create|implement|write|develop.*function|class|module",
            decomposition_steps=[
                "Analyze requirements and specifications",
                "Design the code structure and interfaces",
                "Implement the core functionality",
                "Add error handling and validation",
                "Write tests and documentation",
                "Review and optimize the code",
            ],
            domain="software_development",
            complexity_level=3,
            required_tools=["code_editor", "testing_framework"],
            preconditions=["clear_requirements", "development_environment"],
        )
        self.add_pattern(code_pattern)

        # Problem solving pattern
        problem_pattern = PlanningPattern(
            pattern_name="problem_analysis",
            goal_template="solve|fix|debug|troubleshoot|analyze",
            decomposition_steps=[
                "Understand and define the problem clearly",
                "Gather relevant information and context",
                "Identify possible causes and solutions",
                "Evaluate and select the best approach",
                "Implement the solution step by step",
                "Test and validate the results",
            ],
            domain="general",
            complexity_level=2,
            required_tools=["analysis_tools"],
            preconditions=["problem_description"],
        )
        self.add_pattern(problem_pattern)

        # Research pattern
        research_pattern = PlanningPattern(
            pattern_name="research_task",
            goal_template="research|investigate|study|explore",
            decomposition_steps=[
                "Define research scope and objectives",
                "Identify relevant sources and resources",
                "Collect and organize information",
                "Analyze and synthesize findings",
                "Draw conclusions and insights",
                "Document and present results",
            ],
            domain="research",
            complexity_level=4,
            required_tools=["search_tools", "documentation_tools"],
            preconditions=["research_topic"],
        )
        self.add_pattern(research_pattern)

    def add_entity(self, entity: KnowledgeEntity) -> str:
        """Add an entity to the knowledge graph."""
        self.entities[entity.id] = entity
        self.entity_by_type[entity.entity_type].add(entity.id)
        return entity.id

    def add_relation(self, relation: KnowledgeRelation) -> str:
        """Add a relationship to the knowledge graph."""
        self.relations[relation.id] = relation
        self.relations_by_source[relation.source_entity_id].add(relation.id)
        self.relations_by_target[relation.target_entity_id].add(relation.id)
        return relation.id

    def add_pattern(self, pattern: PlanningPattern) -> str:
        """Add a planning pattern to the knowledge graph."""
        self.patterns[pattern.id] = pattern
        self.patterns_by_domain[pattern.domain].add(pattern.id)
        return pattern.id

    def add_context(self, context: PlanningContext) -> str:
        """Add planning context for learning."""
        self.contexts[context.session_id] = context

        # Learn from successful contexts
        if context.success:
            self._learn_from_success(context)

        return context.session_id

    def _learn_from_success(self, context: PlanningContext) -> None:
        """Learn patterns from successful planning contexts."""
        # Find similar patterns and update their success rates
        for pattern_id in self.patterns_by_domain.get(context.domain, set()):
            pattern = self.patterns[pattern_id]

            # Simple similarity check based on goal keywords
            if self._is_similar_goal(context.goal, pattern.goal_template):
                # Update pattern success rate with exponential moving average
                alpha = 0.1  # Learning rate
                pattern.success_rate = (
                    alpha * 1.0 + (1 - alpha) * pattern.success_rate
                )
                pattern.usage_count += 1
                pattern.last_updated = time.time()

    def _is_similar_goal(self, goal: str, template: str) -> bool:
        """Check if a goal matches a pattern template."""
        import re

        set(goal.lower().split())
        template_parts = template.lower().split("|")

        return any(re.search(part, goal.lower()) for part in template_parts)

    def query(self, query: KnowledgeQuery) -> KnowledgeQueryResult:
        """Query the knowledge graph for relevant information."""
        start_time = time.time()
        result = KnowledgeQueryResult()

        # Find relevant entities
        relevant_entities = self._find_relevant_entities(query)
        result.entities = relevant_entities[: query.max_results]

        # Find relevant relations
        relevant_relations = self._find_relevant_relations(
            query, relevant_entities
        )
        result.relations = relevant_relations

        # Find relevant patterns
        if query.include_patterns:
            relevant_patterns = self._find_relevant_patterns(query)
            result.patterns = relevant_patterns[: query.max_results]

        # Calculate relevance scores
        result.relevance_scores = self._calculate_relevance_scores(
            query, result.entities, result.patterns
        )

        result.total_results = len(result.entities) + len(result.patterns)
        result.query_time = time.time() - start_time

        return result

    def _find_relevant_entities(
        self, query: KnowledgeQuery
    ) -> list[KnowledgeEntity]:
        """Find entities relevant to the query."""
        candidates = []

        # Filter by entity types if specified
        if query.entity_types:
            entity_ids = set()
            for entity_type in query.entity_types:
                entity_ids.update(self.entity_by_type.get(entity_type, set()))
        else:
            entity_ids = set(self.entities.keys())

        # Score and filter entities
        for entity_id in entity_ids:
            entity = self.entities[entity_id]

            # Update access stats
            entity.last_accessed = time.time()
            entity.access_count += 1

            # Check confidence threshold
            if entity.confidence < query.min_confidence:
                continue

            # Simple relevance scoring based on name/description similarity
            relevance = self._calculate_entity_relevance(query, entity)
            if relevance > 0.1:  # Basic threshold
                candidates.append((relevance, entity))

        # Sort by relevance and return
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [entity for _, entity in candidates]

    def _find_relevant_relations(
        self, query: KnowledgeQuery, entities: list[KnowledgeEntity]
    ) -> list[KnowledgeRelation]:
        """Find relations relevant to the query and entities."""
        relevant_relations = []
        entity_ids = {entity.id for entity in entities}

        for entity_id in entity_ids:
            # Get outgoing relations
            for rel_id in self.relations_by_source.get(entity_id, set()):
                relation = self.relations[rel_id]
                if (
                    not query.relation_types
                    or relation.relation_type in query.relation_types
                ):
                    relevant_relations.append(relation)

            # Get incoming relations
            for rel_id in self.relations_by_target.get(entity_id, set()):
                relation = self.relations[rel_id]
                if (
                    not query.relation_types
                    or relation.relation_type in query.relation_types
                ):
                    relevant_relations.append(relation)

        return list(set(relevant_relations))  # Remove duplicates

    def _find_relevant_patterns(
        self, query: KnowledgeQuery
    ) -> list[PlanningPattern]:
        """Find planning patterns relevant to the query."""
        candidates = []

        # Get patterns for the domain
        domain_patterns = self.patterns_by_domain.get(query.domain, set())

        # Also include general patterns
        general_patterns = self.patterns_by_domain.get("general", set())

        pattern_ids = domain_patterns.union(general_patterns)

        for pattern_id in pattern_ids:
            pattern = self.patterns[pattern_id]

            # Score pattern relevance
            relevance = self._calculate_pattern_relevance(query, pattern)
            if relevance > 0.1:
                candidates.append((relevance, pattern))

        # Sort by relevance and success rate
        candidates.sort(
            key=lambda x: (x[0] * x[1].success_rate, x[1].usage_count),
            reverse=True,
        )

        return [pattern for _, pattern in candidates]

    def _calculate_entity_relevance(
        self, query: KnowledgeQuery, entity: KnowledgeEntity
    ) -> float:
        """Calculate relevance score for an entity."""
        score = 0.0
        query_words = set(query.goal.lower().split())

        # Name similarity
        entity_words = set(entity.name.lower().split())
        name_overlap = len(query_words.intersection(entity_words))
        score += name_overlap * 0.4

        # Description similarity
        desc_words = set(entity.description.lower().split())
        desc_overlap = len(query_words.intersection(desc_words))
        score += desc_overlap * 0.3

        # Context similarity
        context_words = set()
        for value in query.context.values():
            if isinstance(value, str):
                context_words.update(value.lower().split())

        prop_words = set()
        for value in entity.properties.values():
            if isinstance(value, str):
                prop_words.update(value.lower().split())

        context_overlap = len(context_words.intersection(prop_words))
        score += context_overlap * 0.2

        # Confidence and access frequency boost
        score *= entity.confidence
        score += min(
            entity.access_count * 0.01, 0.1
        )  # Small boost for frequently accessed

        return score

    def _calculate_pattern_relevance(
        self, query: KnowledgeQuery, pattern: PlanningPattern
    ) -> float:
        """Calculate relevance score for a planning pattern."""
        # Check if goal matches pattern template
        if self._is_similar_goal(query.goal, pattern.goal_template):
            base_score = 1.0
        else:
            # Fallback to word similarity
            query_words = set(query.goal.lower().split())
            pattern_words = set(pattern.pattern_name.lower().split())
            overlap = len(query_words.intersection(pattern_words))
            base_score = overlap * 0.3

        # Boost by success rate and usage
        score = base_score * (0.5 + 0.5 * pattern.success_rate)
        score += min(pattern.usage_count * 0.01, 0.2)

        return score

    def _calculate_relevance_scores(
        self,
        query: KnowledgeQuery,
        entities: list[KnowledgeEntity],
        patterns: list[PlanningPattern],
    ) -> dict[str, float]:
        """Calculate relevance scores for all results."""
        scores = {}

        for entity in entities:
            scores[f"entity_{entity.id}"] = self._calculate_entity_relevance(
                query, entity
            )

        for pattern in patterns:
            scores[f"pattern_{pattern.id}"] = (
                self._calculate_pattern_relevance(query, pattern)
            )

        return scores

    def get_statistics(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "patterns": len(self.patterns),
            "contexts": len(self.contexts),
            "entity_types": {
                entity_type.value: len(entity_ids)
                for entity_type, entity_ids in self.entity_by_type.items()
            },
            "pattern_domains": {
                domain: len(pattern_ids)
                for domain, pattern_ids in self.patterns_by_domain.items()
            },
            "average_pattern_success_rate": (
                sum(p.success_rate for p in self.patterns.values())
                / len(self.patterns)
                if self.patterns
                else 0.0
            ),
        }
