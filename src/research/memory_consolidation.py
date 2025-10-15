"""Memory consolidation pipeline using transformer attention mechanisms."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class MemoryItem:
    id: str
    content: str
    embedding: np.ndarray
    timestamp: datetime
    access_count: int
    last_accessed: datetime
    importance_score: float
    memory_type: str  # episodic, semantic, working
    metadata: Dict[str, Any]

@dataclass 
class ConsolidationDecision:
    item_id: str
    action: str  # retain, consolidate, forget
    confidence: float
    reasoning: str
    new_importance: float

class TransformerMemoryConsolidator:
    """Transformer-based memory consolidation using attention mechanisms."""
    
    def __init__(self, knowledge_graph, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.knowledge_graph = knowledge_graph
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Memory management parameters
        self.working_memory_limit = 100  # Items in working memory
        self.consolidation_threshold = 0.7  # Importance threshold for consolidation
        self.forgetting_threshold = 0.2  # Importance threshold for forgetting
        
        # Attention-based importance model
        self.importance_model = self._build_importance_model()
        
        # Tracking
        self.consolidation_stats = {
            "total_items_processed": 0,
            "items_consolidated": 0,
            "items_forgotten": 0,
            "items_retained": 0
        }
    
    def _build_importance_model(self) -> nn.Module:
        """Build a simple importance scoring model."""
        return nn.Sequential(
            nn.Linear(384, 256),  # Input dimension from sentence transformer
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    async def consolidate_memory(self, working_memory: List[MemoryItem], 
                               episodic_memory: List[MemoryItem]) -> Tuple[List[MemoryItem], List[ConsolidationDecision]]:
        """Run memory consolidation process."""
        self.logger.info("Starting memory consolidation...")
        
        all_items = working_memory + episodic_memory
        decisions = []
        
        for item in all_items:
            decision = await self._evaluate_memory_item(item, all_items)
            decisions.append(decision)
            
            # Execute decision
            await self._execute_consolidation_decision(decision, item)
        
        # Update working memory - keep most important items
        updated_working_memory = await self._update_working_memory(all_items, decisions)
        
        self.logger.info(f"Consolidation complete: {self.consolidation_stats}")
        
        return updated_working_memory, decisions
    
    async def _evaluate_memory_item(self, item: MemoryItem, all_items: List[MemoryItem]) -> ConsolidationDecision:
        """Evaluate a memory item for consolidation decisions."""
        # Calculate updated importance score
        importance_score = await self._calculate_importance_score(item, all_items)
        
        # Determine action based on importance and memory type
        if item.memory_type == "working":
            if importance_score >= self.consolidation_threshold:
                action = "consolidate"
                reasoning = "High importance working memory item promoted to semantic memory"
            elif importance_score <= self.forgetting_threshold:
                action = "forget"
                reasoning = "Low importance working memory item forgotten"
            else:
                action = "retain"
                reasoning = "Moderate importance working memory item retained"
        else:  # episodic or semantic memory
            if importance_score <= self.forgetting_threshold:
                action = "forget"
                reasoning = "Low importance long-term memory item forgotten"
            else:
                action = "retain"
                reasoning = "Long-term memory item retained"
        
        confidence = min(1.0, importance_score + 0.2)  # Base confidence on importance
        
        return ConsolidationDecision(
            item_id=item.id,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            new_importance=importance_score
        )
    
    async def _calculate_importance_score(self, item: MemoryItem, all_items: List[MemoryItem]) -> float:
        """Calculate importance score using transformer attention and contextual factors."""
        base_importance = item.importance_score
        
        # Recency factor - exponential decay
        recency_factor = self._calculate_recency_factor(item)
        
        # Frequency factor
        frequency_factor = min(1.0, item.access_count / 10.0)
        
        # Relevance factor - semantic similarity to current context
        relevance_factor = await self._calculate_relevance_factor(item, all_items)
        
        # Novelty factor - information uniqueness
        novelty_factor = await self._calculate_novelty_factor(item, all_items)
        
        # Combined importance score
        importance = (0.3 * base_importance + 
                     0.2 * recency_factor + 
                     0.2 * frequency_factor + 
                     0.2 * relevance_factor + 
                     0.1 * novelty_factor)
        
        return min(1.0, max(0.0, importance))
    
    def _calculate_recency_factor(self, item: MemoryItem) -> float:
        """Calculate recency factor with exponential decay."""
        hours_since_access = (datetime.now() - item.last_accessed).total_seconds() / 3600
        decay_rate = 0.1  # 10% decay per hour
        return max(0.0, 1.0 - decay_rate * hours_since_access)
    
    async def _calculate_relevance_factor(self, item: MemoryItem, all_items: List[MemoryItem]) -> float:
        """Calculate relevance based on semantic similarity to recent items."""
        if len(all_items) < 2:
            return 0.5
        
        # Get embeddings for recent items (last 10)
        recent_items = sorted(all_items, key=lambda x: x.last_accessed, reverse=True)[:10]
        recent_embeddings = [item.embedding for item in recent_items]
        
        if not recent_embeddings:
            return 0.5
        
        # Calculate average similarity to recent items
        similarities = []
        for recent_embedding in recent_embeddings:
            similarity = self._cosine_similarity(item.embedding, recent_embedding)
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    async def _calculate_novelty_factor(self, item: MemoryItem, all_items: List[MemoryItem]) -> float:
        """Calculate novelty based on uniqueness of information."""
        if len(all_items) < 2:
            return 1.0  # First item is maximally novel
        
        # Compare with all other items
        similarities = []
        for other_item in all_items:
            if other_item.id != item.id:
                similarity = self._cosine_similarity(item.embedding, other_item.embedding)
                similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities)
        return 1.0 - max_similarity
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _execute_consolidation_decision(self, decision: ConsolidationDecision, item: MemoryItem):
        """Execute the consolidation decision for a memory item."""
        self.consolidation_stats["total_items_processed"] += 1
        
        if decision.action == "consolidate":
            await self._consolidate_to_semantic_memory(item)
            self.consolidation_stats["items_consolidated"] += 1
        elif decision.action == "forget":
            await self._forget_memory_item(item)
            self.consolidation_stats["items_forgotten"] += 1
        else:  # retain
            self.consolidation_stats["items_retained"] += 1
        
        # Update item importance
        item.importance_score = decision.new_importance
        item.last_accessed = datetime.now()
    
    async def _consolidate_to_semantic_memory(self, item: MemoryItem):
        """Consolidate item from working/episodic to semantic memory."""
        try:
            # Extract key concepts and relationships
            concepts = await self._extract_semantic_concepts(item.content)
            
            # Create semantic atoms in knowledge graph
            for concept in concepts:
                atom = await self.knowledge_graph.create_atom(
                    "semantic_concept",
                    concept["content"],
                    metadata={
                        "source_memory_id": item.id,
                        "consolidated_at": datetime.now().isoformat(),
                        "confidence": concept["confidence"],
                        "category": concept["category"]
                    }
                )
                
                # Create bonds between related concepts
                await self._create_semantic_bonds(atom, concepts)
            
            self.logger.info(f"Consolidated memory item {item.id} to semantic concepts")
            
        except Exception as e:
            self.logger.error(f"Failed to consolidate memory item {item.id}: {e}")
    
    async def _extract_semantic_concepts(self, content: str) -> List[Dict[str, Any]]:
        """Extract semantic concepts from memory content."""
        concepts = []
        
        # Simple concept extraction - in practice, use NER or similar
        sentences = content.split(".")
        for sentence in sentences[:5]:  # Limit to first 5 sentences
            if len(sentence.strip()) > 10:
                concepts.append({
                    "content": sentence.strip(),
                    "confidence": 0.8,
                    "category": "extracted_concept"
                })
        
        return concepts
    
    async def _create_semantic_bonds(self, source_atom: Dict[str, Any], concepts: List[Dict[str, Any]]):
        """Create semantic bonds between related concepts."""
        for target_concept in concepts:
            if target_concept["content"] != source_atom["content"]:
                # Calculate semantic similarity
                similarity = await self._calculate_concept_similarity(
                    source_atom["content"], target_concept["content"]
                )
                
                if similarity > 0.6:  # Threshold for creating bonds
                    # This would create a bond in the knowledge graph
                    # Implementation depends on KG structure
                    pass
    
    async def _calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate semantic similarity between two concepts."""
        # Generate embeddings and compute similarity
        inputs1 = self.tokenizer(concept1, return_tensors="pt", padding=True, truncation=True)
        inputs2 = self.tokenizer(concept2, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs1 = self.model(**inputs1)
            outputs2 = self.model(**inputs2)
        
        # Use mean pooling
        embedding1 = outputs1.last_hidden_state.mean(dim=1).numpy()[0]
        embedding2 = outputs2.last_hidden_state.mean(dim=1).numpy()[0]
        
        return self._cosine_similarity(embedding1, embedding2)
    
    async def _forget_memory_item(self, item: MemoryItem):
        """Forget a memory item - remove from active memory."""
        # In a real implementation, this might move the item to archival storage
        # or simply mark it as inactive
        self.logger.info(f"Forgot memory item {item.id}")
    
    async def _update_working_memory(self, all_items: List[MemoryItem], 
                                   decisions: List[ConsolidationDecision]) -> List[MemoryItem]:
        """Update working memory based on consolidation decisions."""
        # Filter items that should be in working memory
        working_memory_items = []
        
        for item, decision in zip(all_items, decisions):
            if (decision.action in ["retain", "consolidate"] and 
                item.memory_type == "working" and 
                decision.new_importance > self.forgetting_threshold):
                working_memory_items.append(item)
        
        # Sort by importance and limit size
        working_memory_items.sort(key=lambda x: x.importance_score, reverse=True)
        return working_memory_items[:self.working_memory_limit]
    
    async def evaluate_consolidation_effectiveness(self, test_sessions: List[List[MemoryItem]]) -> Dict[str, float]:
        """Evaluate memory consolidation effectiveness."""
        retention_rates = []
        consolidation_rates = []
        
        for session in test_sessions:
            # Run consolidation
            _, decisions = await self.consolidate_memory(session, [])
            
            # Calculate metrics
            retained_count = sum(1 for d in decisions if d.action != "forget")
            consolidated_count = sum(1 for d in decisions if d.action == "consolidate")
            
            retention_rate = retained_count / len(decisions) if decisions else 0
            consolidation_rate = consolidated_count / len(decisions) if decisions else 0
            
            retention_rates.append(retention_rate)
            consolidation_rates.append(consolidation_rate)
        
        return {
            "average_retention_rate": np.mean(retention_rates),
            "average_consolidation_rate": np.mean(consolidation_rates),
            "retention_std": np.std(retention_rates),
            "test_sessions": len(test_sessions)
        }

class UnifiedMemorySystem:
    """Unified memory system integrating working, episodic, and semantic memory."""
    
    def __init__(self, knowledge_graph, consolidator: TransformerMemoryConsolidator):
        self.knowledge_graph = knowledge_graph
        self.consolidator = consolidator
        self.logger = logging.getLogger(__name__)
        
        # Memory stores
        self.working_memory: List[MemoryItem] = []
        self.episodic_memory: List[MemoryItem] = []
        
        # Memory access patterns
        self.access_patterns = defaultdict(int)
    
    async def store_experience(self, content: str, context: Dict[str, Any], 
                             memory_type: str = "working") -> str:
        """Store a new experience in memory."""
        # Generate embedding
        embedding = await self._generate_embedding(content)
        
        # Create memory item
        item_id = f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        memory_item = MemoryItem(
            id=item_id,
            content=content,
            embedding=embedding,
            timestamp=datetime.now(),
            access_count=1,
            last_accessed=datetime.now(),
            importance_score=await self._estimate_initial_importance(content, context),
            memory_type=memory_type,
            metadata={**context, "source": "agent_experience"}
        )
        
        # Store in appropriate memory
        if memory_type == "working":
            self.working_memory.append(memory_item)
        else:
            self.episodic_memory.append(memory_item)
        
        self.logger.info(f"Stored experience in {memory_type} memory: {item_id}")
        return item_id
    
    async def retrieve_relevant_memories(self, query: str, context: Dict[str, Any], 
                                       limit: int = 10) -> List[MemoryItem]:
        """Retrieve memories relevant to the current context."""
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Search across all memory types
        all_memories = self.working_memory + self.episodic_memory
        
        relevant_memories = []
        for memory in all_memories:
            similarity = self.consolidator._cosine_similarity(query_embedding, memory.embedding)
            if similarity > 0.6:  # Relevance threshold
                # Update access stats
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                relevant_memories.append((memory, similarity))
        
        # Sort by similarity and return top results
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, similarity in relevant_memories[:limit]]
    
    async def run_consolidation_cycle(self):
        """Run a memory consolidation cycle."""
        self.logger.info("Starting memory consolidation cycle...")
        
        updated_working_memory, decisions = await self.consolidator.consolidate_memory(
            self.working_memory, self.episodic_memory
        )
        
        # Update memory stores
        self.working_memory = updated_working_memory
        
        # Log consolidation results
        consolidation_stats = self.consolidator.consolidation_stats
        self.logger.info(f"Consolidation cycle completed: {consolidation_stats}")
        
        return decisions
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using the transformer model."""
        inputs = self.consolidator.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.consolidator.model(**inputs)
        
        # Use mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        return embedding
    
    async def _estimate_initial_importance(self, content: str, context: Dict[str, Any]) -> float:
        """Estimate initial importance of a memory item."""
        base_importance = 0.5
        
        # Context-based adjustments
        if context.get("risk_level") == "high":
            base_importance += 0.3
        if context.get("user_importance") == "high":
            base_importance += 0.2
        if context.get("task_critical", False):
            base_importance += 0.2
        
        # Content-based adjustments
        content_length = len(content.split())
        if content_length > 100:
            base_importance += 0.1
        elif content_length < 10:
            base_importance -= 0.1
        
        return min(1.0, max(0.0, base_importance))
    
    async def evaluate_memory_performance(self, test_queries: List[Tuple[str, List[str]]]) -> Dict[str, float]:
        """Evaluate memory system performance."""
        recall_scores = []
        precision_scores = []
        
        for query, expected_memory_ids in test_queries:
            # Retrieve memories
            retrieved_memories = await self.retrieve_relevant_memories(query, {})
            retrieved_ids = [memory.id for memory in retrieved_memories]
            
            # Calculate recall and precision
            relevant_retrieved = set(retrieved_ids) & set(expected_memory_ids)
            recall = len(relevant_retrieved) / len(expected_memory_ids) if expected_memory_ids else 0
            precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
            
            recall_scores.append(recall)
            precision_scores.append(precision)
        
        return {
            "average_recall": np.mean(recall_scores),
            "average_precision": np.mean(precision_scores),
            "f1_score": (2 * np.mean(recall_scores) * np.mean(precision_scores)) / 
                       (np.mean(recall_scores) + np.mean(precision_scores)) if (np.mean(recall_scores) + np.mean(precision_scores)) > 0 else 0
        }

# Factory functions
async def create_memory_consolidator(knowledge_graph) -> TransformerMemoryConsolidator:
    """Create and initialize a memory consolidator."""
    return TransformerMemoryConsolidator(knowledge_graph)

async def create_unified_memory_system(knowledge_graph) -> UnifiedMemorySystem:
    """Create and initialize a unified memory system."""
    consolidator = await create_memory_consolidator(knowledge_graph)
    return UnifiedMemorySystem(knowledge_graph, consolidator)