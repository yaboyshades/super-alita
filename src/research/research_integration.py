"""Integration manager for research components with production system."""

import asyncio
from typing import Dict, List, Any
import logging
from datetime import datetime

class ResearchIntegrationManager:
    """Integrate research components with production Super Alita system."""
    
    def __init__(self, constitutional_llm, memory_system, event_bus, knowledge_graph):
        self.constitutional_llm = constitutional_llm
        self.memory_system = memory_system
        self.event_bus = event_bus
        self.knowledge_graph = knowledge_graph
        self.logger = logging.getLogger(__name__)
        
        # Research tracking
        self.research_metrics = {
            "constitutional_llm_usage": 0,
            "memory_consolidation_cycles": 0,
            "research_experiments": 0,
            "performance_improvements": []
        }
    
    async def initialize_research_system(self):
        """Initialize the integrated research system."""
        self.logger.info("Initializing Research Integration System...")
        
        # Subscribe to relevant events
        await self.event_bus.subscribe("constitutional.evaluation", self.handle_constitutional_evaluation, "research")
        await self.event_bus.subscribe("agent.session.completed", self.handle_agent_session, "research")
        
        # Start periodic memory consolidation
        asyncio.create_task(self.periodic_memory_consolidation())
        
        self.logger.info("Research Integration System initialized")
    
    async def handle_constitutional_evaluation(self, event: Dict[str, Any]):
        """Handle constitutional evaluation events for research."""
        try:
            # Store evaluation in memory for learning
            evaluation_content = f"Constitutional Evaluation: {event['data']}"
            await self.memory_system.store_experience(
                evaluation_content,
                {
                    **event,
                    "event_type": "constitutional_evaluation",
                    "timestamp": event.get("timestamp", datetime.now().isoformat()),
                    "complexity": "high"  # Constitutional decisions are important
                },
                memory_type="episodic"
            )
            
            # Track LLM usage if applicable
            if "LLM Evaluation" in event['data'].get("reasoning", ""):
                self.research_metrics["constitutional_llm_usage"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to handle constitutional evaluation: {e}")
    
    async def handle_agent_session(self, event: Dict[str, Any]):
        """Handle completed agent sessions for research."""
        try:
            session_data = event['data']
            
            # Store session in memory
            session_content = f"Agent Session: {session_data.get('goal')} - Success: {session_data.get('success')}"
            await self.memory_system.store_experience(
                session_content,
                {
                    "session_id": session_data.get("session_id"),
                    "success": session_data.get("success"),
                    "iterations": session_data.get("iterations"),
                    "reflections": session_data.get("reflections", []),
                },
                memory_type="episodic"
            )
            
            # Extract learning points from reflections
            for reflection in session_data.get("reflections", []):
                await self.extract_learning_from_reflection(reflection, session_data)
                
        except Exception as e:
            self.logger.error(f"Failed to handle agent session: {e}")
    
    async def extract_learning_from_reflection(self, reflection: str, session_data: Dict[str, Any]):
        """Extract learning points from session reflections."""
        # Simple pattern matching for now - could be enhanced with NLP
        learning_keywords = {
            "learned": "knowledge_acquisition",
            "discovered": "discovery", 
            "realized": "insight",
            "improved": "optimization",
            "better": "improvement"
        }
        
        for keyword, category in learning_keywords.items():
            if keyword in reflection.lower():
                learning_content = f"Learning ({category}): {reflection}"
                await self.memory_system.store_experience(
                    learning_content,
                    {
                        "session_id": session_data.get("session_id"),
                        "category": category,
                        "source": "reflection"
                    },
                    memory_type="semantic"
                )
                break
    
    async def periodic_memory_consolidation(self, interval_hours: int = 24):
        """Run periodic memory consolidation."""
        while True:
            try:
                await asyncio.sleep(interval_hours * 3600)  # Wait specified hours
                
                self.logger.info("Running periodic memory consolidation...")
                decisions = await self.memory_system.run_consolidation_cycle()
                
                self.research_metrics["memory_consolidation_cycles"] += 1
                
                # Log consolidation results
                await self.event_bus.emit({
                    "type": "research.memory_consolidation",
                    "data": {
                        "cycle_count": self.research_metrics["memory_consolidation_cycles"],
                        "decisions_made": len(decisions),
                        "timestamp": datetime.now().isoformat()
                    },
                    "source": "research_system"
                })
                
            except Exception as e:
                self.logger.error(f"Periodic memory consolidation failed: {e}")
    
    async def run_research_experiment(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a research experiment and track results."""
        self.research_metrics["research_experiments"] += 1
        
        try:
            experiment_type = experiment_config.get("type")
            
            if experiment_type == "constitutional_llm_ab_test":
                results = await self._run_constitutional_ab_test(experiment_config)
            elif experiment_type == "memory_retention_test":
                results = await self._run_memory_retention_test(experiment_config)
            else:
                results = {"error": f"Unknown experiment type: {experiment_type}"}
            
            # Store experiment results
            await self.knowledge_graph.create_atom(
                "research_experiment",
                {
                    "config": experiment_config,
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "experiment_type": experiment_type,
                    "success": "error" not in results
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Research experiment failed: {e}")
            return {"error": str(e)}
    
    async def _run_constitutional_ab_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run A/B test comparing constitutional LLM vs rule-based system."""
        test_cases = config.get("test_cases", [])
        
        llm_decisions = []
        rule_decisions = []
        human_judgments = config.get("human_judgments", [])
        
        for i, test_case in enumerate(test_cases):
            # LLM evaluation
            llm_approved, llm_reasoning = await self.constitutional_llm.evaluate_action(
                test_case["action"], test_case["context"]
            )
            
            # Rule-based evaluation using the underlying rule engine
            rule_approved, rule_reasoning = await self.constitutional_llm.rule_engine.evaluate_action(
                test_case["action"], test_case["context"]
            )
            
            llm_decisions.append(llm_approved)
            rule_decisions.append(rule_approved)
        
        # Calculate metrics
        metrics = await self._calculate_ab_test_metrics(llm_decisions, rule_decisions, human_judgments)
        
        return {
            "experiment_type": "constitutional_llm_ab_test",
            "test_cases_count": len(test_cases),
            "metrics": metrics,
            "llm_usage_stats": self.constitutional_llm.performance_metrics
        }
    
    async def _run_memory_retention_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test memory retention and consolidation effectiveness."""
        # Create test memories with known relationships
        test_memories = config.get("test_memories", [])
        
        # Store test memories
        memory_ids = []
        for memory in test_memories:
            memory_id = await self.memory_system.store_experience(
                memory["content"],
                memory.get("context", {}),
                memory.get("memory_type", "working")
            )
            memory_ids.append(memory_id)
        
        # Wait for consolidation (simulated)
        await asyncio.sleep(1)
        
        # Run consolidation
        decisions = await self.memory_system.run_consolidation_cycle()
        
        # Test retrieval after consolidation
        retrieval_tests = config.get("retrieval_tests", [])
        retrieval_results = []
        
        for test in retrieval_tests:
            retrieved = await self.memory_system.retrieve_relevant_memories(
                test["query"], test.get("context", {})
            )
            retrieval_results.append({
                "query": test["query"],
                "retrieved_count": len(retrieved),
                "expected_memories": test.get("expected_memories", [])
            })
        
        return {
            "experiment_type": "memory_retention_test",
            "memory_count": len(test_memories),
            "consolidation_decisions": len(decisions),
            "retrieval_results": retrieval_results
        }
    
    async def _calculate_ab_test_metrics(self, llm_decisions: List[bool], 
                                       rule_decisions: List[bool], 
                                       human_judgments: List[bool]) -> Dict[str, float]:
        """Calculate A/B test metrics comparing LLM vs rule-based decisions."""
        if not human_judgments:
            return {"error": "Human judgments required for A/B test"}
        
        # Agreement rates
        llm_agreement = sum(1 for llm, human in zip(llm_decisions, human_judgments) if llm == human) / len(human_judgments)
        rule_agreement = sum(1 for rule, human in zip(rule_decisions, human_judgments) if rule == human) / len(human_judgments)
        
        # Improvement over baseline
        improvement = llm_agreement - rule_agreement
        
        # Edge case performance (assume last few are edge cases)
        edge_case_count = min(5, len(human_judgments))
        llm_edge_performance = sum(1 for llm, human in zip(llm_decisions[-edge_case_count:], human_judgments[-edge_case_count:]) if llm == human) / edge_case_count
        rule_edge_performance = sum(1 for rule, human in zip(rule_decisions[-edge_case_count:], human_judgments[-edge_case_count:]) if rule == human) / edge_case_count
        
        return {
            "llm_agreement_rate": llm_agreement,
            "rule_agreement_rate": rule_agreement,
            "improvement_over_baseline": improvement,
            "llm_edge_case_performance": llm_edge_performance,
            "rule_edge_case_performance": rule_edge_performance,
            "total_test_cases": len(human_judgments)
        }
    
    async def get_research_metrics(self) -> Dict[str, Any]:
        """Get comprehensive research metrics."""
        return {
            "research_operations": self.research_metrics,
            "constitutional_llm_metrics": self.constitutional_llm.performance_metrics,
            "memory_system_metrics": self.memory_system.consolidator.consolidation_stats
        }

# Factory function
async def create_research_integration_manager(constitutional_llm, memory_system, event_bus, knowledge_graph):
    """Create and initialize research integration manager."""
    manager = ResearchIntegrationManager(constitutional_llm, memory_system, event_bus, knowledge_graph)
    await manager.initialize_research_system()
    return manager