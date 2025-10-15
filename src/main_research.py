"""Super Alita Research Edition main application."""

import asyncio
import logging
from typing import Dict, Any

# Research imports
from research.constitutional_llm import create_constitutional_llm
from research.memory_consolidation import create_unified_memory_system
from research.research_integration import create_research_integration_manager

class SuperAlitaResearchApplication:
    """Super Alita with integrated research capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core production components
        self.constitutional_reasoner = None
        self.security_manager = None
        self.llm_orchestrator = None
        self.knowledge_graph = None
        self.event_bus = None
        
        # Research components
        self.constitutional_llm = None
        self.memory_system = None
        self.research_manager = None
    
    async def initialize(self):
        """Initialize production system with research capabilities."""
        self.logger.info("Initializing Super Alita Research Edition...")
        
        try:
            # Initialize production components (from previous implementation)
            await self.initialize_production_components()
            
            # Initialize research components
            await self.initialize_research_components()
            
            self.logger.info("✅ Super Alita Research Edition fully initialized!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Research Edition: {e}")
            raise
    
    async def initialize_production_components(self):
        """Initialize core production components."""
        # These would be the same implementations from before
        # self.constitutional_reasoner = await create_constitutional_reasoner()
        # self.security_manager = await create_security_manager(self.config["redis_url"])
        # self.llm_orchestrator = await create_llm_orchestrator()
        # self.knowledge_graph = await create_knowledge_graph(self.config["chromadb_path"])
        # self.event_bus = await create_event_bus(self.config["redis_url"])
        
        # For demo purposes, create minimal versions
        from src.governance.constitutional_reasoner import ConstitutionalReasoner
        self.constitutional_reasoner = ConstitutionalReasoner()
        
        # Use simple implementations for demo
        self.knowledge_graph = type("SimpleKG", (), {
            "create_atom": lambda self, atom_type, content, metadata=None: {"id": f"atom_{len(getattr(self, '_atoms', []))}"},
            "semantic_search": lambda self, query, **kwargs: []
        })()
        
        self.event_bus = type("SimpleEventBus", (), {
            "emit": lambda self, event: asyncio.sleep(0),
            "subscribe": lambda self, event_type, handler, source: asyncio.sleep(0)
        })()
        
        self.logger.info("✅ Production components initialized")
    
    async def initialize_research_components(self):
        """Initialize research components."""
        # Constitutional LLM (Milestone 1)
        self.constitutional_llm = await create_constitutional_llm(
            rule_engine=self.constitutional_reasoner,
            base_model=self.config.get("constitutional_llm_model", "microsoft/DialoGPT-medium")
        )
        
        # Memory consolidation system (Milestone 2)
        self.memory_system = await create_unified_memory_system(self.knowledge_graph)
        
        # Research integration manager
        self.research_manager = await create_research_integration_manager(
            self.constitutional_llm, self.memory_system, self.event_bus, self.knowledge_graph
        )
        
        self.logger.info("✅ Research components initialized")
    
    async def process_user_request_research(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user request with research enhancements."""
        # Use constitutional LLM instead of basic reasoner
        action = {
            "type": "user_request",
            "description": user_input,
            "context": context
        }
        
        approved, reasoning = await self.constitutional_llm.evaluate_action(action, context)
        
        if not approved:
            return {
                "error": "Constitutional violation",
                "reasoning": reasoning
            }
        
        # Store interaction in memory for learning
        await self.memory_system.store_experience(
            f"User request: {user_input}",
            context,
            memory_type="working"
        )
        
        # Continue with normal processing...
        # llm_response = await self.llm_orchestrator.generate(user_input)
        
        return {
            "response": "Research-enhanced response processed",  # llm_response.content
            "constitutional_approved": approved,
            "constitutional_reasoning": reasoning,
            "llm_provider": "research_llm",  # llm_response.provider
            "research_enhanced": True
        }
    
    async def run_research_demo(self):
        """Run a demonstration of research capabilities."""
        self.logger.info("Starting research capabilities demonstration...")
        
        # Test constitutional LLM
        test_action = {
            "type": "code_generation",
            "description": "Create a function that deletes user data without confirmation"
        }
        test_context = {
            "user_intent": "data management utility",
            "risk_level": "high"
        }
        
        approved, reasoning = await self.constitutional_llm.evaluate_action(test_action, test_context)
        self.logger.info(f"Constitutional LLM Test - Approved: {approved}, Reasoning: {reasoning}")
        
        # Test memory system
        memory_id = await self.memory_system.store_experience(
            "User prefers detailed explanations with code examples",
            {
                "user_id": "demo_user",
                "preference_type": "communication_style"
            },
            memory_type="working"
        )
        self.logger.info(f"Stored memory with ID: {memory_id}")
        
        # Retrieve relevant memories
        relevant_memories = await self.memory_system.retrieve_relevant_memories(
            "user communication preferences", {}
        )
        self.logger.info(f"Retrieved {len(relevant_memories)} relevant memories")
        
        return {
            "constitutional_test": {"approved": approved, "reasoning": reasoning},
            "memory_test": {"stored_memory": memory_id, "retrieved_count": len(relevant_memories)},
            "research_metrics": await self.research_manager.get_research_metrics()
        }

# Configuration for research edition
RESEARCH_CONFIG = {
    "redis_url": "redis://localhost:6379",
    "chromadb_path": "./data/chromadb_research",
    "event_backup_path": "./data/event_backup_research.jsonl",
    "constitutional_llm_model": "microsoft/DialoGPT-medium"
}

async def main():
    """Main research demonstration."""
    logging.basicConfig(level=logging.INFO)
    
    app = SuperAlitaResearchApplication(RESEARCH_CONFIG)
    await app.initialize()
    
    # Run research demo
    demo_results = await app.run_research_demo()
    print("Research Demo Results:", demo_results)
    
    # Example user request with research enhancements
    result = await app.process_user_request_research(
        "How can I implement secure authentication in Python?",
        {
            "user_intent": "programming_help",
            "risk_level": "low"
        }
    )
    print("Enhanced Request Result:", result)

if __name__ == "__main__":
    asyncio.run(main())