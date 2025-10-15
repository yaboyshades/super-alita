"""Bridge between agent operations and knowledge graph memory."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

class MemoryBridge:
    """Helper for agent-knowledge graph memory integration."""
    
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.logger = logging.getLogger(__name__)
    
    async def write_interaction(self, interaction_type: str, data: Dict[str, Any], 
                               context: Dict[str, Any] = None) -> str:
        """Write agent interaction to knowledge graph."""
        try:
            # Create memory atom
            atom = await self.knowledge_graph.create_atom(
                atom_type=f"agent_{interaction_type}",
                content=data,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "interaction_type": interaction_type,
                    **(context or {})
                }
            )
            
            self.logger.info(f"Stored {interaction_type} interaction: {atom['id']}")
            return atom["id"]
            
        except Exception as e:
            self.logger.error(f"Failed to write interaction: {e}")
            return f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def fetch_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch relevant context from knowledge graph."""
        try:
            if hasattr(self.knowledge_graph, 'semantic_search'):
                results = await self.knowledge_graph.semantic_search(query, limit=limit)
                return results or []
            else:
                # Fallback for simple KG
                return await self.knowledge_graph.retrieve_relevant_context(query)
                
        except Exception as e:
            self.logger.error(f"Failed to fetch context: {e}")
            return []
    
    async def store_reflection(self, session_id: str, reflection: str, 
                             success: bool, artifacts: Dict[str, Any] = None) -> str:
        """Store agent reflection in knowledge graph."""
        try:
            reflection_data = {
                "session_id": session_id,
                "reflection": reflection,
                "success": success,
                "artifacts": artifacts or {},
                "timestamp": datetime.now().isoformat()
            }
            
            atom = await self.knowledge_graph.create_atom(
                "agent_reflection",
                reflection_data,
                metadata={
                    "session_id": session_id,
                    "success": success,
                    "type": "reflection"
                }
            )
            
            return atom["id"]
            
        except Exception as e:
            self.logger.error(f"Failed to store reflection: {e}")
            return f"reflection_error_{session_id}"
    
    async def get_similar_successes(self, goal: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get similar successful interactions for context."""
        try:
            # Query for successful agent sessions
            results = await self.fetch_context(f"successful {goal}", limit=limit)
            
            # Filter for successful interactions
            successes = []
            for result in results:
                if isinstance(result, dict) and result.get("success", False):
                    successes.append(result)
            
            return successes[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get similar successes: {e}")
            return []
    
    async def learn_from_failure(self, session_id: str, goal: str, failure_reason: str, 
                               attempted_actions: List[Dict[str, Any]]) -> str:
        """Learn from failed interactions."""
        try:
            failure_data = {
                "session_id": session_id,
                "goal": goal,
                "failure_reason": failure_reason,
                "attempted_actions": attempted_actions,
                "timestamp": datetime.now().isoformat(),
                "type": "failure_analysis"
            }
            
            atom = await self.knowledge_graph.create_atom(
                "agent_failure",
                failure_data,
                metadata={
                    "session_id": session_id,
                    "goal_type": self._categorize_goal(goal),
                    "failure_category": self._categorize_failure(failure_reason)
                }
            )
            
            self.logger.info(f"Stored failure analysis: {atom['id']}")
            return atom["id"]
            
        except Exception as e:
            self.logger.error(f"Failed to store failure analysis: {e}")
            return f"failure_error_{session_id}"
    
    def _categorize_goal(self, goal: str) -> str:
        """Categorize goal type for better organization."""
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ["code", "program", "function"]):
            return "programming"
        elif any(word in goal_lower for word in ["data", "analyze", "report"]):
            return "analysis"
        elif any(word in goal_lower for word in ["create", "build", "generate"]):
            return "creation"
        else:
            return "general"
    
    def _categorize_failure(self, failure_reason: str) -> str:
        """Categorize failure type for pattern recognition."""
        failure_lower = failure_reason.lower()
        
        if "constitutional" in failure_lower:
            return "constitutional_violation"
        elif any(word in failure_lower for word in ["timeout", "time"]):
            return "timeout"
        elif "permission" in failure_lower or "auth" in failure_lower:
            return "authorization"
        else:
            return "execution_error"