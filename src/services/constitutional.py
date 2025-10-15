"""Constitutional service for action evaluation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .base import BaseService
from ..governance import ConstitutionalReasoner

class ConstitutionalService(BaseService):
    """Constitutional evaluation service."""
    
    def __init__(self, config, registry):
        super().__init__(config, registry)
        self.reasoner: ConstitutionalReasoner = None
        self.evaluation_count = 0
        self.approval_rate = 0.0
    
    async def initialize(self) -> None:
        """Initialize constitutional reasoner."""
        try:
            self.reasoner = ConstitutionalReasoner()
            self._initialized = True
            self.logger.info("Constitutional service initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize constitutional service: {e}")
            raise
    
    async def evaluate_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate an action against constitutional principles."""
        if not self.reasoner:
            await self.initialize()
        
        try:
            approved, reasoning = await self.reasoner.evaluate_action(action, context)
            
            # Update statistics
            self.evaluation_count += 1
            if approved:
                self.approval_rate = ((self.approval_rate * (self.evaluation_count - 1)) + 1.0) / self.evaluation_count
            else:
                self.approval_rate = (self.approval_rate * (self.evaluation_count - 1)) / self.evaluation_count
            
            # Emit evaluation event
            event_bus = self.get_service("event_bus")
            if event_bus:
                await event_bus.emit("constitutional_evaluation", {
                    "approved": approved,
                    "reasoning": reasoning,
                    "action_type": action.get("type", "unknown"),
                    "evaluation_count": self.evaluation_count
                })
            
            return approved, reasoning
            
        except Exception as e:
            self.logger.error(f"Constitutional evaluation error: {e}")
            # Fail-safe: reject on error
            return False, f"Evaluation failed: {str(e)}"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check constitutional service health."""
        base_health = await super().health_check()
        
        return {
            **base_health,
            "evaluations_performed": self.evaluation_count,
            "approval_rate": self.approval_rate,
            "reasoner_available": self.reasoner is not None
        }