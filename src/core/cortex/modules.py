"""
Cortex Modules: Pluggable components for perception, reasoning, and action
Implements the modular architecture for cognitive processing
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .markers import CortexPhase, PerformanceTracker

T = TypeVar('T')


class ModuleResult(Generic[T]):
    """Result container for module execution"""
    
    def __init__(
        self, 
        data: T, 
        success: bool = True, 
        error: str | None = None,
        metrics: dict[str, Any] | None = None
    ):
        self.data = data
        self.success = success
        self.error = error
        self.metrics = metrics or {}
        self.timestamp = time.time()


@dataclass
class CortexInput:
    """Input data for Cortex processing cycle"""
    raw_data: Any
    context: dict[str, Any]
    metadata: dict[str, Any]
    cycle_id: str


@dataclass 
class PerceptionResult:
    """Result from perception phase"""
    processed_data: Any
    features: dict[str, Any]
    confidence: float
    attention_weights: dict[str, float] | None = None


@dataclass
class ReasoningResult:
    """Result from reasoning phase"""
    analysis: dict[str, Any]
    conclusions: list[str]
    confidence: float
    reasoning_chain: list[dict[str, Any]] | None = None


@dataclass
class ActionResult:
    """Result from action phase"""
    actions: list[dict[str, Any]]
    priority_scores: dict[str, float]
    execution_plan: dict[str, Any] | None = None


class CortexModule(ABC):
    """Base class for all Cortex modules"""
    
    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}
        self.enabled = True
        self.performance_tracker: PerformanceTracker | None = None
    
    @abstractmethod
    async def process(self, input_data: Any, context: dict[str, Any]) -> ModuleResult:
        """Process input and return result"""
        pass
    
    @abstractmethod
    def get_phase(self) -> CortexPhase:
        """Get the Cortex phase this module belongs to"""
        pass
    
    def set_performance_tracker(self, tracker: PerformanceTracker):
        """Set performance tracker for metrics collection"""
        self.performance_tracker = tracker
    
    async def execute_with_tracking(
        self, 
        input_data: Any, 
        context: dict[str, Any]
    ) -> ModuleResult:
        """Execute module with performance tracking"""
        start_time = time.time()
        
        try:
            result = await self.process(input_data, context)
            
            # Track successful execution
            if self.performance_tracker:
                duration_ms = (time.time() - start_time) * 1000
                self.performance_tracker.track_module_execution(
                    module_name=self.name,
                    phase=self.get_phase(),
                    duration_ms=duration_ms,
                    metrics=result.metrics
                )
            
            return result
            
        except Exception as e:
            # Track error
            if self.performance_tracker:
                self.performance_tracker.track_error(
                    error=str(e),
                    phase=self.get_phase(),
                    module_name=self.name
                )
            
            return ModuleResult(
                data=None,
                success=False,
                error=str(e)
            )
    
    def validate_config(self) -> bool:
        """Validate module configuration"""
        return True
    
    async def initialize(self):
        """Initialize module (called once at startup)"""
        pass
    
    async def shutdown(self):
        """Cleanup module resources"""
        pass


class PerceptionModule(CortexModule):
    """Base class for perception modules"""
    
    def get_phase(self) -> CortexPhase:
        return CortexPhase.PERCEPTION
    
    @abstractmethod
    async def process(self, input_data: CortexInput, context: dict[str, Any]) -> ModuleResult[PerceptionResult]:
        """Process raw input into structured perception data"""
        pass


class ReasoningModule(CortexModule):
    """Base class for reasoning modules"""
    
    def get_phase(self) -> CortexPhase:
        return CortexPhase.REASONING
    
    @abstractmethod
    async def process(self, input_data: PerceptionResult, context: dict[str, Any]) -> ModuleResult[ReasoningResult]:
        """Process perception data into reasoning conclusions"""
        pass


class ActionModule(CortexModule):
    """Base class for action modules"""
    
    def get_phase(self) -> CortexPhase:
        return CortexPhase.ACTION
    
    @abstractmethod
    async def process(self, input_data: ReasoningResult, context: dict[str, Any]) -> ModuleResult[ActionResult]:
        """Process reasoning into actionable plans"""
        pass


# Example concrete implementations

class TextPerceptionModule(PerceptionModule):
    """Example perception module for text processing"""
    
    async def process(self, input_data: CortexInput, context: dict[str, Any]) -> ModuleResult[PerceptionResult]:
        """Process text input"""
        if not isinstance(input_data.raw_data, str):
            return ModuleResult(
                data=None,
                success=False,
                error="Input data must be string for text perception"
            )
        
        text = input_data.raw_data
        
        # Simple text analysis
        features = {
            "length": len(text),
            "word_count": len(text.split()),
            "has_questions": "?" in text,
            "has_commands": any(cmd in text.lower() for cmd in ["create", "build", "implement", "fix"]),
            "complexity_score": len(text.split()) / 100.0  # Simple metric
        }
        
        confidence = min(1.0, len(text) / 1000.0)  # Higher confidence for longer text
        
        result = PerceptionResult(
            processed_data={"text": text, "tokens": text.split()},
            features=features,
            confidence=confidence
        )
        
        return ModuleResult(
            data=result,
            success=True,
            metrics={"processing_time_ms": 1.0}  # Mock metric
        )


class LogicalReasoningModule(ReasoningModule):
    """Example reasoning module for logical analysis"""
    
    async def process(self, input_data: PerceptionResult, context: dict[str, Any]) -> ModuleResult[ReasoningResult]:
        """Perform logical reasoning on perception data"""
        features = input_data.features
        
        # Simple reasoning logic
        conclusions = []
        analysis = {}
        
        if features.get("has_commands", False):
            conclusions.append("User is requesting action")
            analysis["intent"] = "command"
        elif features.get("has_questions", False):
            conclusions.append("User is seeking information")
            analysis["intent"] = "query"
        else:
            conclusions.append("User is providing information")
            analysis["intent"] = "statement"
        
        complexity = features.get("complexity_score", 0)
        if complexity > 0.5:
            conclusions.append("High complexity task detected")
            analysis["complexity"] = "high"
        else:
            analysis["complexity"] = "low"
        
        confidence = input_data.confidence * 0.9  # Slightly lower than perception
        
        result = ReasoningResult(
            analysis=analysis,
            conclusions=conclusions,
            confidence=confidence,
            reasoning_chain=[
                {"step": 1, "rule": "intent_detection", "output": analysis["intent"]},
                {"step": 2, "rule": "complexity_assessment", "output": analysis["complexity"]}
            ]
        )
        
        return ModuleResult(
            data=result,
            success=True,
            metrics={"inference_steps": 2}
        )


class PlanningActionModule(ActionModule):
    """Example action module for planning"""
    
    async def process(self, input_data: ReasoningResult, context: dict[str, Any]) -> ModuleResult[ActionResult]:
        """Generate action plan from reasoning"""
        analysis = input_data.analysis
        
        actions = []
        priority_scores = {}
        
        # Generate actions based on intent
        intent = analysis.get("intent", "statement")
        
        if intent == "command":
            actions.append({
                "type": "execute_task",
                "description": "Execute requested command",
                "parameters": {"complexity": analysis.get("complexity", "low")}
            })
            priority_scores["execute_task"] = 0.9
        
        elif intent == "query":
            actions.append({
                "type": "provide_information", 
                "description": "Provide requested information",
                "parameters": {"search_required": True}
            })
            priority_scores["provide_information"] = 0.8
        
        else:
            actions.append({
                "type": "acknowledge",
                "description": "Acknowledge user input",
                "parameters": {"response_type": "confirmation"}
            })
            priority_scores["acknowledge"] = 0.5
        
        # Add monitoring action for high complexity
        if analysis.get("complexity") == "high":
            actions.append({
                "type": "monitor_progress",
                "description": "Monitor task progress",
                "parameters": {"frequency": "high"}
            })
            priority_scores["monitor_progress"] = 0.7
        
        execution_plan = {
            "sequence": [action["type"] for action in actions],
            "parallel_allowed": False,
            "timeout_seconds": 300
        }
        
        result = ActionResult(
            actions=actions,
            priority_scores=priority_scores,
            execution_plan=execution_plan
        )
        
        return ModuleResult(
            data=result,
            success=True,
            metrics={"actions_generated": len(actions)}
        )