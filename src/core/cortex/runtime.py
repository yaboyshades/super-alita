"""
Cortex Runtime: Main orchestrator for cognitive processing cycles
Implements the perception → reasoning → action flow with pluggable modules
"""

from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field
import asyncio
import uuid
import time
from datetime import datetime, timezone

from ..plugin_interface import PluginInterface
from ..events import create_event
from .markers import (
    PerformanceTracker, 
    CortexPhase, 
    CortexEvent, 
    create_cortex_event
)
from .modules import (
    CortexModule, 
    PerceptionModule, 
    ReasoningModule, 
    ActionModule,
    CortexInput,
    PerceptionResult,
    ReasoningResult, 
    ActionResult,
    ModuleResult
)


@dataclass
class CortexContext:
    """Context for Cortex processing cycle"""
    cycle_id: str
    session_id: str
    user_id: Optional[str]
    workspace: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, **kwargs):
        """Update context variables"""
        self.variables.update(kwargs)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get context variable"""
        return self.variables.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "cycle_id": self.cycle_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "workspace": self.workspace,
            "metadata": self.metadata,
            "variables": self.variables
        }


@dataclass 
class CortexResult:
    """Complete result from Cortex processing cycle"""
    cycle_id: str
    success: bool
    perception_result: Optional[PerceptionResult]
    reasoning_result: Optional[ReasoningResult]
    action_result: Optional[ActionResult]
    performance_markers: List[Any]  # PerformanceMarker
    total_duration_ms: float
    error: Optional[str] = None
    context: Optional[CortexContext] = None


class CortexRuntime(PluginInterface):
    """Main Cortex runtime orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.perception_modules: List[PerceptionModule] = []
        self.reasoning_modules: List[ReasoningModule] = []
        self.action_modules: List[ActionModule] = []
        self.performance_tracker = PerformanceTracker()
        self.active_cycles: Dict[str, CortexContext] = {}
        self.event_handlers: List[Any] = []  # Event handlers
        self.running = False
    
    @property
    def name(self) -> str:
        return "cortex_runtime"
    
    async def setup(self, event_bus=None, **kwargs):
        """Initialize the Cortex runtime"""
        self.event_bus = event_bus
        self.running = True
        
        # Initialize all modules
        all_modules = self.perception_modules + self.reasoning_modules + self.action_modules
        for module in all_modules:
            module.set_performance_tracker(self.performance_tracker)
            await module.initialize()
        
        print(f"🧠 Cortex Runtime initialized with {len(all_modules)} modules")
    
    async def start(self):
        """Start the Cortex runtime (required by PluginInterface)"""
        await self.setup()
    
    async def shutdown(self):
        """Shutdown the Cortex runtime"""
        self.running = False
        
        # Shutdown all modules
        all_modules = self.perception_modules + self.reasoning_modules + self.action_modules
        for module in all_modules:
            await module.shutdown()
        
        print("🧠 Cortex Runtime shutdown complete")
    
    def register_perception_module(self, module: PerceptionModule):
        """Register a perception module"""
        self.perception_modules.append(module)
        print(f"📡 Registered perception module: {module.name}")
    
    def register_reasoning_module(self, module: ReasoningModule):
        """Register a reasoning module"""
        self.reasoning_modules.append(module)
        print(f"🤔 Registered reasoning module: {module.name}")
    
    def register_action_module(self, module: ActionModule):
        """Register an action module"""
        self.action_modules.append(module)
        print(f"⚡ Registered action module: {module.name}")
    
    def create_context(
        self, 
        session_id: str,
        user_id: Optional[str] = None,
        workspace: Optional[str] = None,
        **metadata
    ) -> CortexContext:
        """Create a new processing context"""
        cycle_id = str(uuid.uuid4())
        
        context = CortexContext(
            cycle_id=cycle_id,
            session_id=session_id,
            user_id=user_id,
            workspace=workspace,
            metadata=metadata
        )
        
        self.active_cycles[cycle_id] = context
        return context
    
    async def process_cycle(
        self, 
        input_data: Any,
        context: CortexContext
    ) -> CortexResult:
        """Execute a complete Cortex processing cycle"""
        cycle_start_time = time.time()
        
        # Start cycle tracking
        self.performance_tracker.start_cycle(context.cycle_id)
        
        perception_result = None
        reasoning_result = None
        action_result = None
        error = None
        
        try:
            # Create Cortex input
            cortex_input = CortexInput(
                raw_data=input_data,
                context=context.to_dict(),
                metadata=context.metadata,
                cycle_id=context.cycle_id
            )
            
            # Phase 1: Perception
            perception_result = await self._run_perception_phase(cortex_input, context)
            if not perception_result or not perception_result.success:
                error = f"Perception failed: {perception_result.error if perception_result else 'No result'}"
                raise RuntimeError(error)
            
            # Phase 2: Reasoning
            reasoning_result = await self._run_reasoning_phase(perception_result.data, context)
            if not reasoning_result or not reasoning_result.success:
                error = f"Reasoning failed: {reasoning_result.error if reasoning_result else 'No result'}"
                raise RuntimeError(error)
            
            # Phase 3: Action
            action_result = await self._run_action_phase(reasoning_result.data, context)
            if not action_result or not action_result.success:
                error = f"Action failed: {action_result.error if action_result else 'No result'}"
                raise RuntimeError(error)
            
            success = True
            
        except Exception as e:
            error = str(e)
            success = False
            self.performance_tracker.track_error(error)
        
        finally:
            # End cycle tracking
            self.performance_tracker.end_cycle()
            
            # Calculate total duration
            total_duration_ms = (time.time() - cycle_start_time) * 1000
            
            # Create result
            result = CortexResult(
                cycle_id=context.cycle_id,
                success=success,
                perception_result=perception_result.data if perception_result and perception_result.success else None,
                reasoning_result=reasoning_result.data if reasoning_result and reasoning_result.success else None,
                action_result=action_result.data if action_result and action_result.success else None,
                performance_markers=self.performance_tracker.get_markers(),
                total_duration_ms=total_duration_ms,
                error=error,
                context=context
            )
            
            # Emit Cortex event
            await self._emit_cycle_event(result)
            
            # Clean up
            if context.cycle_id in self.active_cycles:
                del self.active_cycles[context.cycle_id]
            
            return result
    
    async def _run_perception_phase(
        self, 
        cortex_input: CortexInput, 
        context: CortexContext
    ) -> Optional[ModuleResult[PerceptionResult]]:
        """Run perception phase with all registered modules"""
        self.performance_tracker.start_phase(CortexPhase.PERCEPTION)
        
        try:
            # For now, run modules sequentially - could be parallel in future
            for module in self.perception_modules:
                if not module.enabled:
                    continue
                
                result = await module.execute_with_tracking(
                    cortex_input, 
                    context.to_dict()
                )
                
                if result.success:
                    return result
            
            # No successful modules
            return ModuleResult(
                data=None,
                success=False,
                error="No perception modules succeeded"
            )
            
        finally:
            self.performance_tracker.end_phase(CortexPhase.PERCEPTION)
    
    async def _run_reasoning_phase(
        self, 
        perception_result: PerceptionResult, 
        context: CortexContext
    ) -> Optional[ModuleResult[ReasoningResult]]:
        """Run reasoning phase with all registered modules"""
        self.performance_tracker.start_phase(CortexPhase.REASONING)
        
        try:
            for module in self.reasoning_modules:
                if not module.enabled:
                    continue
                
                result = await module.execute_with_tracking(
                    perception_result,
                    context.to_dict()
                )
                
                if result.success:
                    return result
            
            return ModuleResult(
                data=None,
                success=False,
                error="No reasoning modules succeeded"
            )
            
        finally:
            self.performance_tracker.end_phase(CortexPhase.REASONING)
    
    async def _run_action_phase(
        self, 
        reasoning_result: ReasoningResult, 
        context: CortexContext
    ) -> Optional[ModuleResult[ActionResult]]:
        """Run action phase with all registered modules"""
        self.performance_tracker.start_phase(CortexPhase.ACTION)
        
        try:
            for module in self.action_modules:
                if not module.enabled:
                    continue
                
                result = await module.execute_with_tracking(
                    reasoning_result,
                    context.to_dict()
                )
                
                if result.success:
                    return result
            
            return ModuleResult(
                data=None,
                success=False,
                error="No action modules succeeded"
            )
            
        finally:
            self.performance_tracker.end_phase(CortexPhase.ACTION)
    
    async def _emit_cycle_event(self, result: CortexResult):
        """Emit Cortex cycle completion event"""
        if not self.event_bus:
            return
        
        try:
            # Create Cortex event with performance data
            cortex_event = create_cortex_event(
                event_type="cortex_cycle_complete",
                cycle_id=result.cycle_id,
                markers=result.performance_markers,
                context=result.context.to_dict() if result.context else {},
                success=result.success,
                duration_ms=result.total_duration_ms,
                error=result.error
            )
            
            # Emit to event bus
            await self.event_bus.emit_event(cortex_event.base_event)
            
        except Exception as e:
            print(f"⚠️ Failed to emit Cortex event: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get runtime statistics"""
        return {
            "perception_modules": len(self.perception_modules),
            "reasoning_modules": len(self.reasoning_modules),
            "action_modules": len(self.action_modules),
            "active_cycles": len(self.active_cycles),
            "running": self.running
        }


# Factory function for easy setup
def create_cortex_runtime(config: Optional[Dict[str, Any]] = None) -> CortexRuntime:
    """Create and configure a Cortex runtime with default modules"""
    from .modules import TextPerceptionModule, LogicalReasoningModule, PlanningActionModule
    
    runtime = CortexRuntime(config)
    
    # Register default modules
    runtime.register_perception_module(TextPerceptionModule("text_perception"))
    runtime.register_reasoning_module(LogicalReasoningModule("logical_reasoning"))
    runtime.register_action_module(PlanningActionModule("planning_action"))
    
    return runtime