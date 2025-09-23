"""
EOS Main Orchestrator

Integrates all EOS components and provides the main execution interface
for E-UPUSF orchestration with telemetry and governance integration.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging

# Import EOS components
from .schema import EOSSchema, EOSSpec
from .state_machine import EOSStateMachine, StateContext, StateType
from .context import ContextAnalyzer
from .routing import MoERouter
from .operators import LadderOperators, OperatorContext, OperatorType

# Import existing telemetry system
try:
    from ..performance_monitoring.telemetry.opentelemetry_config import (
        OpenTelemetryCollector, TelemetrySpan
    )
    from ..performance_monitoring.middleware.extension_interceptors import (
        track_extension_call
    )
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    # Fallback no-op implementations
    
    def track_extension_call(func):
        return func
    
    class TelemetrySpan:
        pass
    
    class OpenTelemetryCollector:
        def capture_span(self, span):
            pass

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result of EOS orchestration execution"""
    run_id: str
    success: bool
    final_state: StateType
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    telemetry_spans: List[TelemetrySpan] = field(default_factory=list)
    governance_report: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error_message: Optional[str] = None


class EOSOrchestrator:
    """Main E-UPUSF Orchestration System orchestrator"""
    
    def __init__(self, eos_spec: EOSSpec):
        """Initialize orchestrator with EOS specification"""
        self.spec = eos_spec
        self.run_id = str(uuid.uuid4())
        
        # Initialize components
        self.context_analyzer = ContextAnalyzer()
        self.state_machine = EOSStateMachine(self._spec_to_dict())
        self.moe_router = MoERouter(
            [expert.__dict__ for expert in self.spec.experts], 
            self._spec_to_dict().get("routing", {})
        )
        self.ladder_operators = LadderOperators(self._spec_to_dict())
        
        # Initialize telemetry
        if TELEMETRY_AVAILABLE:
            self.telemetry = OpenTelemetryCollector()
        else:
            self.telemetry = None
        
        # Execution state
        self.current_context: Optional[Dict[str, Any]] = None
        self.execution_trace: List[Dict[str, Any]] = []
        self.telemetry_spans: List[TelemetrySpan] = []
    
    def _spec_to_dict(self) -> Dict[str, Any]:
        """Convert EOSSpec to dictionary for compatibility"""
        return {
            "problem": {
                "title": self.spec.problem.title,
                "statement": self.spec.problem.statement,
                "objectives": self.spec.problem.objectives,
                "constraints": self.spec.problem.constraints,
                "stakeholders": self.spec.problem.stakeholders,
                "risk_tolerance": self.spec.problem.risk_tolerance
            },
            "context": {
                "cynefin_prior": {
                    "simple": self.spec.context.cynefin_prior.simple,
                    "complicated": self.spec.context.cynefin_prior.complicated,
                    "complex": self.spec.context.cynefin_prior.complex,
                    "chaotic": self.spec.context.cynefin_prior.chaotic
                },
                "domain_hints": self.spec.context.domain_hints,
                "uncertainty_thresholds": (
                    self.spec.context.uncertainty_thresholds
                )
            },
            "resources": self.spec.resources,
            "methods_registry": [
                {
                    "id": m.id,
                    "states": m.states,
                    "entry_criteria": m.entry_criteria,
                    "operators": m.operators,
                    "artifacts_out": m.artifacts_out
                } for m in self.spec.methods_registry
            ],
            "ladder": self.spec.ladder,
            "experts": [
                {
                    "id": e.id,
                    "kind": e.kind,
                    "inputs": e.inputs,
                    "outputs": e.outputs,
                    "cost": e.cost,
                    "risk": e.risk,
                    "quality_prior": e.quality_prior,
                    "fit_hints": e.fit_hints
                } for e in self.spec.experts
            ],
            "routing": self.spec.routing,
            "pipeline": self.spec.pipeline,
            "evaluation": self.spec.evaluation,
            "governance": self.spec.governance,
            "telemetry": self.spec.telemetry
        }
    
    @track_extension_call("eos_orchestrator", "orchestration")
    async def orchestrate(self) -> OrchestrationResult:
        """Execute complete EOS orchestration"""
        
        start_time = time.time()
        
        try:
            logger.info(
                f"Starting EOS orchestration with run_id: {self.run_id}"
            )
            
            # Step 1: Analyze context
            self.current_context = await self._analyze_initial_context()
            self._trace_event("context_analysis", self.current_context)
            
            # Step 2: Start state machine
            state_context = self.state_machine.start(self.run_id)
            
            # Step 3: Execute state machine with adaptive orchestration
            final_context = await self._execute_adaptive_orchestration(
                state_context
            )
            
            # Step 4: Generate final results
            result = await self._generate_results(final_context, start_time)
            
            logger.info(
                f"EOS orchestration completed successfully: "
                f"duration={result.duration_ms:.0f}ms, "
                f"final_state={result.final_state.value}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"EOS orchestration failed: {e}")
            
            duration_ms = (time.time() - start_time) * 1000
            
            return OrchestrationResult(
                run_id=self.run_id,
                success=False,
                final_state=StateType.OBSERVE,  # Default state
                duration_ms=duration_ms,
                error_message=str(e),
                execution_trace=self.execution_trace,
                telemetry_spans=self.telemetry_spans
            )
    
    async def _analyze_initial_context(self) -> Dict[str, Any]:
        """Analyze initial problem context"""
        
        logger.info("Analyzing initial context")
        
        spec_dict = self._spec_to_dict()
        analysis = self.context_analyzer.analyze_context(spec_dict)
        
        # Create telemetry span if available
        if self.telemetry:
            span = self.telemetry.start_span(
                component="eos_context_analyzer",
                operation="analyze_initial_context",
                trace_id=self.run_id
            )
            self.telemetry.finish_span(span)
            self.telemetry_spans.append(span)
        
        return analysis
    
    async def _execute_adaptive_orchestration(
        self, 
        initial_state_context: StateContext
    ) -> StateContext:
        """Execute adaptive orchestration with state machine and operators"""
        
        current_state_context = initial_state_context
        max_iterations = 50  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get current state
            current_state = self.state_machine.current_state
            if not current_state:
                break
            
            state_type = current_state.state_type
            logger.info(
                f"Iteration {iteration}: Processing state {state_type.value}"
            )
            
            # Determine required operators for current state
            required_operators = await self._determine_required_operators(
                state_type, current_state_context
            )
            
            # Execute operators
            for operator_type in required_operators:
                operator_result = await self._execute_operator_with_routing(
                    operator_type, current_state_context
                )
                
                if operator_result.success:
                    # Update state context with operator results
                    current_state_context = await self._merge_operator_results(
                        current_state_context, operator_result
                    )
                else:
                    logger.warning(
                        f"Operator {operator_type.value} failed: "
                        f"{operator_result.reasoning}"
                    )
            
            # Execute state machine step
            decision = await self.state_machine.step()
            
            self._trace_event("state_transition", {
                "from_state": state_type.value,
                "decision": decision.target_state.value if decision else None,
                "iteration": iteration
            })
            
            # Check stopping conditions
            if await self._should_stop_orchestration(current_state_context):
                break
            
            # If no transition, we're done
            if not decision:
                break
        
        logger.info(f"Orchestration completed after {iteration} iterations")
        return current_state_context
    
    async def _determine_required_operators(
        self, 
        state_type: StateType,
        context: StateContext
    ) -> List[OperatorType]:
        """Determine which operators are needed for current state"""
        
        operators = []
        
        # State-specific operator mapping
        if state_type == StateType.OBSERVE:
            # Check if semantic lifting is needed
            if (self.current_context and 
                    self.current_context.get("recommendations", {})
                    .get("should_semantic_lift", False)):
                operators.append(OperatorType.LIFT)
        
        elif state_type == StateType.ANALYZE:
            operators.append(OperatorType.DECOMPOSE)
            
            # Add lift if evidence score is low
            if context.evidence_score < 0.5:
                operators.append(OperatorType.LIFT)
        
        elif state_type == StateType.SYNTHESIZE:
            operators.append(OperatorType.SYNTHESIZE)
        
        elif state_type == StateType.IMPLEMENT:
            operators.append(OperatorType.DESCEND)
        
        # STATE.EVALUATE typically doesn't need operators
        
        return operators
    
    async def _execute_operator_with_routing(
        self, 
        operator_type: OperatorType,
        state_context: StateContext
    ) -> Any:
        """Execute operator with expert routing"""
        
        # Create operator context from state context
        operator_context = OperatorContext(
            input_artifacts=state_context.artifacts,
            dimensions=[],  # Will be populated by operators
            abstraction_level=0,
            constraints=self.spec.problem.constraints,
            stakeholders=self.spec.problem.stakeholders
        )
        
        # Route to appropriate expert if needed
        current_state = self.state_machine.current_state.state_type.value
        
        # Determine method (simplified - would be more sophisticated)
        current_method = self._determine_current_method()
        
        try:
            # Try to route to expert for complex operations
            if operator_type in [
                OperatorType.SYNTHESIZE, OperatorType.DECOMPOSE
            ]:
                routing_decision = await self.moe_router.route_to_experts(
                    current_state=current_state,
                    current_method=current_method,
                    required_inputs=[operator_type.value.lower()],
                    context={
                        "state_context": state_context.__dict__,
                        "operator_type": operator_type.value
                    }
                )
                
                # Execute with selected expert
                expert_result = await self.moe_router.execute_expert(
                    routing_decision.primary_expert,
                    inputs={"operator_context": operator_context.__dict__},
                    context={}
                )
                
                # Convert expert result to operator result format
                return await self._convert_expert_to_operator_result(
                    expert_result, operator_context
                )
        
        except Exception as e:
            logger.warning(
                f"Expert routing failed for {operator_type.value}: {e}"
            )
        
        # Fallback to direct operator execution
        return await self.ladder_operators.execute_operator(
            operator_type, operator_context
        )
    
    def _determine_current_method(self) -> str:
        """Determine current method based on context"""
        # Simplified method determination
        if self.current_context:
            methods = (self.current_context.get("recommendations", {})
                       .get("methods", []))
            return methods[0] if methods else "POLYA"
        return "POLYA"
    
    async def _merge_operator_results(
        self, 
        state_context: StateContext,
        operator_result: Any
    ) -> StateContext:
        """Merge operator results into state context"""
        
        # Update artifacts
        if hasattr(operator_result, 'updated_context'):
            new_artifacts = operator_result.updated_context.output_artifacts
            state_context.artifacts.update(new_artifacts)
        
        # Update metrics if operator provided insights
        if hasattr(operator_result, 'confidence'):
            state_context.metrics["last_operator_confidence"] = (
                operator_result.confidence
            )
        
        return state_context
    
    async def _convert_expert_to_operator_result(
        self, 
        expert_result: Dict[str, Any],
        operator_context: OperatorContext
    ) -> Any:
        """Convert expert execution result to operator result format"""
        # Convert expert result to operator result format
        # For now, create a mock operator result
        from .operators import OperatorResult
        
        return OperatorResult(
            success=expert_result.get("success", True),
            updated_context=operator_context,
            artifacts_produced=list(expert_result.get("outputs", {}).keys()),
            reasoning=[f"Expert execution: {expert_result.get('outputs', {}).get('result', '')}"],
            confidence=expert_result.get("quality_score", 0.8)
        )
    
    async def _should_stop_orchestration(self, context: StateContext) -> bool:
        """Check if orchestration should stop"""
        
        # Check evaluation metrics
        evaluation_config = self.spec.evaluation
        
        for metric in evaluation_config.get("metrics", []):
            metric_id = metric["id"]
            target = metric["target"]
            
            if metric_id == "time_to_insight":
                max_minutes = target.get("max_minutes", float('inf'))
                actual_minutes = (
                    context.metrics.get("total_duration_ms", 0) / (1000 * 60)
                )
                if actual_minutes > max_minutes:
                    return True
            
            elif metric_id == "breadth":
                min_alternatives = target.get("min_alternatives", 0)
                if context.breadth >= min_alternatives:
                    return True
        
        return False
    
    async def _generate_results(
        self, 
        final_context: StateContext,
        start_time: float
    ) -> OrchestrationResult:
        """Generate final orchestration results"""
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Generate governance report
        governance_report = await self._generate_governance_report(
            final_context
        )
        
        # Get final state
        final_state = (
            self.state_machine.current_state.state_type 
            if self.state_machine.current_state 
            else StateType.OBSERVE
        )
        
        return OrchestrationResult(
            run_id=self.run_id,
            success=True,
            final_state=final_state,
            artifacts=final_context.artifacts,
            metrics=final_context.metrics,
            execution_trace=self.execution_trace,
            telemetry_spans=self.telemetry_spans,
            governance_report=governance_report,
            duration_ms=duration_ms
        )
    
    async def _generate_governance_report(
        self, 
        context: StateContext
    ) -> Dict[str, Any]:
        """Generate governance compliance report"""
        
        # Simplified governance checking
        governance_config = self.spec.governance
        
        report = {
            "rulesets_checked": governance_config.get("rulesets", []),
            "compliance_score": 0.95,  # Mock score
            "violations": [],
            "recommendations": [
                "Continue monitoring telemetry metrics",
                "Review artifacts for quality assurance"
            ],
            "artifacts_validated": len(context.artifacts),
            "telemetry_spans_captured": len(self.telemetry_spans)
        }
        
        return report
    
    def _trace_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add event to execution trace"""
        
        trace_event = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "event_type": event_type,
            "data": data
        }
        
        self.execution_trace.append(trace_event)
        
        # Also log for debugging
        logger.debug(f"Trace event: {event_type} - {data}")
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get orchestration execution statistics"""
        
        return {
            "run_id": self.run_id,
            "spec_version": self.spec.eos_version,
            "execution_trace_length": len(self.execution_trace),
            "telemetry_spans_count": len(self.telemetry_spans),
            "current_state": (self.state_machine.current_state.state_type.value 
                            if self.state_machine.current_state else None),
            "moe_statistics": self.moe_router.get_routing_statistics(),
            "operator_capabilities": self.ladder_operators.get_operator_capabilities()
        }


# CLI interface for EOS orchestration
async def run_eos_orchestration(spec_file_path: str) -> OrchestrationResult:
    """Run EOS orchestration from specification file"""
    
    # Load and validate specification
    eos_schema = EOSSchema()
    validation_result, eos_spec = eos_schema.load_and_validate(spec_file_path)
    
    if not validation_result.valid:
        raise ValueError(f"Invalid EOS specification: {validation_result.errors}")
    
    # Create and run orchestrator
    orchestrator = EOSOrchestrator(eos_spec)
    result = await orchestrator.orchestrate()
    
    return result


def main():
    """CLI entry point for EOS orchestration"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Execute E-UPUSF Orchestration Schema"
    )
    parser.add_argument("spec_file", help="EOS YAML specification file")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Run orchestration
    try:
        result = asyncio.run(run_eos_orchestration(args.spec_file))
        
        # Output results
        result_dict = {
            "run_id": result.run_id,
            "success": result.success,
            "final_state": result.final_state.value,
            "duration_ms": result.duration_ms,
            "artifacts_count": len(result.artifacts),
            "execution_trace_length": len(result.execution_trace),
            "governance_score": result.governance_report.get("compliance_score", 0),
            "error_message": result.error_message
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result_dict, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(result_dict, indent=2))
        
        # Exit with appropriate code
        exit(0 if result.success else 1)
        
    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()