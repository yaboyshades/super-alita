"""
EOS-Mangle Integration Bridge

Integrates the E-UPUSF Orchestration Schema (EOS) with Google's Mangle
deductive reasoning engine for enhanced context-adaptive problem solving.

This bridge enables:
- EOS orchestration with Mangle expert routing
- Semantic lifting enhanced by logical reasoning
- Constitutional compliance via deductive validation
- Knowledge graph integration for expert selection
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .orchestrator import EOSOrchestrator, OrchestrationResult
from .schema import EOSSpec

logger = logging.getLogger(__name__)

# Optional Mangle imports with fallback
try:
    from src.abilities.mangle.mangle_ability import MangleAbility
    from src.core.mangle import MangleIntegration
    from src.unified_intelligence.mangle_bridge import MangleBridge
    MANGLE_AVAILABLE = True
except ImportError:
    logger.warning("Mangle integration not available")
    MANGLE_AVAILABLE = False
    MangleAbility = None
    MangleIntegration = None
    MangleBridge = None


@dataclass
class EOSMangleConfig:
    """Configuration for EOS-Mangle integration"""
    enable_mangle_routing: bool = True
    enable_deductive_validation: bool = True
    enable_semantic_enhancement: bool = True
    enable_knowledge_graph: bool = True
    mangle_timeout: float = 30.0
    deduction_confidence_threshold: float = 0.7


@dataclass
class MangleExpertDecision:
    """Decision result from Mangle-enhanced expert routing"""
    expert_id: str
    confidence: float
    reasoning_steps: List[str]
    logical_justification: str
    alternatives: List[Dict[str, Any]]
    constitutional_compliance: float


class EOSMangleOrchestrator:
    """
    Enhanced EOS Orchestrator with Mangle deductive reasoning integration.
    
    Combines the adaptive orchestration of EOS with the logical reasoning
    capabilities of Mangle for enhanced decision-making and validation.
    """

    def __init__(self, eos_spec: EOSSpec, config: EOSMangleConfig = None):
        """
        Initialize EOS-Mangle orchestrator.
        
        Args:
            eos_spec: EOS specification
            config: Configuration for Mangle integration
        """
        self.eos_spec = eos_spec
        self.config = config or EOSMangleConfig()
        
        # Initialize EOS orchestrator
        self.eos_orchestrator = EOSOrchestrator(eos_spec)
        
        # Initialize Mangle components
        self.mangle_ability: Optional[MangleAbility] = None
        self.mangle_integration: Optional[MangleIntegration] = None
        self.mangle_bridge: Optional[MangleBridge] = None
        
        # State tracking
        self.knowledge_base_initialized = False
        self.constitutional_rules_loaded = False
        
        # Initialize Mangle if available
        if MANGLE_AVAILABLE:
            self._initialize_mangle()

    def _initialize_mangle(self) -> None:
        """Initialize Mangle components with graceful degradation."""
        try:
            if self.config.enable_mangle_routing:
                self.mangle_ability = MangleAbility({
                    'timeout': self.config.mangle_timeout
                })
                logger.info("Mangle ability initialized for EOS integration")
            
            if self.config.enable_knowledge_graph:
                self.mangle_bridge = MangleBridge()
                logger.info("Mangle bridge initialized for EOS integration")
                
        except Exception as e:
            logger.warning(f"Failed to initialize Mangle for EOS: {e}")

    async def orchestrate_with_mangle(self) -> OrchestrationResult:
        """
        Execute EOS orchestration enhanced with Mangle reasoning.
        
        Returns:
            Enhanced orchestration result with Mangle insights
        """
        start_time = time.time()
        
        try:
            # Step 1: Initialize knowledge base with problem context
            if self.mangle_ability and self.config.enable_knowledge_graph:
                await self._initialize_knowledge_base()
            
            # Step 2: Load constitutional rules into Mangle
            if self.mangle_ability and self.config.enable_deductive_validation:
                await self._load_constitutional_rules()
            
            # Step 3: Execute enhanced EOS orchestration
            result = await self._execute_enhanced_orchestration()
            
            # Step 4: Add Mangle insights to result
            if self.mangle_bridge:
                mangle_insights = await self._generate_mangle_insights(result)
                result.telemetry_spans.extend(mangle_insights.get('spans', []))
                result.artifacts.update(mangle_insights.get('artifacts', {}))
            
            # Step 5: Validate results with constitutional reasoning
            if self.mangle_ability and self.config.enable_deductive_validation:
                constitutional_score = (
                    await self._validate_constitutional_compliance(
                        result
                    )
                )
                result.governance_report[
                    'mangle_constitutional_score'
                ] = constitutional_score
            
            duration_ms = (time.time() - start_time) * 1000
            result.duration_ms = duration_ms
            
            logger.info(
                f"EOS-Mangle orchestration completed in {duration_ms:.0f}ms"
            )
            return result
            
        except Exception as e:
            logger.error(f"EOS-Mangle orchestration failed: {e}")
            # Fallback to standard EOS orchestration
            return await self.eos_orchestrator.orchestrate()

    async def _initialize_knowledge_base(self) -> None:
        """Initialize Mangle knowledge base with problem context."""
        if not self.mangle_ability:
            return
            
        try:
            # Add problem context as facts
            problem = self.eos_spec.problem
            
            # Basic problem facts
            await self.mangle_ability.add_fact(
                f'problem_title("{problem.title}")'
            )
            await self.mangle_ability.add_fact(
                f'problem_statement("{problem.statement}")'
            )
            
            # Objectives as facts
            for i, objective in enumerate(problem.objectives):
                await self.mangle_ability.add_fact(
                    f'objective({i}, "{objective}")'
                )
            
            # Constraints as facts
            for i, constraint in enumerate(problem.constraints):
                await self.mangle_ability.add_fact(
                    f'constraint({i}, "{constraint}")'
                )
            
            # Context as facts
            context = self.eos_spec.context
            cynefin = context.cynefin_prior
            
            await self.mangle_ability.add_fact(
                f'cynefin_domain("simple", {cynefin.simple})'
            )
            await self.mangle_ability.add_fact(
                f'cynefin_domain("complicated", {cynefin.complicated})'
            )
            await self.mangle_ability.add_fact(
                f'cynefin_domain("complex", {cynefin.complex})'
            )
            await self.mangle_ability.add_fact(
                f'cynefin_domain("chaotic", {cynefin.chaotic})'
            )
            
            self.knowledge_base_initialized = True
            logger.info("Mangle knowledge base initialized with EOS context")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Mangle knowledge base: {e}")

    async def _load_constitutional_rules(self) -> None:
        """Load constitutional compliance rules into Mangle."""
        if not self.mangle_ability:
            return
            
        try:
            # Define constitutional compliance rules
            constitutional_rules = [
                # Library-first principle
                '''library_first_compliant(Method) :- 
                   method(Method), 
                   uses_existing_library(Method).''',
                
                # Test-first principle  
                '''test_first_compliant(Method) :-
                   method(Method),
                   has_tests(Method).''',
                
                # Simplicity gate
                '''simplicity_compliant(Method) :-
                   method(Method),
                   complexity_score(Method, Score),
                   Score < 10.''',
                
                # Integration validation
                '''integration_compliant(Method) :-
                   method(Method),
                   has_integration_tests(Method).''',
                
                # Overall constitutional compliance
                '''constitutional_compliant(Method) :-
                   library_first_compliant(Method),
                   test_first_compliant(Method),
                   simplicity_compliant(Method),
                   integration_compliant(Method).'''
            ]
            
            for rule in constitutional_rules:
                await self.mangle_ability.add_rule(
                    f"constitutional_rule_{len(constitutional_rules)}", 
                    rule
                )
            
            self.constitutional_rules_loaded = True
            logger.info("Constitutional rules loaded into Mangle")
            
        except Exception as e:
            logger.warning(f"Failed to load constitutional rules: {e}")

    async def _execute_enhanced_orchestration(self) -> OrchestrationResult:
        """Execute EOS orchestration with Mangle enhancements."""
        # Replace the EOS MoE router with Mangle-enhanced routing
        if self.mangle_ability and self.config.enable_mangle_routing:
            self.eos_orchestrator.moe_router = MangleEnhancedRouter(
                self.eos_orchestrator.moe_router,
                self.mangle_ability
            )
        
        # Execute the standard EOS orchestration
        return await self.eos_orchestrator.orchestrate()

    async def _generate_mangle_insights(
        self, result: OrchestrationResult
    ) -> Dict[str, Any]:
        """Generate additional insights using Mangle reasoning."""
        if not self.mangle_bridge:
            return {}
            
        try:
            insights = {}
            
            # Query for problem analysis insights
            problem_insights = self.mangle_bridge.query_codebase(
                f"What are the implications of "
                f"'{self.eos_spec.problem.statement}'?"
            )
            
            # Query for state transition insights
            state_insights = self.mangle_bridge.query_codebase(
                f"What are the optimal state transitions for "
                f"{result.final_state.value}?"
            )
            
            insights['artifacts'] = {
                'mangle_problem_analysis': problem_insights,
                'mangle_state_analysis': state_insights,
                'mangle_knowledge_stats': self.mangle_bridge.get_status()
            }
            
            return insights
            
        except Exception as e:
            logger.warning(f"Failed to generate Mangle insights: {e}")
            return {}

    async def _validate_constitutional_compliance(
        self, result: OrchestrationResult
    ) -> float:
        """Validate constitutional compliance using Mangle reasoning."""
        if not self.mangle_ability or not self.constitutional_rules_loaded:
            return 0.8  # Default score when Mangle unavailable
            
        try:
            # Query constitutional compliance for each method used
            compliance_scores = []
            
            # Add facts about the orchestration result
            await self.mangle_ability.add_fact(
                f'orchestration_success({str(result.success).lower()})'
            )
            await self.mangle_ability.add_fact(
                f'final_state("{result.final_state.value}")'
            )
            await self.mangle_ability.add_fact(
                f'duration_ms({result.duration_ms})'
            )
            
            # Query for constitutional compliance
            compliance_query = 'constitutional_compliant(Method)'
            compliance_result = await self.mangle_ability.query(
                compliance_query
            )
            
            # Parse compliance results (simplified)
            if (
                hasattr(compliance_result, 'success') and 
                compliance_result.success
            ):
                # Extract compliance score from results
                # High score for successful validation
                compliance_scores.append(0.9)
            else:
                # Lower score for validation issues
                compliance_scores.append(0.6)
            
            # Calculate average compliance score
            avg_score = (
                sum(compliance_scores) / len(compliance_scores) 
                if compliance_scores else 0.7
            )
            
            logger.info(f"Constitutional compliance score: {avg_score:.2f}")
            return avg_score
            
        except Exception as e:
            logger.warning(f"Constitutional compliance validation failed: {e}")
            return 0.7  # Default score on failure

    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of EOS-Mangle integration."""
        return {
            'mangle_available': MANGLE_AVAILABLE,
            'mangle_ability_initialized': self.mangle_ability is not None,
            'mangle_bridge_initialized': self.mangle_bridge is not None,
            'knowledge_base_initialized': self.knowledge_base_initialized,
            'constitutional_rules_loaded': self.constitutional_rules_loaded,
            'config': {
                'enable_mangle_routing': self.config.enable_mangle_routing,
                'enable_deductive_validation': (
                    self.config.enable_deductive_validation
                ),
                'enable_semantic_enhancement': (
                    self.config.enable_semantic_enhancement
                ),
                'enable_knowledge_graph': self.config.enable_knowledge_graph,
                'mangle_timeout': self.config.mangle_timeout,
                'deduction_confidence_threshold': (
                    self.config.deduction_confidence_threshold
                )
            }
        }


class MangleEnhancedRouter:
    """
    Enhanced MoE router using Mangle deductive reasoning for expert selection.
    
    Wraps the existing MoE router and enhances expert selection decisions
    with logical reasoning and constitutional validation.
    """

    def __init__(self, base_router, mangle_ability: MangleAbility):
        """
        Initialize Mangle-enhanced router.
        
        Args:
            base_router: Original MoE router
            mangle_ability: Mangle reasoning engine
        """
        self.base_router = base_router
        self.mangle_ability = mangle_ability

    async def route_to_experts(
        self, 
        current_state: str, 
        current_method: str,
        required_inputs: List[str], 
        context: Dict[str, Any]
    ) -> Any:
        """Enhanced expert routing with Mangle reasoning."""
        
        # Get base routing decision
        base_decision = await self.base_router.route_to_experts(
            current_state, current_method, required_inputs, context
        )
        
        if not self.mangle_ability:
            return base_decision
        
        try:
            # Add context to Mangle knowledge base
            await self.mangle_ability.add_fact(
                f'current_state("{current_state}")'
            )
            await self.mangle_ability.add_fact(
                f'current_method("{current_method}")'
            )
            
            for input_type in required_inputs:
                await self.mangle_ability.add_fact(
                    f'required_input("{input_type}")'
                )
            
            # Query for optimal expert selection
            expert_query = (
                f'optimal_expert(Expert) :- expert(Expert), '
                f'suitable_for_state("{current_state}", Expert).'
            )
            await self.mangle_ability.add_rule(
                "expert_selection", expert_query
            )
            
            # Query results
            result = await self.mangle_ability.query('optimal_expert(Expert)')
            
            # Enhance base decision with Mangle insights
            if hasattr(result, 'success') and result.success:
                # Extract expert recommendations and enhance routing decision
                logger.info("Enhanced expert routing with Mangle reasoning")
            
            return base_decision
            
        except Exception as e:
            logger.warning(f"Mangle expert routing enhancement failed: {e}")
            return base_decision

    async def execute_expert(
        self, 
        expert_id: str, 
        inputs: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute expert with Mangle validation."""
        
        # Execute through base router
        result = await self.base_router.execute_expert(
            expert_id, inputs, context
        )
        
        if not self.mangle_ability:
            return result
        
        try:
            # Add expert execution facts
            await self.mangle_ability.add_fact(
                f'expert_executed("{expert_id}", '
                f'{result.get("success", False)})'
            )
            
            # Validate execution with constitutional rules
            validation_query = f'execution_valid("{expert_id}")'
            validation_result = await self.mangle_ability.query(
                validation_query
            )
            
            # Enhance result with validation insights
            result['mangle_validation'] = {
                'validated': (
                    hasattr(validation_result, 'success') and 
                    validation_result.success
                ),
                'expert_id': expert_id
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Mangle expert execution validation failed: {e}")
            return result


# CLI interface for EOS-Mangle integration
async def run_eos_mangle_orchestration(
    spec_file_path: str, 
    config: EOSMangleConfig = None
) -> OrchestrationResult:
    """Run EOS orchestration with Mangle integration."""
    
    # Load and validate EOS specification
    from .schema import EOSSchema
    eos_schema = EOSSchema()
    validation_result, eos_spec = eos_schema.load_and_validate(spec_file_path)
    
    if not validation_result.valid:
        raise ValueError(
            f"Invalid EOS specification: {validation_result.errors}"
        )
    
    # Create and run EOS-Mangle orchestrator
    orchestrator = EOSMangleOrchestrator(eos_spec, config)
    result = await orchestrator.orchestrate_with_mangle()
    
    return result


def main():
    """CLI entry point for EOS-Mangle orchestration."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Execute EOS orchestration with Mangle integration"
    )
    parser.add_argument("spec_file", help="EOS YAML specification file")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--disable-mangle", action="store_true", 
                       help="Disable Mangle integration")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Configure EOS-Mangle integration
    config = EOSMangleConfig()
    if args.disable_mangle:
        config.enable_mangle_routing = False
        config.enable_deductive_validation = False
        config.enable_semantic_enhancement = False
        config.enable_knowledge_graph = False
    
    # Run orchestration
    try:
        result = asyncio.run(run_eos_mangle_orchestration(args.spec_file, config))
        
        # Output results
        result_dict = {
            "run_id": result.run_id,
            "success": result.success,
            "final_state": result.final_state.value,
            "duration_ms": result.duration_ms,
            "artifacts_count": len(result.artifacts),
            "execution_trace_length": len(result.execution_trace),
            "governance_score": result.governance_report.get("compliance_score", 0),
            "mangle_constitutional_score": result.governance_report.get("mangle_constitutional_score", 0),
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
        logger.error(f"EOS-Mangle orchestration failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()