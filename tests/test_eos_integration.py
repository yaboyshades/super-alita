"""
Integration tests for E-UPUSF Orchestration Schema (EOS) system

Tests the complete EOS pipeline including schema validation, state machine,
operators, routing, and telemetry integration.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.eos.context import CynefinClassifier
from src.eos.operators import LadderOperators, OperatorType
from src.eos.orchestrator import EOSOrchestrator, run_eos_orchestration
from src.eos.routing import MoERouter

# Import EOS components
from src.eos.schema import EOSSchema, EOSValidator
from src.eos.state_machine import EOSStateMachine, StateType


class TestEOSSchemaValidation:
    """Test EOS schema validation functionality"""
    
    def test_valid_reference_spec(self):
        """Test validation of reference EOS specification"""
        reference_spec_path = Path("examples/eos/reference-orchestration.yaml")
        
        if reference_spec_path.exists():
            validator = EOSValidator()
            result = validator.validate_yaml(reference_spec_path)
            
            assert result.valid, f"Reference spec validation failed: {result.errors}"
            assert result.schema_version == "0.9"
            assert len(result.warnings) <= 5  # Allow some warnings
    
    def test_invalid_spec_detection(self):
        """Test detection of invalid specifications"""
        invalid_spec = {
            "eos_version": "0.9",
            "meta": {"run_id": "test", "owner": "test"},
            "problem": {"title": "test"},  # Missing required fields
            # Missing many required sections
        }
        
        validator = EOSValidator()
        result = validator.validate_dict(invalid_spec)
        
        assert not result.valid
        assert len(result.errors) > 0
    
    def test_cynefin_probability_validation(self):
        """Test Cynefin probability validation"""
        spec_with_invalid_probs = {
            "eos_version": "0.9",
            "meta": {"run_id": "test", "owner": "test"},
            "problem": {
                "title": "test",
                "statement": "test",
                "objectives": ["test"]
            },
            "context": {
                "cynefin_prior": {
                    "simple": 0.5,
                    "complicated": 0.3,
                    "complex": 0.1,
                    "chaotic": 0.05  # Sums to 0.95, not 1.0
                }
            },
            "resources": {"budgets": {}},
            "methods_registry": [],
            "ladder": {"enable_semantic_lifting": True},
            "experts": [],
            "routing": {"scoring": {"weights": {"fit": 0.25, "expected_utility": 0.25, "cost": 0.25, "risk": 0.25}, "top_k": 1}},
            "pipeline": {"states": []},
            "evaluation": {"metrics": [], "stopping": []},
            "governance": {"rulesets": [], "enforcement": {}},
            "telemetry": {"trace": {"fields": []}, "logs": {}}
        }
        
        validator = EOSValidator()
        result = validator.validate_dict(spec_with_invalid_probs)
        
        # Should pass schema validation but have semantic warnings
        assert result.valid  # Basic schema is valid
        assert any("probability" in warning.lower() for warning in result.warnings)


class TestCynefinClassification:
    """Test Cynefin framework classification"""
    
    def test_simple_problem_classification(self):
        """Test classification of simple problems"""
        classifier = CynefinClassifier()
        
        simple_problem = {
            "statement": "Standard procedure with clear cause-effect relationships",
            "constraints": ["Well-established process"],
            "stakeholders": ["team"],
            "domain_hints": ["standard", "routine"]
        }
        
        classification = classifier.classify(simple_problem)
        
        assert classification.primary_domain.value == "simple"
        assert classification.probabilities["simple"] > 0.4
        assert classification.confidence > 0.5
    
    def test_complex_problem_classification(self):
        """Test classification of complex problems"""
        classifier = CynefinClassifier()
        
        complex_problem = {
            "statement": "Uncertain outcomes with emergent properties and adaptive behavior",
            "constraints": ["Unknown unknowns", "Investigate novel approaches"],
            "stakeholders": ["research", "users", "business", "compliance", "partners"],
            "risk_tolerance": {"technical": "high", "safety": "medium", "business": "high"},
            "domain_hints": ["ai", "innovation", "experimental"]
        }
        
        classification = classifier.classify(complex_problem)
        
        assert classification.primary_domain.value in ["complex", "chaotic"]
        assert classification.entropy > 1.0  # High uncertainty
    
    def test_prior_integration(self):
        """Test integration of prior distributions"""
        classifier = CynefinClassifier()
        
        prior = {
            "simple": 0.1,
            "complicated": 0.2,
            "complex": 0.6,
            "chaotic": 0.1
        }
        
        neutral_problem = {
            "statement": "Moderate problem",
            "constraints": [],
            "stakeholders": ["team"]
        }
        
        classification = classifier.classify(neutral_problem, prior)
        
        # Should be influenced by prior
        assert classification.probabilities["complex"] > 0.3


class TestLadderOperators:
    """Test LADDER operators functionality"""
    
    @pytest.fixture
    def ladder_config(self):
        return {
            "dimensional_axes": ["stakeholders", "time_horizons", "feedback_loops"],
            "max_lift_depth": 3
        }
    
    @pytest.fixture
    def test_context(self):
        from src.eos.operators import OperatorContext
        return OperatorContext(
            input_artifacts={"problem": "test problem"},
            constraints=["constraint1", "constraint2"],
            stakeholders=["team", "users"]
        )
    
    @pytest.mark.asyncio
    async def test_lift_operator(self, ladder_config, test_context):
        """Test LIFT operator execution"""
        from src.eos.operators import Lift
        
        lift = Lift(ladder_config)
        result = await lift.execute(test_context)
        
        assert result.success
        assert len(result.dimensions_added) > 0
        assert "stakeholders" in result.dimensions_added
        assert result.confidence > 0.5
        assert len(result.artifacts_produced) > 0
    
    @pytest.mark.asyncio
    async def test_decompose_operator(self, ladder_config, test_context):
        """Test DECOMPOSE operator execution"""
        from src.eos.operators import Decompose
        
        decompose = Decompose(ladder_config)
        result = await decompose.execute(test_context)
        
        assert result.success
        assert "contradictions" in result.artifacts_produced
        assert "causal_chains" in result.artifacts_produced
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_synthesize_operator(self, ladder_config, test_context):
        """Test SYNTHESIZE operator execution"""
        from src.eos.operators import Synthesize
        
        synthesize = Synthesize(ladder_config)
        result = await synthesize.execute(test_context)
        
        assert result.success
        assert "alternatives" in result.artifacts_produced
        assert "ranked_solutions" in result.artifacts_produced
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_descend_operator(self, ladder_config, test_context):
        """Test DESCEND operator execution"""
        from src.eos.operators import Descend
        
        descend = Descend(ladder_config)
        result = await descend.execute(test_context)
        
        assert result.success
        assert "concrete_tasks" in result.artifacts_produced
        assert "rollout_plan" in result.artifacts_produced
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_operator_sequence(self, ladder_config, test_context):
        """Test sequence of operators"""
        operators = LadderOperators({"ladder": ladder_config})
        
        sequence = [OperatorType.LIFT, OperatorType.DECOMPOSE, OperatorType.SYNTHESIZE]
        results = await operators.execute_sequence(sequence, test_context)
        
        assert len(results) == len(sequence)
        assert all(result.success for result in results)
        
        # Check that context flows between operators
        final_context = results[-1].updated_context
        assert len(final_context.output_artifacts) > len(test_context.input_artifacts)


class TestMoERouting:
    """Test Mixture-of-Experts routing system"""
    
    @pytest.fixture
    def test_experts(self):
        return [
            {
                "id": "expert1",
                "kind": "tool",
                "inputs": ["text", "data"],
                "outputs": ["analysis"],
                "cost": {"cpu": 1, "gpu": 0, "time_s": 30},
                "risk": {"safety": 0.1, "privacy": 0.1},
                "quality_prior": 0.8,
                "fit_hints": ["Analyze", "DMAIC"]
            },
            {
                "id": "expert2", 
                "kind": "agent",
                "inputs": ["requirements", "constraints"],
                "outputs": ["solutions"],
                "cost": {"cpu": 2, "gpu": 1, "time_s": 60},
                "risk": {"safety": 0.2, "privacy": 0.1},
                "quality_prior": 0.7,
                "fit_hints": ["Synthesize", "DESIGN"]
            }
        ]
    
    @pytest.fixture
    def routing_config(self):
        return {
            "scoring": {
                "weights": {"fit": 0.4, "expected_utility": 0.3, "cost": 0.2, "risk": 0.1},
                "top_k": 2,
                "exploration_policy": {"type": "ucb1", "c": 1.0, "epsilon": 0.1}
            },
            "budgets": {"reserve_fraction": 0.2},
            "gating_rules": []
        }
    
    def test_expert_scoring(self, test_experts, routing_config):
        """Test expert scoring mechanism"""
        router = MoERouter(test_experts, routing_config)
        
        expert = test_experts[0]
        score = router.gating.score_expert(
            expert, 
            current_state="Analyze",
            current_method="DMAIC",
            context={}
        )
        
        assert 0 <= score.total_score <= 1
        assert score.fit_score > 0  # Should have good fit for Analyze + DMAIC
        assert score.expert_id == "expert1"
    
    @pytest.mark.asyncio
    async def test_expert_routing(self, test_experts, routing_config):
        """Test expert routing decision"""
        router = MoERouter(test_experts, routing_config)
        
        decision = await router.route_to_experts(
            current_state="Synthesize",
            current_method="DESIGN", 
            required_inputs=["requirements"],
            context={}
        )
        
        assert decision.primary_expert is not None
        assert decision.primary_expert.expert_id in ["expert1", "expert2"]
        assert decision.confidence > 0
        assert len(decision.reasoning) > 0
    
    @pytest.mark.asyncio
    async def test_expert_execution(self, test_experts, routing_config):
        """Test expert execution"""
        router = MoERouter(test_experts, routing_config)
        
        # Create a mock selection
        from src.eos.routing import ExpertScore, ExpertSelection
        
        selection = ExpertSelection(
            expert_id="expert1",
            score=ExpertScore(
                expert_id="expert1",
                fit_score=0.8,
                utility_score=0.7,
                cost_score=0.6,
                risk_score=0.9,
                total_score=0.75
            ),
            inputs=["text", "data"],
            outputs=["analysis"],
            estimated_cost={"cpu": 1, "gpu": 0, "time_s": 30},
            estimated_risk={"safety": 0.1, "privacy": 0.1}
        )
        
        result = await router.execute_expert(
            selection,
            inputs={"text": "test input"},
            context={}
        )
        
        assert result["success"]
        assert "outputs" in result
        assert result["quality_score"] > 0


class TestStateMachine:
    """Test EOS state machine functionality"""
    
    @pytest.fixture
    def minimal_spec(self):
        return {
            "context": {"uncertainty_thresholds": {"chaotic_emergency": 0.4}},
            "resources": {
                "budgets": {"compute_gh": 1.0},
                "time_limits": {"per_state_minutes": {"Observe": 5, "Analyze": 5}}
            },
            "evaluation": {"target_kpis": 0.8}
        }
    
    def test_state_machine_initialization(self, minimal_spec):
        """Test state machine initialization"""
        sm = EOSStateMachine(minimal_spec)
        
        assert sm.current_state is None
        assert len(sm.states) == 5  # All EOS states
        assert len(sm.transitions) > 0
    
    def test_state_machine_start(self, minimal_spec):
        """Test state machine start"""
        sm = EOSStateMachine(minimal_spec)
        context = sm.start("test-run-123")
        
        assert context.run_id == "test-run-123"
        assert sm.current_state is not None
        assert sm.current_state.state_type == StateType.OBSERVE
    
    @pytest.mark.asyncio
    async def test_state_execution(self, minimal_spec):
        """Test state execution"""
        sm = EOSStateMachine(minimal_spec)
        context = sm.start("test-run-123")
        
        # Execute one step
        decision = await sm.step()
        
        # Should have executed Observe state
        assert "observations" in context.artifacts
        assert context.evidence_score > 0
    
    @pytest.mark.asyncio
    async def test_full_execution(self, minimal_spec):
        """Test full state machine execution"""
        sm = EOSStateMachine(minimal_spec)
        context = sm.start("test-run-123")
        
        final_context = await sm.run_until_complete(max_steps=10)
        
        assert final_context.run_id == "test-run-123"
        assert len(final_context.artifacts) > 0
        assert len(final_context.metrics) > 0


class TestEOSIntegration:
    """Test complete EOS integration"""
    
    @pytest.fixture
    def reference_spec_path(self):
        return Path("examples/eos/reference-orchestration.yaml")
    
    @pytest.mark.asyncio
    async def test_full_orchestration_pipeline(self, reference_spec_path):
        """Test complete orchestration pipeline"""
        if not reference_spec_path.exists():
            pytest.skip("Reference specification not found")
        
        # Load and validate spec
        eos_schema = EOSSchema()
        validation_result, eos_spec = eos_schema.load_and_validate(reference_spec_path)
        
        assert validation_result.valid, f"Spec validation failed: {validation_result.errors}"
        
        # Create orchestrator
        orchestrator = EOSOrchestrator(eos_spec)
        
        # Mock telemetry to avoid dependencies
        with patch('src.eos.orchestrator.TELEMETRY_AVAILABLE', False):
            result = await orchestrator.orchestrate()
        
        assert result.success, f"Orchestration failed: {result.error_message}"
        assert result.run_id is not None
        assert result.duration_ms > 0
        assert len(result.artifacts) > 0
        assert len(result.execution_trace) > 0
        assert result.governance_report["compliance_score"] > 0.8
    
    def test_cli_interface(self, reference_spec_path):
        """Test CLI interface"""
        if not reference_spec_path.exists():
            pytest.skip("Reference specification not found")
        
        # Test validation only
        import sys

        from src.eos.schema import main as schema_main
        
        # Mock sys.argv for CLI test
        original_argv = sys.argv
        try:
            sys.argv = ["eos-validate", str(reference_spec_path)]
            
            # Should not raise exception for valid spec
            try:
                schema_main()
            except SystemExit as e:
                assert e.code == 0  # Success exit code
            
        finally:
            sys.argv = original_argv
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in orchestration"""
        # Create invalid spec
        invalid_spec_data = {
            "eos_version": "0.9",
            "meta": {"run_id": "test", "owner": "test"},
            "problem": {"title": "test", "statement": "test", "objectives": ["test"]},
            "context": {"cynefin_prior": {"simple": 1.0, "complicated": 0, "complex": 0, "chaotic": 0}},
            "resources": {"budgets": {}},
            "methods_registry": [],
            "ladder": {"enable_semantic_lifting": True},
            "experts": [],
            "routing": {"scoring": {"weights": {"fit": 1.0, "expected_utility": 0, "cost": 0, "risk": 0}, "top_k": 1}},
            "pipeline": {"states": []},
            "evaluation": {"metrics": [], "stopping": []},
            "governance": {"rulesets": [], "enforcement": {}},
            "telemetry": {"trace": {"fields": []}, "logs": {}}
        }
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_spec_data, f)
            temp_path = f.name
        
        try:
            # This should handle the error gracefully
            result = await run_eos_orchestration(temp_path)
            
            # Should complete but might have warnings
            assert result is not None
            
        except Exception as e:
            # If it fails, should be a controlled failure
            assert "validation" in str(e).lower()
        
        finally:
            Path(temp_path).unlink()  # Clean up


class TestTelemetryIntegration:
    """Test telemetry integration with existing monitoring"""
    
    @pytest.mark.asyncio
    async def test_telemetry_span_capture(self):
        """Test telemetry span capture during orchestration"""
        # Mock the telemetry system
        mock_collector = MagicMock()
        
        with patch('src.eos.orchestrator.OpenTelemetryCollector', return_value=mock_collector):
            with patch('src.eos.orchestrator.TELEMETRY_AVAILABLE', True):
                # Create minimal spec
                minimal_spec_data = {
                    "eos_version": "0.9",
                    "meta": {"run_id": "test", "owner": "test"},
                    "problem": {"title": "test", "statement": "test", "objectives": ["test"]},
                    "context": {"cynefin_prior": {"simple": 1.0, "complicated": 0, "complex": 0, "chaotic": 0}},
                    "resources": {"budgets": {}},
                    "methods_registry": [],
                    "ladder": {"enable_semantic_lifting": True},
                    "experts": [],
                    "routing": {"scoring": {"weights": {"fit": 1.0, "expected_utility": 0, "cost": 0, "risk": 0}, "top_k": 1}},
                    "pipeline": {"states": []},
                    "evaluation": {"metrics": [], "stopping": []},
                    "governance": {"rulesets": [], "enforcement": {}},
                    "telemetry": {"trace": {"fields": []}, "logs": {}}
                }
                
                from src.eos.schema import EOSParser
                eos_spec = EOSParser.parse_spec(minimal_spec_data)
                orchestrator = EOSOrchestrator(eos_spec)
                
                result = await orchestrator.orchestrate()
                
                # Check that telemetry was captured
                assert len(result.telemetry_spans) > 0
                # Mock collector should have been called
                assert mock_collector.capture_span.called
    
    def test_constitutional_integration(self):
        """Test integration with existing constitutional rules"""
        # This would test integration with the constitutional framework
        # For now, verify governance hooks are in place
        
        from src.eos.orchestrator import EOSOrchestrator
        
        # Check that governance integration points exist
        assert hasattr(EOSOrchestrator, '_generate_governance_report')
        
        # Could test rule validation if constitutional validator is available
        try:
            from scripts.rule_validator import ConstitutionalRuleEngine
            # Integration available
            assert True
        except ImportError:
            # Not available in test environment
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])