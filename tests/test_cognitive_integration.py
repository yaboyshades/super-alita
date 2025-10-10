"""
Integration tests for Cognitive Module Integration with LADDER-MultiAgent Bridge.

These tests verify the end-to-end integration of cognitive modules with the
LADDER planning and multi-agent execution workflow.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from src.cognitive_modules import (
    MultiModalAnalysisResult,
    ReasoningChain,
    ReasoningStep,
    ReasoningStepType,
)
from src.core import yaml_utils
from src.integration.ladder_multiagent_bridge import (
    CodeGenerationResult,
    LADDERMultiAgentBridge,
    LADDERTask,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "cognitive": {
            "enabled": True,
            "multimodal_analyzer": {
                "quality_thresholds": {
                    "correctness_min": 0.7,
                    "maintainability_min": 0.6,
                    "security_min": 0.75,
                },
                "enable_caching": True,
            },
            "reasoning_verifier": {
                "max_chain_length": 10,
                "verification_thresholds": {
                    "consistency_min": 0.7,
                    "correctness_min": 0.6,
                    "completeness_min": 0.8,
                },
            },
            "integration": {
                "darwin_weight": 0.6,
                "max_revision_attempts": 3,
                "enable_reasoning_context": True,
            },
        }
    }


@pytest.fixture
def sample_task() -> LADDERTask:
    """Sample LADDER task for testing."""
    return LADDERTask(
        task_id="test-task-001",
        description="Implement a function to calculate factorial",
        requirements="Create a recursive factorial function with input validation",
        complexity="medium",
        context={"priority": "high"},
    )


@pytest.fixture
def sample_analysis_result() -> MultiModalAnalysisResult:
    """Sample analysis result for testing."""
    return MultiModalAnalysisResult(
        understanding_confidence=0.85,
        quality_prediction={
            "correctness": 0.9,
            "maintainability": 0.8,
            "security": 0.85,
            "performance": 0.75,
        },
        improvement_suggestions=[
            "Add input validation",
            "Improve error handling",
            "Add docstring",
        ],
        feature_importance={},
    )


@pytest.fixture
def sample_reasoning_chain() -> ReasoningChain:
    """Sample reasoning chain for testing."""
    return ReasoningChain(
        steps=[
            ReasoningStep(
                step_type=ReasoningStepType.PROBLEM_ANALYSIS,
                content="Analyze factorial requirements",
                confidence=0.9,
                dependencies=[],
            ),
            ReasoningStep(
                step_type=ReasoningStepType.SOLUTION_STRATEGY,
                content="Use recursive approach",
                confidence=0.85,
                dependencies=[0],
            ),
            ReasoningStep(
                step_type=ReasoningStepType.CODE_GENERATION,
                content="Implement recursive function",
                confidence=0.8,
                dependencies=[1],
            ),
        ],
        overall_confidence=0.85,
        completeness_score=0.9,
    )


# =============================================================================
# Unit Tests
# =============================================================================


class TestBridgeInitialization:
    """Test LADDERMultiAgentBridge initialization."""

    def test_initialization_with_config(self, sample_config):
        """Test bridge initializes with configuration."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        assert bridge.cognitive_enabled is True
        assert bridge.config == sample_config
        assert bridge.max_revision_attempts == 3
        assert bridge.darwin_weight == 0.6

    def test_initialization_without_config(self):
        """Test bridge initializes with defaults when no config."""
        bridge = LADDERMultiAgentBridge()

        assert bridge.config == {}
        assert bridge.cognitive_enabled is True

    def test_initialization_cognitive_disabled(self):
        """Test bridge initializes with cognitive modules disabled."""
        config = {"cognitive": {"enabled": False}}
        bridge = LADDERMultiAgentBridge(config=config, enable_cognitive=False)

        assert bridge.cognitive_enabled is False
        assert bridge.cognitive_analyzer is None
        assert bridge.reasoning_verifier is None


class TestQualityAssessment:
    """Test code quality assessment functionality."""

    @pytest.mark.asyncio
    async def test_assess_code_quality_success(
        self, sample_config, sample_analysis_result
    ):
        """Test successful code quality assessment."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        # Mock the cognitive analyzer
        bridge.cognitive_analyzer = Mock()
        bridge.cognitive_analyzer.analyze_code_multimodal = AsyncMock(
            return_value=sample_analysis_result
        )

        code = (
            "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
        )
        requirements = "Recursive factorial function"

        result = await bridge._assess_code_quality(code, requirements)

        assert result is not None
        assert result.understanding_confidence == 0.85
        assert result.quality_prediction["correctness"] == 0.9

    @pytest.mark.asyncio
    async def test_assess_code_quality_disabled(self):
        """Test quality assessment when cognitive modules disabled."""
        bridge = LADDERMultiAgentBridge(enable_cognitive=False)

        result = await bridge._assess_code_quality("code", "requirements")

        assert result is None

    @pytest.mark.asyncio
    async def test_assess_code_quality_error_handling(self, sample_config):
        """Test quality assessment handles errors gracefully."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        # Mock analyzer to raise exception
        bridge.cognitive_analyzer = Mock()
        bridge.cognitive_analyzer.analyze_code_multimodal = AsyncMock(
            side_effect=Exception("Analysis failed")
        )

        result = await bridge._assess_code_quality("code", "requirements")

        assert result is None  # Should return None on error


class TestReasoningContext:
    """Test reasoning chain generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_reasoning_context_success(
        self, sample_config, sample_reasoning_chain
    ):
        """Test successful reasoning chain generation."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        # Mock the reasoning verifier
        bridge.reasoning_verifier = Mock()
        bridge.reasoning_verifier.generate_with_reasoning_chain = AsyncMock(
            return_value=sample_reasoning_chain
        )

        result = await bridge._generate_reasoning_context(
            task_description="Calculate factorial",
            context={"priority": "high"},
        )

        assert result is not None
        assert len(result.steps) == 3
        assert result.overall_confidence == 0.85

    @pytest.mark.asyncio
    async def test_generate_reasoning_context_disabled(self):
        """Test reasoning generation when disabled."""
        bridge = LADDERMultiAgentBridge(enable_cognitive=False)

        result = await bridge._generate_reasoning_context("task description")

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_reasoning_context_config_disabled(
        self, sample_config
    ):
        """Test reasoning generation when disabled by config."""
        config = sample_config.copy()
        config["cognitive"]["integration"]["enable_reasoning_context"] = False
        bridge = LADDERMultiAgentBridge(config=config)

        result = await bridge._generate_reasoning_context("task description")

        assert result is None


class TestCognitiveFitness:
    """Test cognitive fitness calculation."""

    def test_calculate_fitness_with_analysis(
        self, sample_config, sample_analysis_result, sample_reasoning_chain
    ):
        """Test fitness calculation with analysis and reasoning."""
        bridge = LADDERMultiAgentBridge(config=sample_config)
        bridge.cognitive_enabled = True

        darwin_fitness = 0.7

        result = bridge._calculate_cognitive_fitness(
            darwin_fitness=darwin_fitness,
            analysis=sample_analysis_result,
            chain=sample_reasoning_chain,
        )

        # Should be weighted combination
        assert 0 <= result <= 1
        assert (
            result != darwin_fitness
        )  # Should be different from base fitness

    def test_calculate_fitness_without_analysis(self, sample_config):
        """Test fitness calculation without analysis."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        darwin_fitness = 0.7

        result = bridge._calculate_cognitive_fitness(
            darwin_fitness=darwin_fitness,
            analysis=None,
            chain=None,
        )

        assert result == darwin_fitness  # Should return base fitness

    def test_calculate_fitness_disabled(self):
        """Test fitness calculation when cognitive disabled."""
        bridge = LADDERMultiAgentBridge(enable_cognitive=False)

        result = bridge._calculate_cognitive_fitness(
            darwin_fitness=0.7,
            analysis=None,
            chain=None,
        )

        assert result == 0.7


class TestQualityGates:
    """Test quality gate and revision functionality."""

    def test_should_trigger_revision_low_correctness(
        self, sample_config, sample_analysis_result
    ):
        """Test revision triggered by low correctness."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        # Lower correctness below threshold
        analysis = sample_analysis_result
        analysis.quality_prediction["correctness"] = 0.6  # Below 0.7 threshold

        result = bridge._should_trigger_revision(analysis)

        assert result is True

    def test_should_trigger_revision_low_security(
        self, sample_config, sample_analysis_result
    ):
        """Test revision triggered by low security."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        analysis = sample_analysis_result
        analysis.quality_prediction["security"] = 0.7  # Below 0.75 threshold

        result = bridge._should_trigger_revision(analysis)

        assert result is True

    def test_should_not_trigger_revision(
        self, sample_config, sample_analysis_result
    ):
        """Test no revision when quality is acceptable."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        # All scores above thresholds
        result = bridge._should_trigger_revision(sample_analysis_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_trigger_quality_revision(
        self, sample_config, sample_task, sample_analysis_result
    ):
        """Test quality revision workflow."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        # Mock the code generation
        bridge._generate_code_with_multiagent = AsyncMock(
            return_value="# Improved code"
        )

        result = await bridge._trigger_quality_revision(
            code="# Original code",
            analysis=sample_analysis_result,
            task=sample_task,
        )

        assert result == "# Improved code"
        bridge._generate_code_with_multiagent.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegratedWorkflow:
    """Test end-to-end integrated workflow."""

    @pytest.mark.asyncio
    async def test_workflow_with_cognitive_modules(
        self,
        sample_config,
        sample_task,
        sample_analysis_result,
        sample_reasoning_chain,
    ):
        """Test complete workflow with cognitive modules enabled."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        # Mock all cognitive components
        bridge.cognitive_analyzer = Mock()
        bridge.cognitive_analyzer.analyze_code_multimodal = AsyncMock(
            return_value=sample_analysis_result
        )

        bridge.reasoning_verifier = Mock()
        bridge.reasoning_verifier.generate_with_reasoning_chain = AsyncMock(
            return_value=sample_reasoning_chain
        )

        bridge._generate_code_with_multiagent = AsyncMock(
            return_value="def factorial(n): return 1"
        )

        result = await bridge.execute_integrated_workflow(
            ladder_task=sample_task,
            darwin_fitness=0.7,
        )

        assert isinstance(result, CodeGenerationResult)
        assert result.code is not None
        assert result.cognitive_fitness > 0
        assert result.quality_metrics is not None
        assert result.reasoning_chain is not None

    @pytest.mark.asyncio
    async def test_workflow_without_cognitive_modules(self, sample_task):
        """Test workflow with cognitive modules disabled."""
        bridge = LADDERMultiAgentBridge(enable_cognitive=False)

        bridge._generate_code_with_multiagent = AsyncMock(
            return_value="def factorial(n): return 1"
        )

        result = await bridge.execute_integrated_workflow(
            ladder_task=sample_task,
            darwin_fitness=0.7,
        )

        assert isinstance(result, CodeGenerationResult)
        assert result.code is not None
        assert result.cognitive_fitness == 0.7  # Same as Darwin fitness
        assert result.reasoning_chain is None

    @pytest.mark.asyncio
    async def test_workflow_with_revision(self, sample_config, sample_task):
        """Test workflow triggers quality revision."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        # First assessment: low quality
        low_quality_analysis = MultiModalAnalysisResult(
            understanding_confidence=0.5,
            quality_prediction={
                "correctness": 0.6,  # Below threshold
                "maintainability": 0.5,
                "security": 0.7,
                "performance": 0.6,
            },
            improvement_suggestions=["Improve validation"],
            feature_importance={},
        )

        # Second assessment: good quality
        good_quality_analysis = MultiModalAnalysisResult(
            understanding_confidence=0.85,
            quality_prediction={
                "correctness": 0.9,
                "maintainability": 0.8,
                "security": 0.85,
                "performance": 0.75,
            },
            improvement_suggestions=[],
            feature_importance={},
        )

        bridge.cognitive_analyzer = Mock()
        bridge.cognitive_analyzer.analyze_code_multimodal = AsyncMock(
            side_effect=[low_quality_analysis, good_quality_analysis]
        )

        bridge.reasoning_verifier = Mock()
        bridge.reasoning_verifier.generate_with_reasoning_chain = AsyncMock(
            return_value=None
        )

        bridge._generate_code_with_multiagent = AsyncMock(
            return_value="# Improved code"
        )

        result = await bridge.execute_integrated_workflow(
            ladder_task=sample_task,
            darwin_fitness=0.7,
        )

        assert result.revision_count == 1

    @pytest.mark.asyncio
    async def test_workflow_max_revisions(self, sample_config, sample_task):
        """Test workflow respects max revision attempts."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        # Always return low quality
        low_quality_analysis = MultiModalAnalysisResult(
            understanding_confidence=0.5,
            quality_prediction={
                "correctness": 0.5,  # Always below threshold
                "maintainability": 0.5,
                "security": 0.6,
                "performance": 0.5,
            },
            improvement_suggestions=["Needs work"],
            feature_importance={},
        )

        bridge.cognitive_analyzer = Mock()
        bridge.cognitive_analyzer.analyze_code_multimodal = AsyncMock(
            return_value=low_quality_analysis
        )

        bridge.reasoning_verifier = Mock()
        bridge.reasoning_verifier.generate_with_reasoning_chain = AsyncMock(
            return_value=None
        )

        bridge._generate_code_with_multiagent = AsyncMock(
            return_value="# Code"
        )

        result = await bridge.execute_integrated_workflow(
            ladder_task=sample_task,
            darwin_fitness=0.7,
        )

        # Should stop at max attempts
        assert result.revision_count == bridge.max_revision_attempts


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_workflow_performance(
        self, sample_config, sample_task, sample_analysis_result
    ):
        """Test workflow completes in reasonable time."""
        import time

        bridge = LADDERMultiAgentBridge(config=sample_config)

        bridge.cognitive_analyzer = Mock()
        bridge.cognitive_analyzer.analyze_code_multimodal = AsyncMock(
            return_value=sample_analysis_result
        )

        bridge.reasoning_verifier = Mock()
        bridge.reasoning_verifier.generate_with_reasoning_chain = AsyncMock(
            return_value=None
        )

        bridge._generate_code_with_multiagent = AsyncMock(
            return_value="# Code"
        )

        start_time = time.time()

        result = await bridge.execute_integrated_workflow(
            ladder_task=sample_task,
            darwin_fitness=0.7,
        )

        elapsed = time.time() - start_time

        # Should complete quickly with mocked components
        assert elapsed < 1.0  # Less than 1 second
        assert result is not None


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfigurationLoading:
    """Test configuration loading and validation."""

    def test_load_quality_thresholds(self, sample_config):
        """Test quality thresholds loaded from config."""
        bridge = LADDERMultiAgentBridge(config=sample_config)

        thresholds = (
            bridge.config.get("cognitive", {})
            .get("multimodal_analyzer", {})
            .get("quality_thresholds", {})
        )

        assert thresholds.get("correctness_min") == 0.7
        assert thresholds.get("security_min") == 0.75

    def test_config_path_loads_yaml(self, tmp_path):
        """Test configuration is loaded from YAML file."""
        config_data = {
            "cognitive": {
                "enabled": True,
                "multimodal_analyzer": {
                    "quality_thresholds": {
                        "correctness_min": 0.9,
                        "maintainability_min": 0.7,
                        "security_min": 0.8,
                    }
                },
                "integration": {
                    "darwin_weight": 0.55,
                    "max_revision_attempts": 4,
                },
            }
        }
        config_path = tmp_path / "reug_integration.yaml"
        config_path.write_text(
            yaml_utils.safe_dump(config_data), encoding="utf-8"
        )

        bridge = LADDERMultiAgentBridge(config_path=str(config_path))

        thresholds = (
            bridge.config.get("cognitive", {})
            .get("multimodal_analyzer", {})
            .get("quality_thresholds", {})
        )

        assert thresholds.get("correctness_min") == pytest.approx(0.9)
        assert bridge.max_revision_attempts == 4
        assert bridge.darwin_weight == pytest.approx(0.55)
        assert bridge.config_path == config_path

    def test_config_overrides_merge_with_path(self, tmp_path):
        """Test overrides merge correctly with file-backed config."""
        base_config = {
            "cognitive": {
                "integration": {
                    "darwin_weight": 0.45,
                    "max_revision_attempts": 5,
                }
            }
        }
        config_path = tmp_path / "reug_integration.yaml"
        config_path.write_text(
            yaml_utils.safe_dump(base_config), encoding="utf-8"
        )

        overrides = {"cognitive": {"integration": {"darwin_weight": 0.2}}}

        bridge = LADDERMultiAgentBridge(
            config_path=str(config_path),
            config=overrides,
        )

        assert bridge.darwin_weight == pytest.approx(0.2)
        assert bridge.max_revision_attempts == 5

    def test_missing_config_uses_defaults(self):
        """Test missing config values use sensible defaults."""
        bridge = LADDERMultiAgentBridge(config={})

        # Should use default values
        assert bridge.max_revision_attempts == 3
        assert bridge.darwin_weight == 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
