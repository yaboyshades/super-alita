"""Tests for cognitive modules (multimodal analyzer and reasoning chain verifier)."""

import json

import pytest

torch = pytest.importorskip("torch")

from src.cognitive_modules import (
    MultiModalAnalysisResult,
    MultiModalCodeAnalyzer,
    ReasoningChain,
    ReasoningChainVerifier,
    ReasoningStepType,
)
from src.reug_runtime.llm_client import LLMClient


@pytest.fixture
def multimodal_analyzer():
    """Create multimodal analyzer instance."""
    config = {"device": "cpu", "cache_enabled": True}
    return MultiModalCodeAnalyzer(config)


@pytest.fixture
def reasoning_verifier():
    """Create reasoning chain verifier instance."""
    config = {"device": "cpu", "verification_strict": False}
    return ReasoningChainVerifier(config)


@pytest.fixture
def sample_code():
    """Sample Python code for testing."""
    return '''
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)


class MathHelper:
    """Helper class for mathematical operations."""
    
    def __init__(self):
        self.cache = {}
    
    def factorial(self, n: int) -> int:
        """Calculate factorial of n."""
        if n in self.cache:
            return self.cache[n]
        
        if n <= 1:
            result = 1
        else:
            result = n * self.factorial(n - 1)
        
        self.cache[n] = result
        return result
'''


@pytest.fixture
def sample_requirements():
    """Sample requirements for testing."""
    return """
    Create a Python module that implements:
    1. A function to calculate Fibonacci numbers
    2. A class with a memoized factorial method
    3. Proper error handling for edge cases
    4. Type hints for all functions
    """


# --- MultiModalCodeAnalyzer Tests ---


@pytest.mark.asyncio
async def test_multimodal_analyzer_basic(multimodal_analyzer, sample_code):
    """Test basic multimodal analysis."""
    result = await multimodal_analyzer.analyze_code_multimodal(sample_code)

    assert isinstance(result, MultiModalAnalysisResult)
    assert 0.0 <= result.understanding_confidence <= 1.0
    assert isinstance(result.code_intent, dict)
    assert isinstance(result.quality_prediction, dict)
    assert isinstance(result.improvement_suggestions, list)


@pytest.mark.asyncio
async def test_multimodal_analyzer_quality_dimensions(
    multimodal_analyzer, sample_code
):
    """Test quality prediction dimensions."""
    result = await multimodal_analyzer.analyze_code_multimodal(sample_code)

    expected_dims = [
        "correctness",
        "readability",
        "maintainability",
        "performance",
        "security",
        "creativity",
        "documentation",
    ]

    for dim in expected_dims:
        assert dim in result.quality_prediction
        assert 0.0 <= result.quality_prediction[dim] <= 1.0


@pytest.mark.asyncio
async def test_multimodal_analyzer_intent_classification(
    multimodal_analyzer, sample_code
):
    """Test code intent classification."""
    result = await multimodal_analyzer.analyze_code_multimodal(sample_code)

    assert isinstance(result.code_intent, dict)
    assert len(result.code_intent) > 0

    # Intent probabilities should sum to approximately 1.0
    total_prob = sum(result.code_intent.values())
    assert 0.8 <= total_prob <= 1.2  # Allow some numerical tolerance


@pytest.mark.asyncio
async def test_multimodal_analyzer_structural_complexity(
    multimodal_analyzer, sample_code
):
    """Test structural complexity analysis."""
    result = await multimodal_analyzer.analyze_code_multimodal(sample_code)

    assert isinstance(result.structural_complexity, dict)

    # Check for expected complexity metrics
    if "error" not in result.structural_complexity:
        assert "cyclomatic_complexity" in result.structural_complexity
        assert "max_nesting_depth" in result.structural_complexity
        assert "function_count" in result.structural_complexity
        assert result.structural_complexity["function_count"] >= 2


@pytest.mark.asyncio
async def test_multimodal_analyzer_with_requirements(
    multimodal_analyzer, sample_code, sample_requirements
):
    """Test analysis with requirements for alignment checking."""
    result = await multimodal_analyzer.analyze_code_multimodal(
        sample_code, requirements=sample_requirements
    )

    assert isinstance(result, MultiModalAnalysisResult)
    assert -1.0 <= result.requirement_alignment <= 1.0


@pytest.mark.asyncio
async def test_multimodal_analyzer_caching(multimodal_analyzer, sample_code):
    """Test analysis result caching."""
    # First analysis
    result1 = await multimodal_analyzer.analyze_code_multimodal(sample_code)

    # Second analysis (should be cached)
    result2 = await multimodal_analyzer.analyze_code_multimodal(sample_code)

    # Results should be identical (same object from cache)
    assert result1.understanding_confidence == result2.understanding_confidence
    assert result1.code_intent == result2.code_intent


@pytest.mark.asyncio
async def test_multimodal_analyzer_invalid_code(multimodal_analyzer):
    """Test analyzer with invalid code."""
    invalid_code = "def broken( syntax error"

    result = await multimodal_analyzer.analyze_code_multimodal(invalid_code)

    # Should still return a result (with low confidence)
    assert isinstance(result, MultiModalAnalysisResult)
    assert result.understanding_confidence >= 0.0


# --- Graph Neural Network Tests ---


@pytest.mark.asyncio
async def test_graph_features_extraction(multimodal_analyzer, sample_code):
    """Test AST to graph conversion and feature extraction."""
    graph_features = await multimodal_analyzer._extract_graph_features(
        sample_code
    )

    assert isinstance(graph_features, torch.Tensor)
    assert graph_features.shape[0] == 256  # Output dimension


@pytest.mark.asyncio
async def test_ast_to_graph_conversion(multimodal_analyzer):
    """Test AST to graph data conversion."""
    simple_code = "def foo(): return 42"

    import ast

    tree = ast.parse(simple_code)
    graph_data = await multimodal_analyzer._ast_to_graph(tree)

    assert "node_features" in graph_data
    assert "adjacency_matrix" in graph_data
    assert "nodes" in graph_data

    assert graph_data["node_features"].shape[1] == 64  # Feature dimension
    assert (
        graph_data["adjacency_matrix"].shape[0]
        == graph_data["adjacency_matrix"].shape[1]
    )


# --- ReasoningChainVerifier Tests ---


@pytest.mark.asyncio
async def test_reasoning_chain_generation(
    reasoning_verifier, sample_requirements
):
    """Test reasoning chain generation."""
    chain = await reasoning_verifier.generate_with_reasoning_chain(
        sample_requirements
    )

    assert isinstance(chain, ReasoningChain)
    assert len(chain.steps) > 0
    assert 0.0 <= chain.overall_confidence <= 1.0
    assert 0.0 <= chain.logical_consistency_score <= 1.0
    assert 0.0 <= chain.completeness_score <= 1.0


@pytest.mark.asyncio
async def test_reasoning_chain_metadata_flags(
    reasoning_verifier, sample_requirements
):
    """Metadata should reveal heuristic operation."""

    chain = await reasoning_verifier.generate_with_reasoning_chain(
        sample_requirements
    )

    assert chain.metadata.get("generation_mode") == "heuristic"
    assert chain.metadata.get("uses_heuristics") is True


@pytest.mark.asyncio
async def test_reasoning_chain_step_types(
    reasoning_verifier, sample_requirements
):
    """Test that reasoning chain includes required step types."""
    chain = await reasoning_verifier.generate_with_reasoning_chain(
        sample_requirements
    )

    step_types = {step.step_type for step in chain.steps}

    # Should include key reasoning steps
    assert ReasoningStepType.PROBLEM_ANALYSIS in step_types
    assert ReasoningStepType.SOLUTION_STRATEGY in step_types
    assert ReasoningStepType.CODE_GENERATION in step_types


@pytest.mark.asyncio
async def test_reasoning_chain_verifier_llm_mode(sample_requirements):
    """LLM-backed generation should surface llm metadata when available."""

    class DummyLLM(LLMClient):
        def __init__(self) -> None:
            self.model_name = "dummy-llm"

        async def stream_chat(  # type: ignore[override]
            self,
            messages,
            *,
            tools=None,
            timeout=None,
        ):
            payload = {
                "steps": [
                    {
                        "step_type": "problem_analysis",
                        "description": "Understand the problem",
                        "reasoning": "Identify key requirements",
                        "confidence": 0.9,
                        "dependencies": [],
                        "evidence": {"notes": "LLM"},
                    },
                    {
                        "step_type": "solution_strategy",
                        "description": "Plan solution",
                        "reasoning": "Outline approach",
                        "confidence": 0.85,
                        "dependencies": [0],
                        "evidence": {},
                    },
                ]
            }
            yield {"content": json.dumps(payload)}

    verifier = ReasoningChainVerifier(
        {
            "use_llm": True,
            "allow_heuristic_fallback": False,
            "llm_client": DummyLLM(),
        }
    )

    chain = await verifier.generate_with_reasoning_chain(sample_requirements)

    assert chain.metadata.get("generation_mode") == "llm"
    assert chain.metadata.get("llm_used") is True
    assert (
        chain.steps
        and chain.steps[0].step_type == ReasoningStepType.PROBLEM_ANALYSIS
    )


@pytest.mark.asyncio
async def test_reasoning_chain_dependencies(
    reasoning_verifier, sample_requirements
):
    """Test reasoning step dependencies are valid."""
    chain = await reasoning_verifier.generate_with_reasoning_chain(
        sample_requirements
    )

    for i, step in enumerate(chain.steps):
        # Dependencies should reference earlier steps
        for dep_idx in step.dependencies:
            assert 0 <= dep_idx < i


@pytest.mark.asyncio
async def test_reasoning_chain_verification_status(
    reasoning_verifier, sample_requirements
):
    """Test that steps have verification status."""
    chain = await reasoning_verifier.generate_with_reasoning_chain(
        sample_requirements
    )

    for step in chain.steps:
        assert step.verification_status in ["verified", "needs_review", None]


@pytest.mark.asyncio
async def test_reasoning_chain_code_generation(
    reasoning_verifier, sample_requirements
):
    """Test code generation from reasoning chain."""
    chain = await reasoning_verifier.generate_with_reasoning_chain(
        sample_requirements
    )

    # Code generation depends on verification passing
    # Check that the chain structure is correct
    assert isinstance(chain, ReasoningChain)
    assert len(chain.steps) > 0

    # If code was generated, validate it
    if chain.generated_code is not None:
        assert isinstance(chain.generated_code, str)
        assert len(chain.generated_code) > 0


@pytest.mark.asyncio
async def test_consistency_checker():
    """Test logical consistency checker."""
    from src.cognitive_modules.reasoning_chain_verifier import (
        LogicalConsistencyChecker,
        ReasoningStep,
    )

    checker = LogicalConsistencyChecker()

    # Create test steps
    step1 = ReasoningStep(
        step_type=ReasoningStepType.PROBLEM_ANALYSIS,
        description="Analyze problem",
        reasoning="The problem requires sorting a list",
        confidence=0.9,
        dependencies=[],
    )

    step2 = ReasoningStep(
        step_type=ReasoningStepType.SOLUTION_STRATEGY,
        description="Choose solution",
        reasoning="Use quicksort algorithm for efficient sorting",
        confidence=0.85,
        dependencies=[0],
    )

    result = await checker.check_step_consistency(step2, [step1])

    assert "is_consistent" in result
    assert "consistency_score" in result
    assert 0.0 <= result["consistency_score"] <= 1.0


@pytest.mark.asyncio
async def test_completeness_analyzer():
    """Test completeness analyzer."""
    from src.cognitive_modules.reasoning_chain_verifier import (
        CompletenessAnalyzer,
        ReasoningStep,
    )

    analyzer = CompletenessAnalyzer()

    # Create incomplete chain (missing some steps)
    incomplete_chain = [
        ReasoningStep(
            step_type=ReasoningStepType.PROBLEM_ANALYSIS,
            description="Analyze",
            reasoning="Problem analysis",
            confidence=0.9,
        ),
        ReasoningStep(
            step_type=ReasoningStepType.CODE_GENERATION,
            description="Generate",
            reasoning="Generate code",
            confidence=0.8,
        ),
    ]

    result = await analyzer.analyze_completeness(incomplete_chain)

    assert "is_complete" in result
    assert "completeness_score" in result
    assert "missing_step_types" in result

    # Should detect missing steps
    assert not result["is_complete"]
    assert len(result["missing_step_types"]) > 0


@pytest.mark.asyncio
async def test_factual_correctness_verifier():
    """Test factual correctness verifier."""
    from src.cognitive_modules.reasoning_chain_verifier import (
        FactualCorrectnessVerifier,
        ReasoningStep,
    )

    verifier = FactualCorrectnessVerifier()

    step = ReasoningStep(
        step_type=ReasoningStepType.SOLUTION_STRATEGY,
        description="Strategy",
        reasoning="Python uses indentation for code blocks. "
        "Lists are mutable data structures.",
        confidence=0.9,
    )

    result = await verifier.verify_factual_correctness(step)

    assert "is_correct" in result
    assert "correctness_score" in result
    assert 0.0 <= result["correctness_score"] <= 1.0


# --- Integration Tests ---


@pytest.mark.asyncio
async def test_multimodal_and_reasoning_integration(
    multimodal_analyzer, reasoning_verifier, sample_code, sample_requirements
):
    """Test integration of multimodal analysis and reasoning chain."""
    # Analyze existing code
    analysis = await multimodal_analyzer.analyze_code_multimodal(
        sample_code, requirements=sample_requirements
    )

    # Generate new code with reasoning
    chain = await reasoning_verifier.generate_with_reasoning_chain(
        sample_requirements
    )

    # Both should succeed
    assert analysis.understanding_confidence > 0.0
    assert chain.overall_confidence > 0.0


@pytest.mark.asyncio
async def test_quality_improvement_workflow(
    multimodal_analyzer, reasoning_verifier, sample_code
):
    """Test workflow for iterative quality improvement."""
    # Analyze code to identify issues
    analysis = await multimodal_analyzer.analyze_code_multimodal(sample_code)

    # If quality is low, generate reasoning for improvements
    if analysis.quality_prediction.get("maintainability", 1.0) < 0.7:
        improvement_req = (
            "Improve code maintainability based on these suggestions: "
            + ", ".join(analysis.improvement_suggestions[:3])
        )

        chain = await reasoning_verifier.generate_with_reasoning_chain(
            improvement_req
        )

        assert chain.overall_confidence > 0.0


@pytest.mark.asyncio
async def test_cognitive_modules_with_minimal_code(
    multimodal_analyzer, reasoning_verifier
):
    """Test cognitive modules with minimal code snippets."""
    minimal_code = "print('hello')"

    # Multimodal analysis
    analysis = await multimodal_analyzer.analyze_code_multimodal(minimal_code)
    assert isinstance(analysis, MultiModalAnalysisResult)

    # Reasoning chain for simple task
    simple_req = "Print a hello message"
    chain = await reasoning_verifier.generate_with_reasoning_chain(simple_req)
    assert isinstance(chain, ReasoningChain)


@pytest.mark.asyncio
async def test_cognitive_modules_with_complex_code(
    multimodal_analyzer, reasoning_verifier
):
    """Test cognitive modules with complex code."""
    complex_code = """
import asyncio
from typing import List, Dict, Any

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = []
    
    async def process_batch(self, items: List[Any]) -> List[Any]:
        tasks = [self._process_item(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.results.extend(results)
        return results
    
    async def _process_item(self, item: Any) -> Any:
        # Simulate processing
        await asyncio.sleep(0.1)
        return item.upper() if isinstance(item, str) else item
"""

    # Multimodal analysis should handle complexity
    analysis = await multimodal_analyzer.analyze_code_multimodal(complex_code)
    assert analysis.understanding_confidence > 0.0

    # Check structural complexity is captured
    assert (
        "cyclomatic_complexity" in analysis.structural_complexity
        or "error" in analysis.structural_complexity
    )


@pytest.mark.asyncio
async def test_multimodal_analyzer_deterministic_mode(sample_code):
    """Deterministic embeddings should yield repeatable quality scores."""

    config = {
        "device": "cpu",
        "cache_enabled": False,
        "deterministic_embeddings": True,
        "deterministic_seed": 1234,
    }
    analyzer = MultiModalCodeAnalyzer(config)

    first = await analyzer.analyze_code_multimodal(sample_code)
    second = await analyzer.analyze_code_multimodal(sample_code)

    assert first.quality_prediction == second.quality_prediction
    assert first.execution_insights.get("analysis_mode") == "deterministic"


# --- Performance Tests ---


@pytest.mark.asyncio
async def test_multimodal_analyzer_performance(multimodal_analyzer):
    """Test multimodal analyzer performance."""
    import time

    code = "def test(): return 42"

    start = time.time()
    result = await multimodal_analyzer.analyze_code_multimodal(code)
    elapsed = time.time() - start

    # Should complete reasonably fast
    assert elapsed < 5.0  # 5 seconds threshold
    assert isinstance(result, MultiModalAnalysisResult)


@pytest.mark.asyncio
async def test_reasoning_verifier_performance(reasoning_verifier):
    """Test reasoning verifier performance."""
    import time

    requirements = "Create a simple function"

    start = time.time()
    chain = await reasoning_verifier.generate_with_reasoning_chain(
        requirements
    )
    elapsed = time.time() - start

    # Should complete reasonably fast
    assert elapsed < 5.0  # 5 seconds threshold
    assert isinstance(chain, ReasoningChain)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
