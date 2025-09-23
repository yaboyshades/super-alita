"""
Integration test for complete calculus gate analysis workflow.

These tests MUST FAIL initially as the complete workflow does not exist yet.
Part of TDD Phase 3.2 - T010.
"""

import json
import tempfile
from pathlib import Path


class TestCalculusGateWorkflowIntegration:
    """Integration tests for complete calculus gate analysis workflow."""

    def test_end_to_end_analysis_workflow(self):
        """Test complete analysis workflow from function to certificate."""
        # This test MUST FAIL - no complete workflow exists yet
        from src.calculus_gate import analyze_function_performance

        # Define a simple test function
        def test_function(n):
            """Simple O(n) function for testing."""
            return sum(range(n))

        # Run complete analysis
        result = analyze_function_performance(
            target_function=test_function,
            min_input_size=10,
            max_input_size=1000,
            sample_count=8,
        )

        # Should return a complete certificate
        assert "certificate" in result
        assert "analysis_summary" in result

        certificate = result["certificate"]

        # Certificate should have all required fields
        required_fields = [
            "function_name",
            "build_id",
            "analysis_timestamp",
            "certificate_version",
            "sample_set",
            "first_derivatives",
            "second_derivatives",
            "lipschitz_constant",
            "passes_slope_gate",
            "passes_curvature_gate",
            "passes_lipschitz_gate",
            "overall_compliance",
            "certificate_grade",
        ]

        for field in required_fields:
            assert field in certificate, f"Missing field: {field}"

    def test_sampling_to_certificate_pipeline(self):
        """Test pipeline from sampling to certificate generation."""
        from src.calculus_gate.certificate import PerformanceCertificate
        from src.calculus_gate.fitting import CalculusAnalyzer
        from src.calculus_gate.sampling import RuntimeProfiler

        # Step 1: Sample runtime data
        profiler = RuntimeProfiler()

        def linear_function(n):
            return n * 2

        sample_set = profiler.profile_function(
            linear_function, input_sizes=[10, 20, 50, 100, 200, 500], warmup_runs=3
        )

        # Step 2: Analyze derivatives
        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(sample_set.input_sizes, sample_set.wall_times)

        derivatives = analyzer.compute_first_derivatives()
        second_derivatives = analyzer.compute_second_derivatives()
        lipschitz_constant = analyzer.compute_lipschitz_constant()

        # Step 3: Generate certificate
        certificate = PerformanceCertificate.generate(
            sample_set=sample_set,
            first_derivatives=derivatives,
            second_derivatives=second_derivatives,
            lipschitz_constant=lipschitz_constant,
        )

        # Verify complete pipeline worked
        assert certificate.function_name == "linear_function"
        assert len(certificate.first_derivatives) == len(sample_set.input_sizes)
        assert certificate.certificate_grade in ["A", "B", "F"]

    def test_analysis_with_different_complexity_functions(self):
        """Test analysis workflow with functions of different complexities."""
        from src.calculus_gate import analyze_function_performance

        # O(1) function
        def constant_function(n):
            return 42

        result_constant = analyze_function_performance(constant_function)

        # Should detect low/constant growth
        cert_constant = result_constant["certificate"]
        assert cert_constant["passes_slope_gate"] is True

        # O(n) function
        def linear_function(n):
            return sum(range(n))

        result_linear = analyze_function_performance(linear_function)
        cert_linear = result_linear["certificate"]

        # O(n²) function
        def quadratic_function(n):
            return sum(i * j for i in range(n) for j in range(n))

        result_quadratic = analyze_function_performance(quadratic_function)
        cert_quadratic = result_quadratic["certificate"]

        # Quadratic should have higher slopes than linear
        max_slope_linear = max(cert_linear["first_derivatives"])
        max_slope_quadratic = max(cert_quadratic["first_derivatives"])

        assert max_slope_quadratic > max_slope_linear

    def test_threshold_violation_detection_workflow(self):
        """Test complete workflow when thresholds are violated."""
        from src.calculus_gate import analyze_function_performance

        # Deliberately inefficient function
        def exponential_function(n):
            if n > 20:
                n = 20  # Cap to prevent excessive runtime
            return 2**n

        # Use strict thresholds
        result = analyze_function_performance(
            exponential_function,
            slope_limit=1.0,
            curvature_limit=0.5,
            lipschitz_limit=5.0,
        )

        certificate = result["certificate"]

        # Should detect violations
        assert certificate["overall_compliance"] is False
        assert certificate["certificate_grade"] in ["B", "F"]

        # Should have violations recorded
        if "slope_violations" in certificate:
            assert len(certificate["slope_violations"]) > 0

    def test_confidence_interval_workflow(self):
        """Test complete workflow including confidence interval computation."""
        from src.calculus_gate import analyze_function_performance

        def test_function(n):
            import time

            # Add small random delay to simulate real runtime variation
            time.sleep(0.001 * (1 + 0.1 * hash(n) % 100 / 100))
            return n

        result = analyze_function_performance(
            test_function,
            bootstrap_samples=50,  # Reduced for testing speed
            confidence_level=0.95,
        )

        certificate = result["certificate"]

        # Should have confidence intervals
        if "derivative_confidence_intervals" in certificate:
            intervals = certificate["derivative_confidence_intervals"]

            # Each interval should be [lower, upper]
            for interval in intervals:
                assert len(interval) == 2
                assert interval[0] <= interval[1]

    def test_artifact_generation_workflow(self):
        """Test complete workflow including artifact file generation."""
        from src.calculus_gate import analyze_function_performance

        def test_function(n):
            return n * 2

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "test_certificate.json"

            result = analyze_function_performance(
                test_function, artifact_path=artifact_path
            )

            # Artifact file should be created
            assert artifact_path.exists()

            # File should contain valid JSON certificate
            with open(artifact_path) as f:
                saved_certificate = json.load(f)

            # Should match in-memory certificate
            assert (
                saved_certificate["function_name"]
                == result["certificate"]["function_name"]
            )
            assert (
                saved_certificate["certificate_grade"]
                == result["certificate"]["certificate_grade"]
            )

    def test_error_handling_workflow(self):
        """Test complete workflow error handling and recovery."""
        from src.calculus_gate import analyze_function_performance

        # Function that raises exception
        def failing_function(n):
            if n > 10:
                raise ValueError("Test error")
            return n

        result = analyze_function_performance(failing_function)

        # Should handle gracefully
        assert "error" in result or result["certificate"]["certificate_grade"] == "F"

        # Function with invalid return type
        def invalid_function(n):
            return "not a number"

        result_invalid = analyze_function_performance(invalid_function)

        # Should detect and report error
        assert (
            "error" in result_invalid
            or result_invalid["certificate"]["certificate_grade"] == "F"
        )

    def test_memory_tracking_workflow(self):
        """Test complete workflow including memory usage tracking."""
        from src.calculus_gate import analyze_function_performance

        def memory_intensive_function(n):
            # Allocate temporary memory
            temp_list = list(range(n * 100))
            return len(temp_list)

        result = analyze_function_performance(
            memory_intensive_function, track_memory=True
        )

        certificate = result["certificate"]
        sample_set = certificate["sample_set"]

        # Should have memory tracking data
        assert "memory_peaks" in sample_set
        assert "memory_deltas" in sample_set

        # Memory usage should increase with input size
        memory_peaks = sample_set["memory_peaks"]
        assert len(memory_peaks) > 0

        # Later samples should generally use more memory
        if len(memory_peaks) >= 2:
            assert max(memory_peaks) > min(memory_peaks)

    def test_baseline_comparison_workflow(self):
        """Test workflow with baseline certificate comparison."""
        from src.calculus_gate import analyze_function_performance

        def test_function(n):
            return n * 2

        # Generate baseline
        baseline_result = analyze_function_performance(test_function)
        baseline_cert = baseline_result["certificate"]

        # Generate new analysis with baseline
        result = analyze_function_performance(
            test_function, baseline_certificate=baseline_cert
        )

        certificate = result["certificate"]

        # Should have baseline comparison data
        if "baseline_comparison" in certificate:
            assert certificate["baseline_comparison"] is not None

        if "trend_analysis" in certificate:
            trend = certificate["trend_analysis"]
            assert isinstance(trend, dict)

    def test_parallel_analysis_workflow(self):
        """Test workflow with parallel function analysis."""
        from src.calculus_gate import analyze_multiple_functions

        def function_a(n):
            return n

        def function_b(n):
            return n * 2

        def function_c(n):
            return n**1.5

        functions = [function_a, function_b, function_c]

        results = analyze_multiple_functions(functions)

        # Should have results for all functions
        assert len(results) == 3

        # Each result should be complete
        for result in results:
            assert "certificate" in result
            assert result["certificate"]["certificate_grade"] in ["A", "B", "F"]
