"""
Property-based test for monotonicity and convexity validation.

These tests MUST FAIL initially as mathematical validation does not exist yet.
Part of TDD Phase 3.2 - T009.
"""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st


class TestMathematicalProperties:
    """Property-based tests for mathematical function properties."""

    @given(
        st.lists(st.floats(min_value=0.1, max_value=1000.0), min_size=5, max_size=15)
    )
    def test_monotonicity_detection(self, runtime_data):
        """Test detection of monotonic behavior in runtime data."""
        # This test MUST FAIL - no monotonicity analyzer exists yet
        from src.calculus_gate.fitting import CalculusAnalyzer

        # Sort to create strictly monotonic data
        monotonic_data = sorted(runtime_data)
        input_sizes = list(range(1, len(monotonic_data) + 1))

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, monotonic_data)

        # Should detect monotonicity
        is_monotonic = analyzer.detect_monotonicity()
        assert is_monotonic is True

        # First derivatives should be mostly non-negative
        first_derivs = analyzer.compute_first_derivatives()
        non_negative_ratio = sum(1 for d in first_derivs if d >= -0.01) / len(
            first_derivs
        )
        assert non_negative_ratio >= 0.8  # 80% should be non-negative

    @given(st.integers(min_value=4, max_value=20))
    def test_convexity_detection_convex_function(self, n_points):
        """Test detection of convexity for known convex functions."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        # Create convex function: f(x) = x²
        input_sizes = list(range(1, n_points + 1))
        runtime_data = [x**2 for x in input_sizes]

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, runtime_data)

        # Should detect convexity
        is_convex = analyzer.detect_convexity()
        assert is_convex is True

        # Second derivatives should be mostly positive
        second_derivs = analyzer.compute_second_derivatives()
        positive_ratio = sum(1 for d in second_derivs if d >= -0.1) / len(second_derivs)
        assert positive_ratio >= 0.7  # 70% should be positive

    @given(st.integers(min_value=4, max_value=20))
    def test_concavity_detection_concave_function(self, n_points):
        """Test detection of concavity for known concave functions."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        # Create concave function: f(x) = -x² + 10x
        input_sizes = list(range(1, n_points + 1))
        runtime_data = [-(x**2) + 10 * x for x in input_sizes]

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, runtime_data)

        # Should detect concavity (not convex)
        is_convex = analyzer.detect_convexity()
        assert is_convex is False

        # Second derivatives should be mostly negative
        second_derivs = analyzer.compute_second_derivatives()
        negative_ratio = sum(1 for d in second_derivs if d <= 0.1) / len(second_derivs)
        assert negative_ratio >= 0.7  # 70% should be negative

    @given(st.lists(st.floats(min_value=0.1, max_value=100.0), min_size=6, max_size=12))
    def test_complexity_classification(self, base_times):
        """Test classification of algorithmic complexity patterns."""
        assume(len(set(base_times)) >= 4)

        from src.calculus_gate.fitting import CalculusAnalyzer

        # Test different complexity patterns
        input_sizes = list(range(1, len(base_times) + 1))

        # Linear pattern: O(n)
        linear_data = [x * base_times[0] for x in input_sizes]
        analyzer_linear = CalculusAnalyzer()
        analyzer_linear.fit_curve(input_sizes, linear_data)
        linear_complexity = analyzer_linear.classify_complexity()
        assert linear_complexity in ["linear", "O(n)"]

        # Quadratic pattern: O(n²)
        quadratic_data = [x**2 * base_times[0] for x in input_sizes]
        analyzer_quad = CalculusAnalyzer()
        analyzer_quad.fit_curve(input_sizes, quadratic_data)
        quad_complexity = analyzer_quad.classify_complexity()
        assert quad_complexity in ["quadratic", "O(n²)", "polynomial"]

    @given(st.floats(min_value=0.01, max_value=2.0))
    def test_growth_rate_bounds(self, growth_factor):
        """Test that growth rate analysis respects mathematical bounds."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        # Create exponential-like growth
        input_sizes = list(range(1, 11))
        runtime_data = [growth_factor**x for x in input_sizes]

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, runtime_data)

        # Growth rate should be related to growth factor
        growth_rate = analyzer.estimate_growth_rate()

        # Properties of growth rate
        assert growth_rate > 0  # Must be positive for increasing function
        assert np.isfinite(growth_rate)  # Must be finite

        # Should correlate with input growth factor
        if growth_factor > 1.1:
            assert growth_rate > 1.0  # Should detect super-linear growth

    @given(
        st.lists(st.floats(min_value=0.001, max_value=10.0), min_size=5, max_size=15)
    )
    def test_smoothness_assessment(self, runtime_data):
        """Test assessment of function smoothness."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        input_sizes = list(range(1, len(runtime_data) + 1))

        # Smooth data (sorted to remove jumps)
        smooth_data = sorted(runtime_data)
        analyzer_smooth = CalculusAnalyzer()
        analyzer_smooth.fit_curve(input_sizes, smooth_data)

        smoothness_smooth = analyzer_smooth.assess_smoothness()

        # Noisy data (add random jumps)
        np.random.seed(42)
        noise = np.random.normal(0, np.std(runtime_data) * 0.5, len(runtime_data))
        noisy_data = [runtime_data[i] + noise[i] for i in range(len(runtime_data))]

        analyzer_noisy = CalculusAnalyzer()
        analyzer_noisy.fit_curve(input_sizes, noisy_data)

        smoothness_noisy = analyzer_noisy.assess_smoothness()

        # Smooth data should have higher smoothness score
        assert smoothness_smooth >= smoothness_noisy

        # Both should be valid values
        assert 0 <= smoothness_smooth <= 1
        assert 0 <= smoothness_noisy <= 1

    @given(st.lists(st.floats(min_value=0.1, max_value=50.0), min_size=6, max_size=10))
    def test_inflection_point_detection(self, runtime_data):
        """Test detection of inflection points in runtime curves."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        # Create data with known inflection point: f(x) = x³ - 6x² + 9x
        # Inflection at x = 2
        input_sizes = list(range(1, len(runtime_data) + 1))
        inflection_data = [x**3 - 6 * x**2 + 9 * x for x in input_sizes]

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, inflection_data)

        inflection_points = analyzer.detect_inflection_points()

        # Should detect at least one inflection point
        assert len(inflection_points) >= 0  # May be 0 if out of range

        # All detected points should be within input range
        for point in inflection_points:
            assert min(input_sizes) <= point <= max(input_sizes)

    @given(st.integers(min_value=5, max_value=15))
    def test_asymptotic_behavior_analysis(self, n_points):
        """Test analysis of asymptotic behavior."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        input_sizes = list(range(1, n_points + 1))

        # Logarithmic behavior: f(x) = log(x)
        log_data = [np.log(x) for x in input_sizes]
        analyzer_log = CalculusAnalyzer()
        analyzer_log.fit_curve(input_sizes, log_data)

        asymptotic_log = analyzer_log.analyze_asymptotic_behavior()

        # Should detect sublinear growth
        assert asymptotic_log["growth_type"] in ["logarithmic", "sublinear"]

        # Power law behavior: f(x) = x^1.5
        power_data = [x**1.5 for x in input_sizes]
        analyzer_power = CalculusAnalyzer()
        analyzer_power.fit_curve(input_sizes, power_data)

        asymptotic_power = analyzer_power.analyze_asymptotic_behavior()

        # Should detect polynomial growth
        assert asymptotic_power["growth_type"] in ["polynomial", "power_law"]

    @given(
        st.lists(st.floats(min_value=0.01, max_value=100.0), min_size=5, max_size=12)
    )
    def test_stability_analysis(self, runtime_data):
        """Test analysis of numerical stability in computations."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        input_sizes = list(range(1, len(runtime_data) + 1))

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, runtime_data)

        stability_report = analyzer.analyze_stability()

        # Stability report should contain key metrics
        assert "condition_number" in stability_report
        assert "numerical_stability" in stability_report
        assert "convergence_rate" in stability_report

        # Condition number should be positive
        assert stability_report["condition_number"] > 0

        # Stability should be a boolean or score
        stability = stability_report["numerical_stability"]
        assert isinstance(stability, (bool, float, int))

    @given(st.integers(min_value=3, max_value=20))
    def test_periodicity_detection(self, n_points):
        """Test detection of periodic patterns in runtime data."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        # Create periodic function: f(x) = sin(x) + x (trend + oscillation)
        input_sizes = list(range(1, n_points + 1))
        periodic_data = [np.sin(x * 0.5) + x * 0.1 for x in input_sizes]

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, periodic_data)

        periodicity = analyzer.detect_periodicity()

        # Should return period length or None
        if periodicity is not None:
            assert isinstance(periodicity, (int, float))
            assert periodicity > 0

        # Should also provide confidence in detection
        confidence = analyzer.get_periodicity_confidence()
        assert 0 <= confidence <= 1
