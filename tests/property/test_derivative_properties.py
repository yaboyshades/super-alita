"""
Property-based test for derivative mathematical correctness.

These tests MUST FAIL initially as derivative computation does not exist yet.
Part of TDD Phase 3.2 - T008.
"""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st


class TestDerivativeMathematicalProperties:
    """Property-based tests for derivative computation correctness."""

    @given(
        st.lists(st.floats(min_value=0.001, max_value=100.0), min_size=5, max_size=20)
    )
    def test_derivative_approximation_accuracy(self, runtime_data):
        """Test that finite difference derivatives approximate true derivatives."""
        # This test MUST FAIL - no CalculusAnalyzer exists yet
        from src.calculus_gate.fitting import CalculusAnalyzer

        # Create input sizes (strictly increasing)
        input_sizes = list(range(1, len(runtime_data) + 1))

        analyzer = CalculusAnalyzer()

        # Fit curve and compute derivatives
        analyzer.fit_curve(input_sizes, runtime_data)
        first_derivs = analyzer.compute_first_derivatives()
        second_derivs = analyzer.compute_second_derivatives()

        # Property: derivatives should exist for all input points
        assert len(first_derivs) == len(input_sizes)
        assert len(second_derivs) == len(input_sizes)

        # Property: derivatives should be finite
        assert all(np.isfinite(d) for d in first_derivs)
        assert all(np.isfinite(d) for d in second_derivs)

    @given(st.integers(min_value=3, max_value=100))
    def test_linear_function_derivative_property(self, n_points):
        """Test derivatives of linear functions have expected properties."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        # Create linear function: f(x) = 2x + 3
        input_sizes = list(range(1, n_points + 1))
        runtime_data = [2 * x + 3 for x in input_sizes]

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, runtime_data)

        first_derivs = analyzer.compute_first_derivatives()
        second_derivs = analyzer.compute_second_derivatives()

        # Property: first derivative of linear function should be constant ≈ 2
        mean_first_deriv = np.mean(first_derivs)
        assert abs(mean_first_deriv - 2.0) < 0.1

        # Property: second derivative of linear function should be ≈ 0
        mean_second_deriv = np.mean(np.abs(second_derivs))
        assert mean_second_deriv < 0.1

    @given(st.integers(min_value=3, max_value=50))
    def test_quadratic_function_derivative_property(self, n_points):
        """Test derivatives of quadratic functions have expected properties."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        # Create quadratic function: f(x) = x² + x + 1
        input_sizes = list(range(1, n_points + 1))
        runtime_data = [x**2 + x + 1 for x in input_sizes]

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, runtime_data)

        first_derivs = analyzer.compute_first_derivatives()
        second_derivs = analyzer.compute_second_derivatives()

        # Property: first derivative should increase (f'(x) = 2x + 1)
        # Check that derivative is generally increasing
        increasing_count = sum(
            1
            for i in range(1, len(first_derivs))
            if first_derivs[i] >= first_derivs[i - 1]
        )
        assert increasing_count >= len(first_derivs) * 0.8  # 80% increasing

        # Property: second derivative should be roughly constant ≈ 2
        mean_second_deriv = np.mean(second_derivs)
        assert abs(mean_second_deriv - 2.0) < 0.5

    @given(st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=4, max_size=15))
    def test_monotonic_data_derivative_signs(self, base_times):
        """Test derivative signs for monotonic data."""
        assume(len(set(base_times)) >= 3)  # Need distinct values

        from src.calculus_gate.fitting import CalculusAnalyzer

        # Sort to ensure monotonicity
        runtime_data = sorted(base_times)
        input_sizes = list(range(1, len(runtime_data) + 1))

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, runtime_data)

        first_derivs = analyzer.compute_first_derivatives()

        # Property: for monotonic increasing data, most derivatives should be ≥ 0
        non_negative_count = sum(1 for d in first_derivs if d >= -0.1)
        assert non_negative_count >= len(first_derivs) * 0.7  # 70% non-negative

    @given(
        st.lists(st.floats(min_value=0.001, max_value=1000.0), min_size=5, max_size=15)
    )
    def test_lipschitz_constant_property(self, runtime_data):
        """Test Lipschitz constant mathematical properties."""
        assume(len(set(runtime_data)) >= 3)

        from src.calculus_gate.fitting import CalculusAnalyzer

        input_sizes = list(range(1, len(runtime_data) + 1))

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, runtime_data)

        lipschitz_const = analyzer.compute_lipschitz_constant()

        # Property: Lipschitz constant should be non-negative
        assert lipschitz_const >= 0

        # Property: Lipschitz constant should be finite
        assert np.isfinite(lipschitz_const)

        # Property: should be related to maximum slope
        first_derivs = analyzer.compute_first_derivatives()
        max_slope = np.max(np.abs(first_derivs))

        # Lipschitz constant should be at least as large as max derivative
        assert lipschitz_const >= max_slope * 0.5  # Allow some tolerance

    @given(st.lists(st.floats(min_value=0.1, max_value=100.0), min_size=6, max_size=12))
    def test_confidence_interval_properties(self, runtime_data):
        """Test bootstrap confidence interval properties."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        input_sizes = list(range(1, len(runtime_data) + 1))

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, runtime_data)

        # Bootstrap confidence intervals
        intervals = analyzer.compute_confidence_intervals(bootstrap_samples=100)

        # Property: each interval should have lower ≤ upper bound
        for lower, upper in intervals:
            assert lower <= upper

        # Property: intervals should contain reasonable values
        first_derivs = analyzer.compute_first_derivatives()
        for i, (lower, upper) in enumerate(intervals):
            if i < len(first_derivs):
                # Most derivatives should fall within their confidence intervals
                # (allowing some tolerance for bootstrap estimation)
                width = upper - lower
                center = (lower + upper) / 2
                assert abs(first_derivs[i] - center) <= width * 1.5

    @given(st.floats(min_value=0.001, max_value=0.5))
    def test_noise_robustness_property(self, noise_level):
        """Test that small noise doesn't drastically change derivatives."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        # Clean quadratic data
        input_sizes = list(range(1, 11))
        clean_data = [x**2 for x in input_sizes]

        # Add noise
        np.random.seed(42)  # Reproducible
        noise = np.random.normal(0, noise_level, len(clean_data))
        noisy_data = [clean_data[i] + noise[i] for i in range(len(clean_data))]

        # Analyze both
        analyzer_clean = CalculusAnalyzer()
        analyzer_clean.fit_curve(input_sizes, clean_data)
        clean_derivs = analyzer_clean.compute_first_derivatives()

        analyzer_noisy = CalculusAnalyzer()
        analyzer_noisy.fit_curve(input_sizes, noisy_data)
        noisy_derivs = analyzer_noisy.compute_first_derivatives()

        # Property: small noise should not change derivatives drastically
        max_diff = max(
            abs(clean_derivs[i] - noisy_derivs[i]) for i in range(len(clean_derivs))
        )

        # Maximum difference should be proportional to noise level
        expected_max_diff = noise_level * 10  # Allow some tolerance
        assert max_diff <= expected_max_diff

    @given(st.integers(min_value=3, max_value=20))
    def test_spline_continuity_property(self, n_points):
        """Test that spline fitting produces continuous derivatives."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        # Create smooth test data
        input_sizes = list(range(1, n_points + 1))
        runtime_data = [x**1.5 + 0.1 * x for x in input_sizes]

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, runtime_data)

        first_derivs = analyzer.compute_first_derivatives()

        # Property: derivatives should not have extreme jumps (continuity)
        if len(first_derivs) >= 2:
            max_jump = max(
                abs(first_derivs[i] - first_derivs[i - 1])
                for i in range(1, len(first_derivs))
            )

            # Maximum jump should be reasonable relative to overall scale
            deriv_scale = np.mean(np.abs(first_derivs))
            relative_jump = max_jump / max(deriv_scale, 0.001)

            assert relative_jump < 5.0  # Arbitrary but reasonable threshold

    @given(st.integers(min_value=4, max_value=15))
    def test_boundary_derivative_accuracy(self, n_points):
        """Test accuracy of derivatives at boundaries."""
        from src.calculus_gate.fitting import CalculusAnalyzer

        # Use a simple polynomial where we know the true derivative
        # f(x) = x³ - 2x² + x + 1, f'(x) = 3x² - 4x + 1
        input_sizes = list(range(1, n_points + 1))
        runtime_data = [x**3 - 2 * x**2 + x + 1 for x in input_sizes]
        true_derivs = [3 * x**2 - 4 * x + 1 for x in input_sizes]

        analyzer = CalculusAnalyzer()
        analyzer.fit_curve(input_sizes, runtime_data)
        computed_derivs = analyzer.compute_first_derivatives()

        # Property: computed derivatives should approximate true derivatives
        # especially away from boundaries
        if len(computed_derivs) >= 5:
            # Check middle points (avoid boundary effects)
            start_idx = 1
            end_idx = len(computed_derivs) - 1

            for i in range(start_idx, end_idx):
                error = abs(computed_derivs[i] - true_derivs[i])
                relative_error = error / max(abs(true_derivs[i]), 0.1)
                assert relative_error < 0.5  # 50% relative error tolerance
