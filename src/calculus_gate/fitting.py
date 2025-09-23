"""
Calculus analysis and derivative computation for runtime curves.

This module provides facilities to fit smooth curves to runtime data
and compute mathematical derivatives for performance analysis.
"""

from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.signal import savgol_filter


class CalculusAnalyzer:
    """Analyzes runtime curves using calculus-based methods."""

    def __init__(self):
        """Initialize the calculus analyzer."""
        self.input_sizes: list[int] | None = None
        self.runtime_data: list[float] | None = None
        self.fitted_spline: CubicSpline | None = None
        self.fitting_method: str = "not_fitted"
        self.fitting_quality_score: float = 0.0
        self.noise_handling_applied: bool = False

    def fit_curve(self, input_sizes: list[int], runtime_data: list[float]) -> None:
        """
        Fit a smooth curve to runtime data.

        Args:
            input_sizes: Input sizes (x-axis)
            runtime_data: Runtime measurements (y-axis)
        """
        if len(input_sizes) != len(runtime_data):
            raise ValueError("input_sizes and runtime_data must have same length")
        if len(input_sizes) < 3:
            raise ValueError("Need at least 3 data points for curve fitting")

        self.input_sizes = input_sizes.copy()
        self.runtime_data = runtime_data.copy()

        # Try cubic spline first
        try:
            self._fit_cubic_spline()
        except Exception:
            # Fall back to Savitzky-Golay smoothing
            try:
                self._fit_savgol_fallback()
            except Exception:
                # Last resort: linear interpolation
                self._fit_linear_fallback()

    def _fit_cubic_spline(self) -> None:
        """Fit cubic spline to the data."""
        x = np.array(self.input_sizes, dtype=float)
        y = np.array(self.runtime_data, dtype=float)

        # Check for valid data
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("Invalid runtime data (NaN or Inf)")

        # Create cubic spline
        self.fitted_spline = CubicSpline(x, y)
        self.fitting_method = "cubic_spline"

        # Calculate R² score
        y_pred = self.fitted_spline(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot > 0:
            self.fitting_quality_score = 1 - (ss_res / ss_tot)
        else:
            self.fitting_quality_score = 1.0

    def _fit_savgol_fallback(self) -> None:
        """Fallback using Savitzky-Golay smoothing."""
        x = np.array(self.input_sizes, dtype=float)
        y = np.array(self.runtime_data, dtype=float)

        # Apply Savitzky-Golay filter for smoothing
        window_length = min(len(y), 5) if len(y) % 2 == 1 else min(len(y) - 1, 5)
        if window_length < 3:
            window_length = 3

        polyorder = min(2, window_length - 1)
        y_smooth = savgol_filter(y, window_length, polyorder)

        # Fit spline to smoothed data
        self.fitted_spline = CubicSpline(x, y_smooth)
        self.fitting_method = "savgol_fallback"
        self.noise_handling_applied = True

        # Calculate R² against original data
        y_pred = self.fitted_spline(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot > 0:
            self.fitting_quality_score = 1 - (ss_res / ss_tot)
        else:
            self.fitting_quality_score = 1.0

    def _fit_linear_fallback(self) -> None:
        """Last resort: piecewise linear interpolation."""
        x = np.array(self.input_sizes, dtype=float)
        y = np.array(self.runtime_data, dtype=float)

        # Use linear interpolation (degree=1 spline)
        self.fitted_spline = UnivariateSpline(x, y, k=1, s=0)
        self.fitting_method = "linear_fallback"
        self.fitting_quality_score = 0.5  # Moderate quality for linear fit

    def compute_first_derivatives(self) -> list[float]:
        """
        Compute first derivatives (df/dn) at each input size.

        Returns:
            List of first derivative values
        """
        if self.fitted_spline is None or self.input_sizes is None:
            raise ValueError("Must call fit_curve() first")

        x = np.array(self.input_sizes, dtype=float)

        # Get first derivative from spline
        first_deriv = self.fitted_spline.derivative(1)
        derivatives = first_deriv(x)

        return derivatives.tolist()

    def compute_second_derivatives(self) -> list[float]:
        """
        Compute second derivatives (d²f/dn²) at each input size.

        Returns:
            List of second derivative values
        """
        if self.fitted_spline is None or self.input_sizes is None:
            raise ValueError("Must call fit_curve() first")

        x = np.array(self.input_sizes, dtype=float)

        # Get second derivative from spline
        second_deriv = self.fitted_spline.derivative(2)
        derivatives = second_deriv(x)

        return derivatives.tolist()

    def compute_lipschitz_constant(self) -> float:
        """
        Compute Lipschitz constant: max |f(x1) - f(x2)| / |x1 - x2|.

        Returns:
            Lipschitz constant value
        """
        if self.fitted_spline is None or self.input_sizes is None:
            raise ValueError("Must call fit_curve() first")

        x = np.array(self.input_sizes, dtype=float)
        y = self.fitted_spline(x)

        max_slope = 0.0

        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if x[j] != x[i]:  # Avoid division by zero
                    slope = abs(y[j] - y[i]) / abs(x[j] - x[i])
                    max_slope = max(max_slope, slope)

        return max_slope

    def compute_confidence_intervals(
        self, bootstrap_samples: int = 1000, confidence_level: float = 0.95
    ) -> list[list[float]]:
        """
        Compute bootstrap confidence intervals for first derivatives.

        Args:
            bootstrap_samples: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            List of (lower_bound, upper_bound) tuples
        """
        if self.input_sizes is None or self.runtime_data is None:
            raise ValueError("Must call fit_curve() first")

        n_points = len(self.input_sizes)
        bootstrap_derivatives = []

        # Generate bootstrap samples
        for _ in range(bootstrap_samples):
            # Resample with replacement
            indices = np.random.choice(n_points, size=n_points, replace=True)
            boot_x = [self.input_sizes[i] for i in indices]
            boot_y = [self.runtime_data[i] for i in indices]

            # Sort by x to maintain order
            sorted_pairs = sorted(zip(boot_x, boot_y, strict=False))
            boot_x_sorted = [x for x, y in sorted_pairs]
            boot_y_sorted = [y for x, y in sorted_pairs]

            try:
                # Fit curve to bootstrap sample
                boot_analyzer = CalculusAnalyzer()
                boot_analyzer.fit_curve(boot_x_sorted, boot_y_sorted)
                boot_derivs = boot_analyzer.compute_first_derivatives()
                bootstrap_derivatives.append(boot_derivs)
            except Exception:
                # Skip failed bootstrap samples
                continue

        if not bootstrap_derivatives:
            # Return empty intervals if all bootstrap samples failed
            return [(0.0, 0.0)] * n_points

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        intervals = []
        for i in range(n_points):
            values_at_i = [
                derivs[i] for derivs in bootstrap_derivatives if i < len(derivs)
            ]
            if values_at_i:
                lower = np.percentile(values_at_i, lower_percentile)
                upper = np.percentile(values_at_i, upper_percentile)
                intervals.append([float(lower), float(upper)])
            else:
                intervals.append([0.0, 0.0])

        return intervals

    def detect_monotonicity(self) -> bool:
        """
        Detect if the runtime function is monotonic.

        Returns:
            True if function is monotonic (non-decreasing)
        """
        if self.runtime_data is None:
            raise ValueError("Must call fit_curve() first")

        # Check if runtime generally increases
        increasing_count = 0
        total_pairs = 0

        for i in range(len(self.runtime_data) - 1):
            if self.runtime_data[i + 1] >= self.runtime_data[i]:
                increasing_count += 1
            total_pairs += 1

        # Consider monotonic if 80% of pairs are non-decreasing
        return increasing_count / total_pairs >= 0.8 if total_pairs > 0 else True

    def detect_convexity(self) -> bool:
        """
        Detect if the runtime function is convex.

        Returns:
            True if function is convex (second derivative mostly positive)
        """
        try:
            second_derivatives = self.compute_second_derivatives()

            # Count positive second derivatives
            positive_count = sum(1 for d in second_derivatives if d >= -0.1)

            # Consider convex if 70% of second derivatives are non-negative
            return positive_count / len(second_derivatives) >= 0.7
        except Exception:
            return False

    def classify_complexity(self) -> str:
        """
        Classify algorithmic complexity based on curve shape.

        Returns:
            String description of complexity class
        """
        if self.input_sizes is None or self.runtime_data is None:
            return "unknown"

        try:
            first_derivs = self.compute_first_derivatives()
            second_derivs = self.compute_second_derivatives()

            # Analyze derivative patterns
            avg_first_deriv = np.mean(np.abs(first_derivs))
            avg_second_deriv = np.mean(np.abs(second_derivs))

            # Classification heuristics
            if avg_first_deriv < 0.1:
                return "constant"
            elif avg_second_deriv < 0.1:
                return "linear"
            elif avg_second_deriv < 1.0:
                return "quadratic"
            else:
                return "polynomial"

        except Exception:
            return "unknown"

    def estimate_growth_rate(self) -> float:
        """
        Estimate the growth rate of the function.

        Returns:
            Growth rate factor
        """
        if self.runtime_data is None or len(self.runtime_data) < 2:
            return 1.0

        # Calculate average growth factor between consecutive points
        growth_factors = []
        for i in range(1, len(self.runtime_data)):
            if self.runtime_data[i - 1] > 0:
                factor = self.runtime_data[i] / self.runtime_data[i - 1]
                growth_factors.append(factor)

        return np.mean(growth_factors) if growth_factors else 1.0

    def assess_smoothness(self) -> float:
        """
        Assess the smoothness of the runtime curve.

        Returns:
            Smoothness score between 0 and 1 (1 = very smooth)
        """
        if self.runtime_data is None or len(self.runtime_data) < 3:
            return 1.0

        # Calculate variance of second differences (measure of jaggedness)
        second_diffs = []
        for i in range(1, len(self.runtime_data) - 1):
            second_diff = (
                self.runtime_data[i + 1]
                - 2 * self.runtime_data[i]
                + self.runtime_data[i - 1]
            )
            second_diffs.append(abs(second_diff))

        if not second_diffs:
            return 1.0

        # Normalize by data scale
        data_range = max(self.runtime_data) - min(self.runtime_data)
        if data_range == 0:
            return 1.0

        avg_jaggedness = np.mean(second_diffs) / data_range

        # Convert to smoothness score (higher jaggedness = lower smoothness)
        smoothness = max(0.0, 1.0 - min(1.0, avg_jaggedness * 10))

        return smoothness

    def detect_inflection_points(self) -> list[float]:
        """
        Detect inflection points where second derivative changes sign.

        Returns:
            List of input sizes where inflection points occur
        """
        try:
            second_derivs = self.compute_second_derivatives()
            x = np.array(self.input_sizes, dtype=float)

            inflection_points = []
            for i in range(1, len(second_derivs)):
                # Check for sign change in second derivative
                if (second_derivs[i - 1] > 0 and second_derivs[i] < 0) or (
                    second_derivs[i - 1] < 0 and second_derivs[i] > 0
                ):
                    # Interpolate the exact location
                    inflection_x = (x[i - 1] + x[i]) / 2
                    inflection_points.append(float(inflection_x))

            return inflection_points
        except Exception:
            return []

    def analyze_asymptotic_behavior(self) -> dict[str, Any]:
        """
        Analyze asymptotic behavior of the function.

        Returns:
            Dictionary with asymptotic analysis results
        """
        if self.input_sizes is None or self.runtime_data is None:
            return {"growth_type": "unknown"}

        try:
            # Look at the latter half of the data for asymptotic behavior
            half_point = len(self.runtime_data) // 2
            x_late = self.input_sizes[half_point:]
            y_late = self.runtime_data[half_point:]

            if len(x_late) < 2:
                return {"growth_type": "insufficient_data"}

            # Calculate growth pattern
            growth_ratios = []
            for i in range(1, len(y_late)):
                if y_late[i - 1] > 0:
                    growth_ratios.append(y_late[i] / y_late[i - 1])

            if not growth_ratios:
                return {"growth_type": "unknown"}

            avg_growth = np.mean(growth_ratios)

            # Classify growth type
            if avg_growth < 1.1:
                growth_type = "logarithmic"
            elif avg_growth < 1.5:
                growth_type = "linear"
            elif avg_growth < 2.5:
                growth_type = "polynomial"
            else:
                growth_type = "exponential"

            return {
                "growth_type": growth_type,
                "average_growth_ratio": avg_growth,
                "growth_stability": (
                    np.std(growth_ratios) if len(growth_ratios) > 1 else 0.0
                ),
            }

        except Exception:
            return {"growth_type": "unknown"}

    def analyze_stability(self) -> dict[str, Any]:
        """
        Analyze numerical stability of the curve fitting.

        Returns:
            Dictionary with stability metrics
        """
        if self.fitted_spline is None or self.input_sizes is None:
            return {
                "condition_number": float("inf"),
                "numerical_stability": False,
                "convergence_rate": 0.0,
            }

        try:
            x = np.array(self.input_sizes, dtype=float)

            # Estimate condition number based on input range
            condition_number = (
                (max(x) - min(x)) / min(x) if min(x) > 0 else float("inf")
            )

            # Check numerical stability
            numerical_stability = (
                self.fitting_quality_score > 0.8
                and condition_number < 1000
                and not np.isnan(self.fitting_quality_score)
            )

            # Estimate convergence rate based on fitting quality
            convergence_rate = max(0.0, self.fitting_quality_score)

            return {
                "condition_number": float(condition_number),
                "numerical_stability": numerical_stability,
                "convergence_rate": convergence_rate,
            }

        except Exception:
            return {
                "condition_number": float("inf"),
                "numerical_stability": False,
                "convergence_rate": 0.0,
            }

    def detect_periodicity(self) -> float | None:
        """
        Detect periodic patterns in runtime data.

        Returns:
            Period length if detected, None otherwise
        """
        if self.runtime_data is None or len(self.runtime_data) < 6:
            return None

        try:
            # Simple autocorrelation-based periodicity detection
            data = np.array(self.runtime_data)
            data_normalized = (data - np.mean(data)) / np.std(data)

            autocorr = np.correlate(data_normalized, data_normalized, mode="full")
            autocorr = autocorr[autocorr.size // 2 :]

            # Look for peaks in autocorrelation
            for lag in range(2, len(autocorr) // 2):
                if (
                    autocorr[lag] > 0.5
                    and autocorr[lag] > autocorr[lag - 1]
                    and autocorr[lag] > autocorr[lag + 1]
                ):
                    return float(lag)

            return None

        except Exception:
            return None

    def get_periodicity_confidence(self) -> float:
        """
        Get confidence in periodicity detection.

        Returns:
            Confidence score between 0 and 1
        """
        period = self.detect_periodicity()
        if period is None:
            return 0.0

        # Return confidence based on data quality and period detection
        return min(1.0, self.fitting_quality_score * 0.8)
