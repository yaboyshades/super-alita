"""
Certificate generation and performance assessment for calculus gate.

This module provides the core certificate generation logic that combines
sampling, analysis, and compliance assessment into performance certificates.
"""

from typing import Any

from .fitting import CalculusAnalyzer
from .models import AlertEvent, DerivativeCertificate, RuntimeSampleSet


class PerformanceCertificate:
    """Generates performance certificates with mathematical validation."""

    @classmethod
    def generate_for_function(
        cls,
        function_name: str,
        build_id: str = "unknown",
        slope_limit: float = 2.0,
        curvature_limit: float = 1.0,
        lipschitz_limit: float = 10.0,
        sample_set: RuntimeSampleSet | None = None,
    ) -> DerivativeCertificate:
        """
        Generate a performance certificate for a function.

        Args:
            function_name: Name of the function to analyze
            build_id: Build or commit identifier
            slope_limit: Maximum allowed first derivative
            curvature_limit: Maximum allowed second derivative
            lipschitz_limit: Maximum allowed Lipschitz constant
            sample_set: Pre-collected runtime samples (optional)

        Returns:
            Complete performance certificate
        """
        # Create basic certificate structure
        certificate = DerivativeCertificate(
            function_name=function_name,
            build_id=build_id,
            slope_limit=slope_limit,
            curvature_limit=curvature_limit,
            lipschitz_limit=lipschitz_limit,
            sample_set=sample_set,
        )

        # If no sample set provided, create minimal one for testing
        if sample_set is None:
            certificate.sample_set = cls._create_minimal_sample_set(
                function_name, build_id
            )

        # Perform analysis if we have data
        if certificate.sample_set and certificate.sample_set.sample_count >= 3:
            cls._perform_analysis(certificate)
        else:
            # No sufficient data - mark as failed
            certificate.certificate_grade = "F"
            certificate.overall_compliance = False

        return certificate

    @classmethod
    def generate(
        cls,
        sample_set: RuntimeSampleSet,
        first_derivatives: list[float],
        second_derivatives: list[float],
        lipschitz_constant: float,
        slope_limit: float = 2.0,
        curvature_limit: float = 1.0,
        lipschitz_limit: float = 10.0,
        bootstrap_iterations: int = 1000,
    ) -> DerivativeCertificate:
        """
        Generate certificate from pre-computed analysis results.

        Args:
            sample_set: Runtime measurement data
            first_derivatives: Computed first derivatives
            second_derivatives: Computed second derivatives
            lipschitz_constant: Computed Lipschitz constant
            slope_limit: Threshold for first derivatives
            curvature_limit: Threshold for second derivatives
            lipschitz_limit: Threshold for Lipschitz constant
            bootstrap_iterations: Number of bootstrap samples

        Returns:
            Complete performance certificate
        """
        certificate = DerivativeCertificate(
            function_name=sample_set.target_function,
            build_id=sample_set.build_id,
            sample_set=sample_set,
            first_derivatives=first_derivatives,
            second_derivatives=second_derivatives,
            lipschitz_constant=lipschitz_constant,
            slope_limit=slope_limit,
            curvature_limit=curvature_limit,
            lipschitz_limit=lipschitz_limit,
            bootstrap_iterations=bootstrap_iterations,
        )

        # Assess compliance
        cls._assess_compliance(certificate)

        # Generate grade
        cls._assign_grade(certificate)

        return certificate

    @classmethod
    def _create_minimal_sample_set(
        cls, function_name: str, build_id: str
    ) -> RuntimeSampleSet:
        """Create minimal sample set for testing purposes."""
        sample_set = RuntimeSampleSet(target_function=function_name, build_id=build_id)

        # Add minimal test data
        sample_set.add_sample(10, 0.01, 0.009, 1000, 500)
        sample_set.add_sample(20, 0.02, 0.018, 2000, 1000)
        sample_set.add_sample(50, 0.05, 0.045, 5000, 2500)

        return sample_set

    @classmethod
    def _perform_analysis(cls, certificate: DerivativeCertificate) -> None:
        """Perform complete mathematical analysis on the certificate."""
        if not certificate.sample_set or certificate.sample_set.sample_count < 3:
            return

        # Set up analyzer
        analyzer = CalculusAnalyzer()

        try:
            # Fit curve to wall time data
            analyzer.fit_curve(
                certificate.sample_set.input_sizes, certificate.sample_set.wall_times
            )

            # Store fitting metadata
            certificate.fitting_method = analyzer.fitting_method
            certificate.fitting_quality_score = analyzer.fitting_quality_score
            certificate.noise_handling_applied = analyzer.noise_handling_applied

            # Compute derivatives
            certificate.first_derivatives = analyzer.compute_first_derivatives()
            certificate.second_derivatives = analyzer.compute_second_derivatives()
            certificate.lipschitz_constant = analyzer.compute_lipschitz_constant()

            # Compute confidence intervals
            try:
                intervals = analyzer.compute_confidence_intervals(
                    bootstrap_samples=min(
                        certificate.bootstrap_iterations, 100
                    ),  # Limit for performance
                    confidence_level=0.95,
                )
                certificate.derivative_confidence_intervals = intervals
            except Exception:
                # Bootstrap failed - continue without confidence intervals
                pass

            # Assess compliance
            cls._assess_compliance(certificate)

            # Assign grade
            cls._assign_grade(certificate)

        except Exception as e:
            # Analysis failed
            print(f"Warning: Analysis failed for {certificate.function_name}: {e}")
            certificate.certificate_grade = "F"
            certificate.overall_compliance = False

    @classmethod
    def _assess_compliance(cls, certificate: DerivativeCertificate) -> None:
        """Assess compliance with threshold limits."""
        # Check slope gate (first derivatives)
        slope_violations = []
        max_slope = 0.0

        if certificate.first_derivatives:
            for i, derivative in enumerate(certificate.first_derivatives):
                abs_derivative = abs(derivative)
                max_slope = max(max_slope, abs_derivative)

                if abs_derivative > certificate.slope_limit:
                    if certificate.sample_set and i < len(
                        certificate.sample_set.input_sizes
                    ):
                        input_size = certificate.sample_set.input_sizes[i]
                        slope_violations.append((input_size, abs_derivative))

        certificate.slope_violations = slope_violations
        certificate.passes_slope_gate = len(slope_violations) == 0

        # Check curvature gate (second derivatives)
        curvature_violations = []
        max_curvature = 0.0

        if certificate.second_derivatives:
            for i, derivative in enumerate(certificate.second_derivatives):
                abs_derivative = abs(derivative)
                max_curvature = max(max_curvature, abs_derivative)

                if abs_derivative > certificate.curvature_limit:
                    if certificate.sample_set and i < len(
                        certificate.sample_set.input_sizes
                    ):
                        input_size = certificate.sample_set.input_sizes[i]
                        curvature_violations.append((input_size, abs_derivative))

        certificate.curvature_violations = curvature_violations
        certificate.passes_curvature_gate = len(curvature_violations) == 0

        # Check Lipschitz gate
        certificate.lipschitz_violation = bool(
            certificate.lipschitz_constant > certificate.lipschitz_limit
        )
        certificate.passes_lipschitz_gate = not certificate.lipschitz_violation

        # Overall compliance
        certificate.overall_compliance = (
            certificate.passes_slope_gate
            and certificate.passes_curvature_gate
            and certificate.passes_lipschitz_gate
        )

    @classmethod
    def _assign_grade(cls, certificate: DerivativeCertificate) -> None:
        """Assign performance grade based on compliance and violations."""
        if certificate.overall_compliance:
            certificate.certificate_grade = "A"
            return

        # Count severe violations (>2x threshold)
        severe_violations = 0

        for _, value in certificate.slope_violations:
            if value > certificate.slope_limit * 2:
                severe_violations += 1

        for _, value in certificate.curvature_violations:
            if value > certificate.curvature_limit * 2:
                severe_violations += 1

        if (
            certificate.lipschitz_violation
            and certificate.lipschitz_constant > certificate.lipschitz_limit * 2
        ):
            severe_violations += 1

        # Assign grade based on severity
        if severe_violations == 0:
            certificate.certificate_grade = "B"  # Minor violations only
        else:
            certificate.certificate_grade = "F"  # Severe violations

    @classmethod
    def generate_bootstrap_confidence_intervals(
        cls,
        sample_set: RuntimeSampleSet,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Generate bootstrap confidence intervals for derivatives.

        Args:
            sample_set: Runtime measurement data
            bootstrap_samples: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            Tuple of (first_derivative_intervals, second_derivative_intervals)
        """
        analyzer = CalculusAnalyzer()

        try:
            analyzer.fit_curve(sample_set.input_sizes, sample_set.wall_times)
            first_intervals = analyzer.compute_confidence_intervals(
                bootstrap_samples, confidence_level
            )

            # For second derivatives, we would need to modify the analyzer
            # For now, return empty second derivative intervals
            second_intervals = [(0.0, 0.0)] * len(first_intervals)

            return first_intervals, second_intervals

        except Exception:
            # Return empty intervals if bootstrap fails
            n_points = len(sample_set.input_sizes)
            empty_intervals = [(0.0, 0.0)] * n_points
            return empty_intervals, empty_intervals

    @classmethod
    def create_alert_events(
        cls, certificate: DerivativeCertificate
    ) -> list[AlertEvent]:
        """
        Create alert events for threshold violations.

        Args:
            certificate: Performance certificate with violations

        Returns:
            List of alert events
        """
        alerts = []

        # Slope violations
        for input_size, value in certificate.slope_violations:
            alert = AlertEvent.create_slope_violation(
                certificate.function_name,
                certificate.build_id,
                certificate.certificate_id,
                certificate.slope_limit,
                value,
                input_size,
            )
            alerts.append(alert)

        # Curvature violations
        for input_size, value in certificate.curvature_violations:
            alert = AlertEvent.create_curvature_violation(
                certificate.function_name,
                certificate.build_id,
                certificate.certificate_id,
                certificate.curvature_limit,
                value,
                input_size,
            )
            alerts.append(alert)

        # Lipschitz violation
        if certificate.lipschitz_violation:
            alert = AlertEvent.create_lipschitz_violation(
                certificate.function_name,
                certificate.build_id,
                certificate.certificate_id,
                certificate.lipschitz_limit,
                certificate.lipschitz_constant,
            )
            alerts.append(alert)

        return alerts

    @classmethod
    def compare_with_baseline(
        cls,
        current_certificate: DerivativeCertificate,
        baseline_certificate: DerivativeCertificate,
    ) -> dict[str, Any]:
        """
        Compare current certificate with baseline.

        Args:
            current_certificate: Current performance certificate
            baseline_certificate: Baseline certificate for comparison

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "baseline_id": baseline_certificate.certificate_id,
            "grade_change": f"{baseline_certificate.certificate_grade} → {current_certificate.certificate_grade}",
            "compliance_change": baseline_certificate.overall_compliance
            != current_certificate.overall_compliance,
            "performance_trends": {},
        }

        # Compare Lipschitz constants
        if baseline_certificate.lipschitz_constant > 0:
            lipschitz_change = (
                current_certificate.lipschitz_constant
                - baseline_certificate.lipschitz_constant
            ) / baseline_certificate.lipschitz_constant
            comparison["performance_trends"]["lipschitz_change_percent"] = (
                lipschitz_change * 100
            )

        # Compare violation counts
        current_violations = (
            len(current_certificate.slope_violations)
            + len(current_certificate.curvature_violations)
            + (1 if current_certificate.lipschitz_violation else 0)
        )

        baseline_violations = (
            len(baseline_certificate.slope_violations)
            + len(baseline_certificate.curvature_violations)
            + (1 if baseline_certificate.lipschitz_violation else 0)
        )

        comparison["violation_change"] = current_violations - baseline_violations

        # Determine trend
        if (
            current_certificate.certificate_grade == "A"
            and baseline_certificate.certificate_grade != "A"
        ):
            comparison["trend"] = "improving"
        elif (
            current_certificate.certificate_grade == "F"
            and baseline_certificate.certificate_grade != "F"
        ):
            comparison["trend"] = "degrading"
        elif current_violations < baseline_violations:
            comparison["trend"] = "improving"
        elif current_violations > baseline_violations:
            comparison["trend"] = "degrading"
        else:
            comparison["trend"] = "stable"

        return comparison

    @classmethod
    def update_certificate_with_baseline_comparison(
        cls,
        certificate: DerivativeCertificate,
        baseline_certificate: DerivativeCertificate,
    ) -> None:
        """
        Update certificate with baseline comparison data.

        Args:
            certificate: Certificate to update
            baseline_certificate: Baseline for comparison
        """
        comparison = cls.compare_with_baseline(certificate, baseline_certificate)
        certificate.baseline_comparison = baseline_certificate.certificate_id
        certificate.trend_analysis = comparison


class CertificateGrader:
    """Grades performance certificates based on compliance and severity."""

    @staticmethod
    def calculate_compliance_score(certificate: DerivativeCertificate) -> float:
        """
        Calculate numerical compliance score (0.0 to 1.0).

        Args:
            certificate: Certificate to score

        Returns:
            Compliance score between 0.0 and 1.0
        """
        total_gates = 3
        passed_gates = sum(
            [
                certificate.passes_slope_gate,
                certificate.passes_curvature_gate,
                certificate.passes_lipschitz_gate,
            ]
        )

        base_score = passed_gates / total_gates

        # Apply penalties for severe violations
        penalty = 0.0

        # Slope penalties
        for _, value in certificate.slope_violations:
            violation_ratio = value / certificate.slope_limit
            if violation_ratio > 3.0:
                penalty += 0.2
            elif violation_ratio > 2.0:
                penalty += 0.1
            else:
                penalty += 0.05

        # Curvature penalties
        for _, value in certificate.curvature_violations:
            violation_ratio = value / certificate.curvature_limit
            if violation_ratio > 3.0:
                penalty += 0.2
            elif violation_ratio > 2.0:
                penalty += 0.1
            else:
                penalty += 0.05

        # Lipschitz penalty
        if certificate.lipschitz_violation:
            violation_ratio = (
                certificate.lipschitz_constant / certificate.lipschitz_limit
            )
            if violation_ratio > 3.0:
                penalty += 0.2
            elif violation_ratio > 2.0:
                penalty += 0.1
            else:
                penalty += 0.05

        # Apply data quality bonus/penalty
        quality_factor = 1.0
        if certificate.fitting_quality_score > 0:
            quality_factor = 0.9 + 0.1 * certificate.fitting_quality_score

        final_score = max(0.0, min(1.0, (base_score - penalty) * quality_factor))
        return final_score

    @staticmethod
    def grade_from_score(score: float) -> str:
        """
        Convert compliance score to letter grade.

        Args:
            score: Compliance score (0.0 to 1.0)

        Returns:
            Letter grade ("A", "B", or "F")
        """
        if score >= 0.9:
            return "A"
        elif score >= 0.6:
            return "B"
        else:
            return "F"

    @classmethod
    def regrade_certificate(cls, certificate: DerivativeCertificate) -> None:
        """
        Recalculate and update certificate grade.

        Args:
            certificate: Certificate to regrade
        """
        score = cls.calculate_compliance_score(certificate)
        certificate.certificate_grade = cls.grade_from_score(score)
