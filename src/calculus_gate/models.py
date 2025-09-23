"""
Core data models for calculus runtime derivative gate.

These models represent the fundamental entities for runtime analysis,
derivative computation, and compliance assessment.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TargetFunction:
    """Configuration for a function to be monitored by calculus gate."""

    # Identity
    name: str  # Function name (e.g., "search_documents")
    file_path: str  # Absolute path to source file
    module_path: str  # Python import path (e.g., "src.core.search")

    # Sampling Configuration
    min_input_size: int = 1  # Minimum input size for testing
    max_input_size: int = 10000  # Maximum input size for testing
    sample_count: int = 20  # Number of sample points
    warmup_runs: int = 3  # Warmup iterations per sample

    # Threshold Configuration
    slope_limit: float = 2.0  # Max |df/dn| before violation
    curvature_limit: float = 1.0  # Max |d²f/dn²| before violation
    lipschitz_limit: float = 10.0  # Max Lipschitz constant

    # Input Generation Strategy
    input_generator: str = "default"  # Strategy name for generating test inputs
    input_config: dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    active: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.min_input_size <= 0:
            raise ValueError("min_input_size must be positive")
        if self.max_input_size <= self.min_input_size:
            raise ValueError("max_input_size must be greater than min_input_size")
        if self.sample_count < 3:
            raise ValueError("sample_count must be at least 3")
        if self.warmup_runs < 1:
            raise ValueError("warmup_runs must be at least 1")
        if any(
            limit <= 0
            for limit in [self.slope_limit, self.curvature_limit, self.lipschitz_limit]
        ):
            raise ValueError("All threshold limits must be positive")


@dataclass
class RuntimeSampleSet:
    """Set of runtime measurements for derivative analysis."""

    # Identity
    target_function: str  # Function name
    build_id: str  # Git commit hash or build identifier
    measurement_timestamp: datetime = field(default_factory=datetime.now)

    # Measurement Data
    input_sizes: list[int] = field(
        default_factory=list
    )  # Input sizes tested (strictly increasing)
    wall_times: list[float] = field(default_factory=list)  # Wall clock times in seconds
    cpu_times: list[float] = field(default_factory=list)  # CPU times in seconds
    memory_peaks: list[int] = field(default_factory=list)  # Peak memory usage in bytes
    memory_deltas: list[int] = field(
        default_factory=list
    )  # Memory allocated during execution

    # Sampling Metadata
    warmup_runs: int = 3  # Number of warmup runs per sample
    measurement_conditions: dict[str, Any] = field(
        default_factory=dict
    )  # System state during measurement

    # Quality Indicators
    measurement_noise: float = 0.0  # Coefficient of variation across runs
    convergence_achieved: bool = False  # Whether measurements stabilized
    outliers_removed: int = 0  # Number of outlier measurements dropped

    def __post_init__(self) -> None:
        """Validate measurement data consistency."""
        if not self.input_sizes:
            return  # Allow empty initialization

        # Check all arrays have same length
        lengths = [
            len(self.input_sizes),
            len(self.wall_times),
            len(self.cpu_times),
            len(self.memory_peaks),
            len(self.memory_deltas),
        ]
        if len(set(lengths)) > 1:
            raise ValueError("All measurement arrays must have the same length")

        # Check input sizes are strictly increasing
        if not all(
            self.input_sizes[i] < self.input_sizes[i + 1]
            for i in range(len(self.input_sizes) - 1)
        ):
            raise ValueError("Input sizes must be strictly increasing")

        # Check non-negative values
        if any(t < 0 for t in self.wall_times + self.cpu_times):
            raise ValueError("Time measurements must be non-negative")
        if any(m < 0 for m in self.memory_peaks):
            raise ValueError("Memory measurements must be non-negative")

    @property
    def sample_count(self) -> int:
        """Number of samples in this set."""
        return len(self.input_sizes)

    def add_sample(
        self,
        input_size: int,
        wall_time: float,
        cpu_time: float,
        memory_peak: int,
        memory_delta: int,
    ) -> None:
        """Add a new measurement sample."""
        if self.input_sizes and input_size <= self.input_sizes[-1]:
            raise ValueError("Input size must be larger than previous samples")

        self.input_sizes.append(input_size)
        self.wall_times.append(wall_time)
        self.cpu_times.append(cpu_time)
        self.memory_peaks.append(memory_peak)
        self.memory_deltas.append(memory_delta)


@dataclass
class DerivativeCertificate:
    """Mathematical certificate of runtime derivative analysis."""

    # Identity and Tracking
    function_name: str
    build_id: str
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    certificate_version: str = "1.0"

    # Analysis Results
    sample_set: RuntimeSampleSet | None = None
    fitted_curve_params: dict[str, Any] = field(
        default_factory=dict
    )  # Spline coefficients or fit parameters

    # Mathematical Derivatives
    first_derivatives: list[float] = field(
        default_factory=list
    )  # df/dn at each input size
    second_derivatives: list[float] = field(
        default_factory=list
    )  # d²f/dn² at each input size
    lipschitz_constant: float = 0.0  # Max |f(x1)-f(x2)|/|x1-x2|

    # Statistical Confidence
    derivative_confidence_intervals: list[tuple[float, float]] = field(
        default_factory=list
    )  # 95% CI for df/dn
    curvature_confidence_intervals: list[tuple[float, float]] = field(
        default_factory=list
    )  # 95% CI for d²f/dn²
    bootstrap_iterations: int = 1000  # Bootstrap sample count

    # Compliance Assessment
    slope_violations: list[tuple[int, float]] = field(
        default_factory=list
    )  # (input_size, df_dn_value)
    curvature_violations: list[tuple[int, float]] = field(
        default_factory=list
    )  # (input_size, d2f_dn2_value)
    lipschitz_violation: bool = False  # True if exceeds limit

    # Thresholds Applied
    slope_limit: float = 2.0
    curvature_limit: float = 1.0
    lipschitz_limit: float = 10.0

    # Quality Gates
    passes_slope_gate: bool = True
    passes_curvature_gate: bool = True
    passes_lipschitz_gate: bool = True
    overall_compliance: bool = True
    certificate_grade: str = "A"  # "A", "B", or "F"

    # Analysis Quality
    fitting_method: str = "cubic_spline"  # "cubic_spline", "savgol_fallback", etc.
    fitting_quality_score: float = 0.0  # R² or similar goodness-of-fit
    noise_handling_applied: bool = False  # Whether noise mitigation was used

    # Historical Context
    baseline_comparison: str | None = None  # Previous certificate ID for comparison
    trend_analysis: dict[str, Any] = field(
        default_factory=dict
    )  # Trend indicators vs baseline

    def __post_init__(self) -> None:
        """Validate certificate data consistency."""
        if self.certificate_grade not in ["A", "B", "F"]:
            raise ValueError("Certificate grade must be A, B, or F")

        # Validate threshold limits
        if any(
            limit <= 0
            for limit in [self.slope_limit, self.curvature_limit, self.lipschitz_limit]
        ):
            raise ValueError("All threshold limits must be positive")

        # Update overall compliance based on individual gates
        self.overall_compliance = (
            self.passes_slope_gate
            and self.passes_curvature_gate
            and self.passes_lipschitz_gate
        )

    @property
    def certificate_id(self) -> str:
        """Generate a unique certificate ID."""
        timestamp_str = self.analysis_timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{self.function_name}_{self.build_id[:8]}_{timestamp_str}"

    def to_dict(self) -> dict[str, Any]:
        """Convert certificate to dictionary for JSON serialization."""
        return {
            "function_name": self.function_name,
            "build_id": self.build_id,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "certificate_version": self.certificate_version,
            "sample_set": self._sample_set_to_dict() if self.sample_set else None,
            "fitted_curve_params": self.fitted_curve_params,
            "first_derivatives": self.first_derivatives,
            "second_derivatives": self.second_derivatives,
            "lipschitz_constant": self.lipschitz_constant,
            "derivative_confidence_intervals": self.derivative_confidence_intervals,
            "curvature_confidence_intervals": self.curvature_confidence_intervals,
            "bootstrap_iterations": self.bootstrap_iterations,
            "slope_violations": self.slope_violations,
            "curvature_violations": self.curvature_violations,
            "lipschitz_violation": self.lipschitz_violation,
            "slope_limit": self.slope_limit,
            "curvature_limit": self.curvature_limit,
            "lipschitz_limit": self.lipschitz_limit,
            "passes_slope_gate": self.passes_slope_gate,
            "passes_curvature_gate": self.passes_curvature_gate,
            "passes_lipschitz_gate": self.passes_lipschitz_gate,
            "overall_compliance": self.overall_compliance,
            "certificate_grade": self.certificate_grade,
            "fitting_method": self.fitting_method,
            "fitting_quality_score": self.fitting_quality_score,
            "noise_handling_applied": self.noise_handling_applied,
            "baseline_comparison": self.baseline_comparison,
            "trend_analysis": self.trend_analysis,
        }

    def _sample_set_to_dict(self) -> dict[str, Any]:
        """Convert sample set to dictionary."""
        if not self.sample_set:
            return {}

        return {
            "target_function": self.sample_set.target_function,
            "build_id": self.sample_set.build_id,
            "measurement_timestamp": self.sample_set.measurement_timestamp.isoformat(),
            "input_sizes": self.sample_set.input_sizes,
            "wall_times": self.sample_set.wall_times,
            "cpu_times": self.sample_set.cpu_times,
            "memory_peaks": self.sample_set.memory_peaks,
            "memory_deltas": self.sample_set.memory_deltas,
            "warmup_runs": self.sample_set.warmup_runs,
            "measurement_conditions": self.sample_set.measurement_conditions,
            "measurement_noise": self.sample_set.measurement_noise,
            "convergence_achieved": self.sample_set.convergence_achieved,
            "outliers_removed": self.sample_set.outliers_removed,
        }


@dataclass
class AlertEvent:
    """Alert event for CI/MCP consumption when violations occur."""

    # Event Identity
    event_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )  # UUID for tracking
    event_type: str = (
        ""  # "slope_violation", "curvature_violation", "lipschitz_violation"
    )
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "warning"  # "warning", "error", "critical"

    # Context
    function_name: str = ""
    build_id: str = ""
    certificate_id: str = ""  # Reference to full certificate

    # Violation Details
    threshold_name: str = ""  # "slope_limit", "curvature_limit", "lipschitz_limit"
    threshold_value: float = 0.0  # Configured limit that was exceeded
    actual_value: float = 0.0  # Measured value that exceeded limit
    violation_magnitude: float = 0.0  # How much the limit was exceeded (ratio)

    # Location Information
    input_size_at_violation: int | None = None  # Input size where violation occurred
    derivative_type: str = ""  # "first", "second", "lipschitz"

    # Actionable Information
    suggested_actions: list[str] = field(default_factory=list)  # Recommended next steps
    related_files: list[str] = field(
        default_factory=list
    )  # Files that might need review

    # Integration Fields
    ci_failure_recommended: bool = False  # Whether CI should fail

    def __post_init__(self) -> None:
        """Validate alert event data."""
        valid_event_types = [
            "slope_violation",
            "curvature_violation",
            "lipschitz_violation",
            "overall_failure",
            "analysis_error",
        ]
        if self.event_type and self.event_type not in valid_event_types:
            raise ValueError(f"Invalid event_type: {self.event_type}")

        valid_severities = ["warning", "error", "critical"]
        if self.severity not in valid_severities:
            raise ValueError(f"Invalid severity: {self.severity}")

        valid_derivative_types = ["first", "second", "lipschitz", ""]
        if self.derivative_type not in valid_derivative_types:
            raise ValueError(f"Invalid derivative_type: {self.derivative_type}")

    def to_dict(self) -> dict[str, Any]:
        """Convert alert event to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "function_name": self.function_name,
            "build_id": self.build_id,
            "certificate_id": self.certificate_id,
            "threshold_name": self.threshold_name,
            "threshold_value": self.threshold_value,
            "actual_value": self.actual_value,
            "violation_magnitude": self.violation_magnitude,
            "input_size_at_violation": self.input_size_at_violation,
            "derivative_type": self.derivative_type,
            "suggested_actions": self.suggested_actions,
            "related_files": self.related_files,
            "ci_failure_recommended": self.ci_failure_recommended,
        }

    @classmethod
    def create_slope_violation(
        cls,
        function_name: str,
        build_id: str,
        certificate_id: str,
        threshold_value: float,
        actual_value: float,
        input_size: int,
    ) -> "AlertEvent":
        """Create a slope violation alert event."""
        violation_magnitude = actual_value / threshold_value
        severity = (
            "critical"
            if violation_magnitude > 3.0
            else "error" if violation_magnitude > 2.0 else "warning"
        )

        return cls(
            event_type="slope_violation",
            severity=severity,
            function_name=function_name,
            build_id=build_id,
            certificate_id=certificate_id,
            threshold_name="slope_limit",
            threshold_value=threshold_value,
            actual_value=actual_value,
            violation_magnitude=violation_magnitude,
            input_size_at_violation=input_size,
            derivative_type="first",
            suggested_actions=[
                "Review algorithm complexity",
                "Consider performance optimizations",
                "Check for inefficient loops or recursive calls",
            ],
            ci_failure_recommended=violation_magnitude > 2.0,
        )

    @classmethod
    def create_curvature_violation(
        cls,
        function_name: str,
        build_id: str,
        certificate_id: str,
        threshold_value: float,
        actual_value: float,
        input_size: int,
    ) -> "AlertEvent":
        """Create a curvature violation alert event."""
        violation_magnitude = actual_value / threshold_value
        severity = (
            "critical"
            if violation_magnitude > 3.0
            else "error" if violation_magnitude > 2.0 else "warning"
        )

        return cls(
            event_type="curvature_violation",
            severity=severity,
            function_name=function_name,
            build_id=build_id,
            certificate_id=certificate_id,
            threshold_name="curvature_limit",
            threshold_value=threshold_value,
            actual_value=actual_value,
            violation_magnitude=violation_magnitude,
            input_size_at_violation=input_size,
            derivative_type="second",
            suggested_actions=[
                "Investigate acceleration in performance degradation",
                "Check for quadratic or exponential complexity patterns",
                "Review nested loops and data structure choices",
            ],
            ci_failure_recommended=violation_magnitude > 2.0,
        )

    @classmethod
    def create_lipschitz_violation(
        cls,
        function_name: str,
        build_id: str,
        certificate_id: str,
        threshold_value: float,
        actual_value: float,
    ) -> "AlertEvent":
        """Create a Lipschitz constant violation alert event."""
        violation_magnitude = actual_value / threshold_value
        severity = (
            "critical"
            if violation_magnitude > 3.0
            else "error" if violation_magnitude > 2.0 else "warning"
        )

        return cls(
            event_type="lipschitz_violation",
            severity=severity,
            function_name=function_name,
            build_id=build_id,
            certificate_id=certificate_id,
            threshold_name="lipschitz_limit",
            threshold_value=threshold_value,
            actual_value=actual_value,
            violation_magnitude=violation_magnitude,
            derivative_type="lipschitz",
            suggested_actions=[
                "Check for unstable performance characteristics",
                "Investigate sudden performance changes",
                "Review algorithm sensitivity to input size",
            ],
            ci_failure_recommended=violation_magnitude > 1.5,
        )
