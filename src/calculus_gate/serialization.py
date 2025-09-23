"""
JSON serialization and deserialization for calculus gate data models.

Provides conversion between data models and JSON format for:
- Certificate storage and transmission
- MCP API responses
- CI artifact generation
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import AlertEvent, DerivativeCertificate, RuntimeSampleSet, TargetFunction


class CalculusGateEncoder(json.JSONEncoder):
    """Custom JSON encoder for calculus gate data models."""

    def default(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(
            obj, (TargetFunction, RuntimeSampleSet, DerivativeCertificate, AlertEvent)
        ):
            return obj.to_dict() if hasattr(obj, "to_dict") else obj.__dict__
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def serialize_certificate(certificate: DerivativeCertificate) -> str:
    """Serialize certificate to JSON string."""
    return json.dumps(certificate.to_dict(), cls=CalculusGateEncoder, indent=2)


def deserialize_certificate(json_str: str) -> DerivativeCertificate:
    """Deserialize certificate from JSON string."""
    data = json.loads(json_str)
    return certificate_from_dict(data)


def serialize_sample_set(sample_set: RuntimeSampleSet) -> str:
    """Serialize runtime sample set to JSON string."""
    return json.dumps(sample_set_to_dict(sample_set), cls=CalculusGateEncoder, indent=2)


def deserialize_sample_set(json_str: str) -> RuntimeSampleSet:
    """Deserialize runtime sample set from JSON string."""
    data = json.loads(json_str)
    return sample_set_from_dict(data)


def serialize_alert_event(alert: AlertEvent) -> str:
    """Serialize alert event to JSON string."""
    return json.dumps(alert.to_dict(), cls=CalculusGateEncoder, indent=2)


def deserialize_alert_event(json_str: str) -> AlertEvent:
    """Deserialize alert event from JSON string."""
    data = json.loads(json_str)
    return alert_event_from_dict(data)


def save_certificate_to_file(
    certificate: DerivativeCertificate, file_path: str | Path
) -> None:
    """Save certificate to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(certificate.to_dict(), f, cls=CalculusGateEncoder, indent=2)


def load_certificate_from_file(file_path: str | Path) -> DerivativeCertificate:
    """Load certificate from JSON file."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return certificate_from_dict(data)


def certificate_from_dict(data: dict[str, Any]) -> DerivativeCertificate:
    """Convert dictionary to DerivativeCertificate."""
    # Parse timestamps
    analysis_timestamp = datetime.fromisoformat(
        data.get("analysis_timestamp", datetime.now().isoformat())
    )

    # Parse sample set if present
    sample_set = None
    if data.get("sample_set"):
        sample_set = sample_set_from_dict(data["sample_set"])

    # Create certificate
    certificate = DerivativeCertificate(
        function_name=data["function_name"],
        build_id=data["build_id"],
        analysis_timestamp=analysis_timestamp,
        certificate_version=data.get("certificate_version", "1.0"),
        sample_set=sample_set,
        fitted_curve_params=data.get("fitted_curve_params", {}),
        first_derivatives=data.get("first_derivatives", []),
        second_derivatives=data.get("second_derivatives", []),
        lipschitz_constant=data.get("lipschitz_constant", 0.0),
        derivative_confidence_intervals=data.get("derivative_confidence_intervals", []),
        curvature_confidence_intervals=data.get("curvature_confidence_intervals", []),
        bootstrap_iterations=data.get("bootstrap_iterations", 1000),
        slope_violations=data.get("slope_violations", []),
        curvature_violations=data.get("curvature_violations", []),
        lipschitz_violation=data.get("lipschitz_violation", False),
        slope_limit=data.get("slope_limit", 2.0),
        curvature_limit=data.get("curvature_limit", 1.0),
        lipschitz_limit=data.get("lipschitz_limit", 10.0),
        passes_slope_gate=data.get("passes_slope_gate", True),
        passes_curvature_gate=data.get("passes_curvature_gate", True),
        passes_lipschitz_gate=data.get("passes_lipschitz_gate", True),
        overall_compliance=data.get("overall_compliance", True),
        certificate_grade=data.get("certificate_grade", "A"),
        fitting_method=data.get("fitting_method", "cubic_spline"),
        fitting_quality_score=data.get("fitting_quality_score", 0.0),
        noise_handling_applied=data.get("noise_handling_applied", False),
        baseline_comparison=data.get("baseline_comparison"),
        trend_analysis=data.get("trend_analysis", {}),
    )

    return certificate


def sample_set_from_dict(data: dict[str, Any]) -> RuntimeSampleSet:
    """Convert dictionary to RuntimeSampleSet."""
    # Parse timestamp
    measurement_timestamp = datetime.fromisoformat(
        data.get("measurement_timestamp", datetime.now().isoformat())
    )

    sample_set = RuntimeSampleSet(
        target_function=data["target_function"],
        build_id=data["build_id"],
        measurement_timestamp=measurement_timestamp,
        input_sizes=data.get("input_sizes", []),
        wall_times=data.get("wall_times", []),
        cpu_times=data.get("cpu_times", []),
        memory_peaks=data.get("memory_peaks", []),
        memory_deltas=data.get("memory_deltas", []),
        warmup_runs=data.get("warmup_runs", 3),
        measurement_conditions=data.get("measurement_conditions", {}),
        measurement_noise=data.get("measurement_noise", 0.0),
        convergence_achieved=data.get("convergence_achieved", False),
        outliers_removed=data.get("outliers_removed", 0),
    )

    return sample_set


def sample_set_to_dict(sample_set: RuntimeSampleSet) -> dict[str, Any]:
    """Convert RuntimeSampleSet to dictionary."""
    return {
        "target_function": sample_set.target_function,
        "build_id": sample_set.build_id,
        "measurement_timestamp": sample_set.measurement_timestamp.isoformat(),
        "input_sizes": sample_set.input_sizes,
        "wall_times": sample_set.wall_times,
        "cpu_times": sample_set.cpu_times,
        "memory_peaks": sample_set.memory_peaks,
        "memory_deltas": sample_set.memory_deltas,
        "warmup_runs": sample_set.warmup_runs,
        "measurement_conditions": sample_set.measurement_conditions,
        "measurement_noise": sample_set.measurement_noise,
        "convergence_achieved": sample_set.convergence_achieved,
        "outliers_removed": sample_set.outliers_removed,
    }


def alert_event_from_dict(data: dict[str, Any]) -> AlertEvent:
    """Convert dictionary to AlertEvent."""
    # Parse timestamp
    timestamp = datetime.fromisoformat(
        data.get("timestamp", datetime.now().isoformat())
    )

    alert = AlertEvent(
        event_id=data.get("event_id", ""),
        event_type=data.get("event_type", ""),
        timestamp=timestamp,
        severity=data.get("severity", "warning"),
        function_name=data.get("function_name", ""),
        build_id=data.get("build_id", ""),
        certificate_id=data.get("certificate_id", ""),
        threshold_name=data.get("threshold_name", ""),
        threshold_value=data.get("threshold_value", 0.0),
        actual_value=data.get("actual_value", 0.0),
        violation_magnitude=data.get("violation_magnitude", 0.0),
        input_size_at_violation=data.get("input_size_at_violation"),
        derivative_type=data.get("derivative_type", ""),
        suggested_actions=data.get("suggested_actions", []),
        related_files=data.get("related_files", []),
        ci_failure_recommended=data.get("ci_failure_recommended", False),
    )

    return alert


def target_function_from_dict(data: dict[str, Any]) -> TargetFunction:
    """Convert dictionary to TargetFunction."""
    # Parse timestamps
    created_at = datetime.fromisoformat(
        data.get("created_at", datetime.now().isoformat())
    )
    updated_at = datetime.fromisoformat(
        data.get("updated_at", datetime.now().isoformat())
    )

    target_func = TargetFunction(
        name=data["name"],
        file_path=data["file_path"],
        module_path=data["module_path"],
        min_input_size=data.get("min_input_size", 1),
        max_input_size=data.get("max_input_size", 10000),
        sample_count=data.get("sample_count", 20),
        warmup_runs=data.get("warmup_runs", 3),
        slope_limit=data.get("slope_limit", 2.0),
        curvature_limit=data.get("curvature_limit", 1.0),
        lipschitz_limit=data.get("lipschitz_limit", 10.0),
        input_generator=data.get("input_generator", "default"),
        input_config=data.get("input_config", {}),
        created_at=created_at,
        updated_at=updated_at,
        active=data.get("active", True),
    )

    return target_func


def target_function_to_dict(target_func: TargetFunction) -> dict[str, Any]:
    """Convert TargetFunction to dictionary."""
    return {
        "name": target_func.name,
        "file_path": target_func.file_path,
        "module_path": target_func.module_path,
        "min_input_size": target_func.min_input_size,
        "max_input_size": target_func.max_input_size,
        "sample_count": target_func.sample_count,
        "warmup_runs": target_func.warmup_runs,
        "slope_limit": target_func.slope_limit,
        "curvature_limit": target_func.curvature_limit,
        "lipschitz_limit": target_func.lipschitz_limit,
        "input_generator": target_func.input_generator,
        "input_config": target_func.input_config,
        "created_at": target_func.created_at.isoformat(),
        "updated_at": target_func.updated_at.isoformat(),
        "active": target_func.active,
    }


def create_mcp_response(
    certificate: DerivativeCertificate, status: str = "success"
) -> dict[str, Any]:
    """Create MCP-formatted response from certificate."""
    # Count passed gates
    gates_passed = sum(
        [
            certificate.passes_slope_gate,
            certificate.passes_curvature_gate,
            certificate.passes_lipschitz_gate,
        ]
    )

    # Count critical violations
    critical_violations = (
        len(
            [
                v
                for v in certificate.slope_violations
                if v[1] > certificate.slope_limit * 2
            ]
        )
        + len(
            [
                v
                for v in certificate.curvature_violations
                if v[1] > certificate.curvature_limit * 2
            ]
        )
        + (
            1
            if certificate.lipschitz_violation
            and certificate.lipschitz_constant > certificate.lipschitz_limit * 2
            else 0
        )
    )

    response = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "function_name": certificate.function_name,
        "build_id": certificate.build_id,
        "analysis_summary": {
            "overall_grade": certificate.certificate_grade,
            "gates_passed": gates_passed,
            "gates_total": 3,
            "critical_violations": critical_violations,
        },
        "gate_results": {
            "slope_gate": {
                "passed": certificate.passes_slope_gate,
                "threshold": certificate.slope_limit,
                "measured_value": (
                    max(certificate.first_derivatives)
                    if certificate.first_derivatives
                    else 0.0
                ),
            },
            "curvature_gate": {
                "passed": certificate.passes_curvature_gate,
                "threshold": certificate.curvature_limit,
                "measured_value": (
                    max(abs(d) for d in certificate.second_derivatives)
                    if certificate.second_derivatives
                    else 0.0
                ),
            },
            "lipschitz_gate": {
                "passed": certificate.passes_lipschitz_gate,
                "threshold": certificate.lipschitz_limit,
                "measured_value": certificate.lipschitz_constant,
            },
        },
        "performance_metrics": {
            "sample_count": (
                certificate.sample_set.sample_count if certificate.sample_set else 0
            ),
            "lipschitz_constant": certificate.lipschitz_constant,
        },
    }

    # Add input size and runtime range if sample set available
    if certificate.sample_set and certificate.sample_set.input_sizes:
        response["performance_metrics"]["input_size_range"] = [
            min(certificate.sample_set.input_sizes),
            max(certificate.sample_set.input_sizes),
        ]
        response["performance_metrics"]["runtime_range"] = [
            min(certificate.sample_set.wall_times),
            max(certificate.sample_set.wall_times),
        ]

    return response
