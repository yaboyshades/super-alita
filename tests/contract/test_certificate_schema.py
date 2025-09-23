"""
Contract test for certificate JSON schema validation.

These tests MUST FAIL initially as implementation does not exist yet.
Part of TDD Phase 3.2 - T005.
"""

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate


class TestCertificateSchemaContract:
    """Contract tests for calculus gate certificate JSON schema."""

    @pytest.fixture
    def certificate_schema(self):
        """Load the certificate JSON schema."""
        schema_path = (
            Path(__file__).parent.parent.parent
            / "specs"
            / "018-calculus-runtime-derivative-gate"
            / "contracts"
            / "certificate-schema.json"
        )
        with open(schema_path) as f:
            return json.load(f)

    def test_valid_certificate_validates_against_schema(self, certificate_schema):
        """Test that a valid certificate passes schema validation."""
        # This test MUST FAIL - no certificate generation exists yet
        from src.calculus_gate.certificate import PerformanceCertificate

        # This will fail because PerformanceCertificate doesn't exist
        cert = PerformanceCertificate.generate_for_function("test_function")
        cert_dict = cert.to_dict()

        # Schema validation should pass
        validate(instance=cert_dict, schema=certificate_schema)

    def test_certificate_has_required_fields(self, certificate_schema):
        """Test that generated certificates contain all required fields."""
        from src.calculus_gate.certificate import PerformanceCertificate

        cert = PerformanceCertificate.generate_for_function("example_func")
        cert_dict = cert.to_dict()

        required_fields = certificate_schema["required"]
        for field in required_fields:
            assert (
                field in cert_dict
            ), f"Required field '{field}' missing from certificate"

    def test_certificate_derivative_arrays_minimum_length(self, certificate_schema):
        """Test that derivative arrays meet minimum length requirements."""
        from src.calculus_gate.certificate import PerformanceCertificate

        cert = PerformanceCertificate.generate_for_function("test_func")
        cert_dict = cert.to_dict()

        # First derivatives must have at least 3 items
        assert len(cert_dict["first_derivatives"]) >= 3

        # Second derivatives must have at least 3 items
        assert len(cert_dict["second_derivatives"]) >= 3

    def test_certificate_function_name_pattern(self, certificate_schema):
        """Test that function names follow valid Python identifier pattern."""
        from src.calculus_gate.certificate import PerformanceCertificate

        # Valid function names should work
        valid_names = ["test_function", "_private_func", "CamelCase", "func123"]
        for name in valid_names:
            cert = PerformanceCertificate.generate_for_function(name)
            cert_dict = cert.to_dict()
            validate(instance=cert_dict, schema=certificate_schema)

        # Invalid function names should fail schema validation
        invalid_names = ["123invalid", "with-dashes", "with spaces", ""]
        for name in invalid_names:
            with pytest.raises(ValidationError):
                cert = PerformanceCertificate.generate_for_function(name)
                cert_dict = cert.to_dict()
                validate(instance=cert_dict, schema=certificate_schema)

    def test_certificate_grade_enum_values(self, certificate_schema):
        """Test that certificate grades are limited to A, B, F."""
        from src.calculus_gate.certificate import PerformanceCertificate

        cert = PerformanceCertificate.generate_for_function("test_func")
        cert_dict = cert.to_dict()

        grade = cert_dict["certificate_grade"]
        assert grade in ["A", "B", "F"], f"Invalid grade: {grade}"

    def test_certificate_boolean_gates_present(self, certificate_schema):
        """Test that all gate pass/fail booleans are present."""
        from src.calculus_gate.certificate import PerformanceCertificate

        cert = PerformanceCertificate.generate_for_function("test_func")
        cert_dict = cert.to_dict()

        gate_fields = [
            "passes_slope_gate",
            "passes_curvature_gate",
            "passes_lipschitz_gate",
            "overall_compliance",
        ]

        for field in gate_fields:
            assert field in cert_dict
            assert isinstance(cert_dict[field], bool)

    def test_certificate_sample_set_structure(self, certificate_schema):
        """Test that embedded sample set follows schema structure."""
        from src.calculus_gate.certificate import PerformanceCertificate

        cert = PerformanceCertificate.generate_for_function("test_func")
        cert_dict = cert.to_dict()

        sample_set = cert_dict["sample_set"]

        # Check required sample_set fields
        required_sample_fields = [
            "target_function",
            "build_id",
            "measurement_timestamp",
            "input_sizes",
            "wall_times",
            "cpu_times",
            "memory_peaks",
            "memory_deltas",
        ]

        for field in required_sample_fields:
            assert field in sample_set, f"Sample set missing field: {field}"

        # Check array lengths match
        input_count = len(sample_set["input_sizes"])
        assert len(sample_set["wall_times"]) == input_count
        assert len(sample_set["cpu_times"]) == input_count
        assert len(sample_set["memory_peaks"]) == input_count
        assert len(sample_set["memory_deltas"]) == input_count

    def test_certificate_confidence_intervals_structure(self, certificate_schema):
        """Test confidence interval arrays have correct structure."""
        from src.calculus_gate.certificate import PerformanceCertificate

        cert = PerformanceCertificate.generate_for_function("test_func")
        cert_dict = cert.to_dict()

        if "derivative_confidence_intervals" in cert_dict:
            intervals = cert_dict["derivative_confidence_intervals"]

            for interval in intervals:
                assert (
                    len(interval) == 2
                ), "Each confidence interval must have exactly 2 values"
                assert interval[0] <= interval[1], "Lower bound must be <= upper bound"

    def test_certificate_violations_structure(self, certificate_schema):
        """Test that violation arrays have correct [input_size, value] structure."""
        from src.calculus_gate.certificate import PerformanceCertificate

        cert = PerformanceCertificate.generate_for_function("test_func")
        cert_dict = cert.to_dict()

        if "slope_violations" in cert_dict:
            for violation in cert_dict["slope_violations"]:
                assert (
                    len(violation) == 2
                ), "Each violation must have [input_size, value]"
                assert isinstance(violation[0], int), "Input size must be integer"
                assert isinstance(violation[1], (int, float)), "Value must be numeric"

        if "curvature_violations" in cert_dict:
            for violation in cert_dict["curvature_violations"]:
                assert (
                    len(violation) == 2
                ), "Each violation must have [input_size, value]"
                assert isinstance(violation[0], int), "Input size must be integer"
                assert isinstance(violation[1], (int, float)), "Value must be numeric"
