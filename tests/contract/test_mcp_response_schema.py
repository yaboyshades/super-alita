"""
Contract test for MCP response schema compliance.

These tests MUST FAIL initially as MCP integration does not exist yet.
Part of TDD Phase 3.2 - T006.
"""

import json
from pathlib import Path

import pytest
from jsonschema import validate


class TestMCPResponseSchemaContract:
    """Contract tests for calculus gate MCP response JSON schema."""

    @pytest.fixture
    def mcp_response_schema(self):
        """Load the MCP response JSON schema."""
        schema_path = (
            Path(__file__).parent.parent.parent
            / "specs"
            / "018-calculus-runtime-derivative-gate"
            / "contracts"
            / "mcp-response-schema.json"
        )
        with open(schema_path) as f:
            return json.load(f)

    def test_successful_mcp_response_validates_against_schema(
        self, mcp_response_schema
    ):
        """Test that successful MCP response passes schema validation."""
        # This test MUST FAIL - no MCP server exists yet
        from src.mcp.calculus_server import CalculusAnalysisServer

        # This will fail because CalculusAnalysisServer doesn't exist
        server = CalculusAnalysisServer()
        response = server.analyze_function("test_function")

        # Schema validation should pass
        validate(instance=response, schema=mcp_response_schema)

    def test_error_mcp_response_includes_error_details(self, mcp_response_schema):
        """Test that error responses include required error_details."""
        from src.mcp.calculus_server import CalculusAnalysisServer

        server = CalculusAnalysisServer()

        # Force an error condition
        response = server.analyze_function("nonexistent_function")

        assert response["status"] == "error"
        assert "error_details" in response

        # Schema validation should pass
        validate(instance=response, schema=mcp_response_schema)

    def test_mcp_response_has_required_fields(self, mcp_response_schema):
        """Test that MCP responses contain all required fields."""
        from src.mcp.calculus_server import CalculusAnalysisServer

        server = CalculusAnalysisServer()
        response = server.analyze_function("example_func")

        required_fields = mcp_response_schema["required"]
        for field in required_fields:
            assert field in response, f"Required field '{field}' missing"

    def test_mcp_response_status_enum_values(self, mcp_response_schema):
        """Test that status field uses only allowed enum values."""
        from src.mcp.calculus_server import CalculusAnalysisServer

        server = CalculusAnalysisServer()
        response = server.analyze_function("test_func")

        status = response["status"]
        assert status in ["success", "error", "warning"], f"Invalid status: {status}"

    def test_successful_response_includes_analysis_summary(self, mcp_response_schema):
        """Test that successful responses include analysis_summary."""
        from src.mcp.calculus_server import CalculusAnalysisServer

        server = CalculusAnalysisServer()
        response = server.analyze_function("test_func")

        if response["status"] in ["success", "warning"]:
            assert "analysis_summary" in response
            summary = response["analysis_summary"]

            assert "overall_grade" in summary
            assert summary["overall_grade"] in ["A", "B", "F"]

            assert "gates_passed" in summary
            assert 0 <= summary["gates_passed"] <= 3

            assert "gates_total" in summary
            assert summary["gates_total"] == 3

    def test_gate_results_structure(self, mcp_response_schema):
        """Test that gate_results follow correct structure."""
        from src.mcp.calculus_server import CalculusAnalysisServer

        server = CalculusAnalysisServer()
        response = server.analyze_function("test_func")

        if "gate_results" in response:
            gates = response["gate_results"]

            for gate_name in ["slope_gate", "curvature_gate", "lipschitz_gate"]:
                if gate_name in gates:
                    gate = gates[gate_name]

                    # Check required GateResult fields
                    assert "passed" in gate
                    assert isinstance(gate["passed"], bool)

                    assert "threshold" in gate
                    assert isinstance(gate["threshold"], (int, float))

                    assert "measured_value" in gate
                    assert isinstance(gate["measured_value"], (int, float))

    def test_performance_metrics_structure(self, mcp_response_schema):
        """Test that performance_metrics have correct structure."""
        from src.mcp.calculus_server import CalculusAnalysisServer

        server = CalculusAnalysisServer()
        response = server.analyze_function("test_func")

        if "performance_metrics" in response:
            metrics = response["performance_metrics"]

            if "sample_count" in metrics:
                assert metrics["sample_count"] >= 3

            if "input_size_range" in metrics:
                range_vals = metrics["input_size_range"]
                assert len(range_vals) == 2
                assert range_vals[0] <= range_vals[1]

            if "runtime_range" in metrics:
                range_vals = metrics["runtime_range"]
                assert len(range_vals) == 2
                assert range_vals[0] <= range_vals[1]

            if "lipschitz_constant" in metrics:
                assert metrics["lipschitz_constant"] >= 0

    def test_recommendations_structure(self, mcp_response_schema):
        """Test that recommendations follow correct structure."""
        from src.mcp.calculus_server import CalculusAnalysisServer

        server = CalculusAnalysisServer()
        response = server.analyze_function("test_func")

        if "recommendations" in response:
            recommendations = response["recommendations"]

            for rec in recommendations:
                assert "type" in rec
                assert rec["type"] in ["optimization", "investigation", "monitoring"]

                assert "priority" in rec
                assert rec["priority"] in ["low", "medium", "high", "critical"]

                assert "description" in rec
                assert isinstance(rec["description"], str)

    def test_trending_structure(self, mcp_response_schema):
        """Test that trending data follows correct structure."""
        from src.mcp.calculus_server import CalculusAnalysisServer

        server = CalculusAnalysisServer()
        response = server.analyze_function("test_func")

        if "trending" in response:
            trending = response["trending"]

            if "slope_trend" in trending:
                trend = trending["slope_trend"]
                assert trend in ["improving", "stable", "degrading", "unknown"]

            if "performance_change" in trending:
                assert isinstance(trending["performance_change"], (int, float))

            if "baseline_available" in trending:
                assert isinstance(trending["baseline_available"], bool)

    def test_conditional_schema_validation(self, mcp_response_schema):
        """Test that conditional schema rules are enforced."""
        from src.mcp.calculus_server import CalculusAnalysisServer

        server = CalculusAnalysisServer()

        # Test error response
        error_response = server.simulate_error_response()
        assert error_response["status"] == "error"
        assert "error_details" in error_response
        validate(instance=error_response, schema=mcp_response_schema)

        # Test success response
        success_response = server.analyze_function("test_func")
        if success_response["status"] in ["success", "warning"]:
            required_fields = [
                "analysis_summary",
                "gate_results",
                "performance_metrics",
            ]
            for field in required_fields:
                assert field in success_response
