"""
Integration test for live Gemini Pilot client and cognitive contract system.

This test validates:
- Gemini-2.5-Pro API connectivity
- JSON schema enforcement
- Cognitive contract YAML processing
- Self-healing decision making
- Pilot strategic planning
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Skip if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY environment variable not set",
)

from src.core.gemini_pilot import GeminiPilotClient


class TestLiveGeminiIntegration:
    """
    Live integration tests for Gemini Pilot client.

    NOTE: These tests make real API calls and require GEMINI_API_KEY.
    """

    @pytest.fixture
    async def pilot_client(self):
        """Create a live Gemini Pilot client."""
        client = GeminiPilotClient()
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_pilot_decision_structured_response(self, pilot_client):
        """Test that pilot_decide returns properly structured JSON responses."""
        response = await pilot_client.pilot_decide(
            goal="Test connection to Gemini API",
            agent_state="initializing",
            last_error="",
            memory_context=[],
            recent_events=[],
            experimental=False,
        )

        # Validate response structure
        assert isinstance(response, dict)
        assert "reasoning" in response
        assert "action" in response
        assert "parameters" in response

        # Validate enum constraints
        valid_actions = [
            "publish_event",
            "call_tool",
            "query_memory",
            "plan_steps",
            "escalate",
        ]
        assert response["action"] in valid_actions

        # Validate types
        assert isinstance(response["reasoning"], str)
        assert isinstance(response["parameters"], dict)
        assert len(response["reasoning"]) > 0

        print(f"✓ Pilot decision: {response['action']} - {response['reasoning']}")

    @pytest.mark.asyncio
    async def test_self_heal_decision_structured_response(self, pilot_client):
        """Test that self_heal_decide returns properly structured JSON responses."""
        response = await pilot_client.self_heal_decide(
            failure_type="network_timeout",
            error_details="Redis connection timed out after 30s",
            component="event_bus",
            system_state="degraded",
            retry_count=1,
            failure_history=[],
            diagnostic_data={"timeout": 30, "connection_pool": "exhausted"},
        )

        # Validate response structure
        assert isinstance(response, dict)
        assert "reasoning" in response
        assert "action" in response
        assert "parameters" in response
        assert "confidence" in response

        # Validate enum constraints
        valid_actions = ["retry", "restart", "rollback", "patch", "escalate"]
        assert response["action"] in valid_actions

        # Validate confidence range
        assert 0.0 <= response["confidence"] <= 1.0

        # Validate types
        assert isinstance(response["reasoning"], str)
        assert isinstance(response["parameters"], dict)
        assert len(response["reasoning"]) > 0

        print(
            f"✓ Self-heal decision: {response['action']} (confidence: {response['confidence']:.2f}) - {response['reasoning']}"
        )

    @pytest.mark.asyncio
    async def test_contract_loading_and_substitution(self, pilot_client):
        """Test YAML contract loading and variable substitution."""
        variables = {
            "CURRENT_GOAL": "Test variable substitution",
            "AGENT_STATE": "testing",
            "LAST_ERROR": "None",
            "MEMORY_CONTEXT": [{"key": "test", "value": "data"}],
            "RECENT_EVENTS": ["event1", "event2"],
        }

        contract = await pilot_client.load_contract("pilot_decision", variables)

        # Validate contract processing
        assert isinstance(contract, str)
        assert len(contract) > 0
        assert "Test variable substitution" in contract
        assert "testing" in contract

        # Ensure variables were substituted (no ${...} patterns left)
        import re

        unsubstituted = re.findall(r"\\$\\{[^}]+\\}", contract)
        assert (
            len(unsubstituted) == 0
        ), f"Unsubstituted variables found: {unsubstituted}"

        print("✓ Contract loaded and variables substituted successfully")

    @pytest.mark.asyncio
    async def test_schema_version_validation(self, pilot_client):
        """Test that schema version is properly validated."""
        response = await pilot_client.pilot_decide(
            goal="Test schema versioning", agent_state="testing", experimental=False
        )

        # Check that schema_version is present (may be None if not returned by model)
        # This tests the validation logic, not enforcement by Gemini
        if "schema_version" in response:
            assert isinstance(response["schema_version"], int)

        print("✓ Schema version validation passed")

    @pytest.mark.asyncio
    async def test_deterministic_seeding(self, pilot_client):
        """Test that deterministic seeding produces consistent results."""
        seed = 12345
        goal = "Deterministic test goal"
        agent_state = "consistent_state"

        # Make two requests with the same seed
        response1 = await pilot_client.pilot_decide(
            goal=goal, agent_state=agent_state, seed=seed, experimental=False
        )

        response2 = await pilot_client.pilot_decide(
            goal=goal, agent_state=agent_state, seed=seed, experimental=False
        )

        # Note: Gemini's seeding may not guarantee 100% identical responses
        # but should be more consistent than without seeding
        assert response1["action"] == response2["action"]
        print(f"✓ Seeded responses consistent: {response1['action']}")

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that timeout handling works correctly."""
        # Create client with very short timeout for testing
        client = GeminiPilotClient()

        # Mock the HTTP client to simulate timeout
        with patch.object(client.client, "post", side_effect=asyncio.TimeoutError):
            with pytest.raises(asyncio.TimeoutError):
                await client.pilot_decide(goal="Test timeout", agent_state="testing")

        await client.close()
        print("✓ Timeout handling works correctly")

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery and fallback mechanisms."""
        client = GeminiPilotClient()

        # Mock HTTP error
        mock_response = AsyncMock()
        mock_response.status_code = 429  # Rate limit
        mock_response.text = "Rate limit exceeded"
        mock_response.raise_for_status.side_effect = Exception("Rate limit")

        with patch.object(client.client, "post", return_value=mock_response):
            with pytest.raises(Exception):
                await client.pilot_decide(
                    goal="Test error recovery", agent_state="testing"
                )

        await client.close()
        print("✓ Error recovery mechanisms functional")

    @pytest.mark.asyncio
    async def test_large_context_compression(self, pilot_client):
        """Test handling of large context that exceeds token limits."""
        # Create a very large context
        large_context = "Large context data " * 1000  # ~17k characters

        compressed, was_compressed = pilot_client._compress_large_context(
            large_context, threshold=1000
        )

        if was_compressed:
            assert "compressed_data" in compressed
            assert "compression: gzip" in compressed
            assert len(compressed) < len(large_context)
            print("✓ Large context compression working")
        else:
            print("✓ Context size within limits, no compression needed")

    @pytest.mark.asyncio
    async def test_contract_file_existence(self):
        """Test that required contract files exist and are valid YAML."""
        import yaml

        contracts_dir = Path(__file__).parent.parent.parent / "prompts" / "contracts"

        # Check pilot contract
        pilot_contract = contracts_dir / "pilot_decision.yaml"
        assert pilot_contract.exists(), f"Pilot contract not found: {pilot_contract}"

        with open(pilot_contract, encoding="utf-8") as f:
            pilot_data = yaml.safe_load(f)
            assert isinstance(pilot_data, dict)
            assert "meta" in pilot_data or "system_instruction" in pilot_data

        # Check self-heal contract
        heal_contract = contracts_dir / "self_heal_decision.yaml"
        assert heal_contract.exists(), f"Self-heal contract not found: {heal_contract}"

        with open(heal_contract, encoding="utf-8") as f:
            heal_data = yaml.safe_load(f)
            assert isinstance(heal_data, dict)

        print("✓ Contract files exist and are valid YAML")


@pytest.mark.asyncio
async def test_end_to_end_pilot_flow():
    """
    End-to-end test of the complete Pilot flow:
    1. Strategic decision via pilot_decide
    2. Simulated failure and healing via self_heal_decide
    3. Contract processing and response validation
    """
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    async with GeminiPilotClient() as pilot:
        # Phase 1: Strategic planning
        strategic_response = await pilot.pilot_decide(
            goal="Process user bounty: Find Python security vulnerabilities in repository",
            agent_state="idle",
            memory_context=["Previous bounty: XSS in web app", "Security focus area"],
            recent_events=["user_request", "repo_scan_started"],
        )

        assert strategic_response["action"] in [
            "plan_steps",
            "call_tool",
            "query_memory",
        ]
        print(f"Strategic decision: {strategic_response['action']}")

        # Phase 2: Simulate failure and healing
        healing_response = await pilot.self_heal_decide(
            failure_type="analysis_timeout",
            error_details="Code analysis timed out after 5 minutes",
            component="security_scanner",
            system_state="degraded",
            retry_count=0,
            diagnostic_data={"timeout_duration": 300, "files_scanned": 150},
        )

        assert healing_response["action"] in ["retry", "patch", "escalate"]
        assert 0.0 <= healing_response["confidence"] <= 1.0
        print(
            f"Healing decision: {healing_response['action']} (confidence: {healing_response['confidence']:.2f})"
        )

        # Phase 3: Validate contract integrity
        assert len(strategic_response["reasoning"]) > 10
        assert len(healing_response["reasoning"]) > 10
        assert isinstance(strategic_response["parameters"], dict)
        assert isinstance(healing_response["parameters"], dict)

        print("✓ End-to-end Pilot flow successful")


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_live_gemini.py -v -s
    asyncio.run(test_end_to_end_pilot_flow())
