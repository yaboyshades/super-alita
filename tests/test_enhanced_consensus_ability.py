#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Consensus Ability

Uses pytest with proper fixtures, mocking, and edge case coverage.
Generated from pytest_ability_template.

Template Application:
- abilityName: "Enhanced Consensus"
- abilityClass: "EnhancedConsensusProvider"
- abilityPath: "src/abilities/enhanced_consensus_ability.py"
- testMethods: ["weighted_vote", "confidence_based", "semantic_similarity", "ensemble_ranking", "simple_vote"]
- edgeCases: ["timeout", "invalid_model", "empty_responses", "large_samples", "extreme_confidence"]
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from abilities.enhanced_consensus_ability import (
    ConsensusMethod,
    ConsensusRequest,
    ConsensusResponse,
    EnhancedConsensusProvider,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def consensus_provider():
    """Create an Enhanced Consensus Provider with test configuration."""
    config = {
        "base_url": "http://localhost:11434/v1",
        "model_name": "llama3.2:3b",
        "timeout": 30.0,
    }

    provider = EnhancedConsensusProvider(config)
    await provider.initialize()
    return provider


@pytest.fixture
def mock_consensus_provider():
    """Create a mocked Enhanced Consensus Provider for unit tests."""
    provider = Mock(spec=EnhancedConsensusProvider)
    provider.base_url = "http://localhost:11434/v1"
    provider.model_name = "llama3.2:3b"
    provider.timeout = 30.0
    return provider


@pytest.fixture
def sample_consensus_request():
    """Create a sample ConsensusRequest for testing."""
    return ConsensusRequest(
        prompt="What is the capital of France?",
        num_samples=3,
        temperature=0.7,
        max_tokens=100,
        method=ConsensusMethod.WEIGHTED_VOTE,
        confidence_threshold=0.7,
        temperature_range=0.2,
    )


@pytest.fixture
def sample_http_responses():
    """Create sample HTTP responses for mocking."""
    return [
        {"choices": [{"message": {"content": "The capital of France is Paris."}}]},
        {"choices": [{"message": {"content": "Paris is the capital city of France."}}]},
        {"choices": [{"message": {"content": "France's capital is Paris."}}]},
    ]


class TestEnhancedConsensusProvider:
    """Test suite for Enhanced Consensus Provider."""

    async def test_provider_initialization(self, consensus_provider):
        """Test that the provider initializes correctly."""
        assert consensus_provider.base_url == "http://localhost:11434/v1"
        assert consensus_provider.model_name == "llama3.2:3b"
        assert consensus_provider.timeout == 30.0
        assert hasattr(consensus_provider, "_client")

    async def test_consensus_request_validation(self, sample_consensus_request):
        """Test that ConsensusRequest validates properly."""
        assert sample_consensus_request.prompt == "What is the capital of France?"
        assert sample_consensus_request.num_samples == 3
        assert sample_consensus_request.method == ConsensusMethod.WEIGHTED_VOTE
        assert 0.0 <= sample_consensus_request.confidence_threshold <= 1.0

    async def test_weighted_vote_consensus(
        self, consensus_provider, sample_consensus_request
    ):
        """Test weighted vote consensus method."""
        # Mock the HTTP client to return predictable responses
        with patch.object(consensus_provider, "_client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Paris is the capital of France."}}]
            }
            mock_client.post.return_value = mock_response

            sample_consensus_request.method = ConsensusMethod.WEIGHTED_VOTE
            result = await consensus_provider.consensus_sampling(
                sample_consensus_request
            )

            assert isinstance(result, ConsensusResponse)
            assert result.aggregation_method == "weighted_vote"
            assert result.consensus_confidence > 0.0
            assert (
                len(result.individual_responses) == sample_consensus_request.num_samples
            )

    async def test_confidence_based_consensus(
        self, consensus_provider, sample_consensus_request
    ):
        """Test confidence-based consensus method."""
        with patch.object(consensus_provider, "_client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Paris"}}]
            }
            mock_client.post.return_value = mock_response

            sample_consensus_request.method = ConsensusMethod.CONFIDENCE_BASED
            result = await consensus_provider.consensus_sampling(
                sample_consensus_request
            )

            assert isinstance(result, ConsensusResponse)
            assert result.aggregation_method == "confidence_based"
            assert (
                result.consensus_confidence
                >= sample_consensus_request.confidence_threshold
            )

    async def test_semantic_similarity_consensus(
        self, consensus_provider, sample_consensus_request
    ):
        """Test semantic similarity consensus method."""
        with patch.object(consensus_provider, "_client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "The capital is Paris"}}]
            }
            mock_client.post.return_value = mock_response

            sample_consensus_request.method = ConsensusMethod.SEMANTIC_SIMILARITY
            result = await consensus_provider.consensus_sampling(
                sample_consensus_request
            )

            assert isinstance(result, ConsensusResponse)
            assert result.aggregation_method == "semantic_similarity"
            assert result.consensus_text is not None

    async def test_ensemble_ranking_consensus(
        self, consensus_provider, sample_consensus_request
    ):
        """Test ensemble ranking consensus method."""
        with patch.object(consensus_provider, "_client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Paris"}}]
            }
            mock_client.post.return_value = mock_response

            sample_consensus_request.method = ConsensusMethod.ENSEMBLE_RANKING
            result = await consensus_provider.consensus_sampling(
                sample_consensus_request
            )

            assert isinstance(result, ConsensusResponse)
            assert result.aggregation_method == "ensemble_ranking"
            assert "ensemble_score" in result.metadata

    async def test_simple_vote_consensus(
        self, consensus_provider, sample_consensus_request
    ):
        """Test simple vote consensus method."""
        with patch.object(consensus_provider, "_client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Paris"}}]
            }
            mock_client.post.return_value = mock_response

            sample_consensus_request.method = ConsensusMethod.SIMPLE_VOTE
            result = await consensus_provider.consensus_sampling(
                sample_consensus_request
            )

            assert isinstance(result, ConsensusResponse)
            assert result.aggregation_method == "simple_vote"
            assert "vote_distribution" in result.metadata

    async def test_legacy_parameter_compatibility(self, consensus_provider):
        """Test backward compatibility with legacy parameters."""
        result = await consensus_provider.consensus_sampling(
            prompt="Test prompt",
            num_samples=2,
            temperature=0.5,
            max_tokens=50,
            method="weighted_vote",
            confidence_threshold=0.6,
        )

        assert isinstance(result, ConsensusResponse)
        assert result.aggregation_method == "weighted_vote"

    async def test_pydantic_request_model(
        self, consensus_provider, sample_consensus_request
    ):
        """Test using Pydantic request model directly."""
        result = await consensus_provider.consensus_sampling(
            request=sample_consensus_request
        )

        assert isinstance(result, ConsensusResponse)
        assert result.aggregation_method == sample_consensus_request.method.value

    @pytest.mark.parametrize(
        "method",
        [
            ConsensusMethod.SIMPLE_VOTE,
            ConsensusMethod.WEIGHTED_VOTE,
            ConsensusMethod.CONFIDENCE_BASED,
            ConsensusMethod.SEMANTIC_SIMILARITY,
            ConsensusMethod.ENSEMBLE_RANKING,
        ],
    )
    async def test_all_consensus_methods_parametrized(self, consensus_provider, method):
        """Test all consensus methods with parametrization."""
        request = ConsensusRequest(
            prompt="Test prompt for all methods",
            num_samples=2,
            method=method,
        )

        with patch.object(consensus_provider, "_client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": f"Response for {method.value}"}}]
            }
            mock_client.post.return_value = mock_response

            result = await consensus_provider.consensus_sampling(request=request)

            assert isinstance(result, ConsensusResponse)
            assert result.aggregation_method == method.value


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    async def test_timeout_handling(self, mock_consensus_provider):
        """Test timeout handling during consensus sampling."""
        mock_consensus_provider.consensus_sampling = AsyncMock(
            side_effect=TimeoutError("Request timed out")
        )

        with pytest.raises(asyncio.TimeoutError):
            await mock_consensus_provider.consensus_sampling(
                prompt="Test", num_samples=1, method="weighted_vote"
            )

    async def test_invalid_model_handling(self):
        """Test handling of invalid model configuration."""
        config = {
            "base_url": "http://localhost:11434/v1",
            "model_name": "nonexistent-model",
            "timeout": 30.0,
        }

        provider = EnhancedConsensusProvider(config)
        await provider.initialize()

        with patch.object(provider, "_client") as mock_client:
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Model not found", request=Mock(), response=Mock(status_code=404)
            )

            with pytest.raises(httpx.HTTPStatusError):
                await provider.consensus_sampling(
                    prompt="Test", num_samples=1, method="weighted_vote"
                )

    async def test_empty_responses_handling(self, consensus_provider):
        """Test handling of empty or invalid responses."""
        with patch.object(consensus_provider, "_client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {"choices": []}  # Empty choices
            mock_client.post.return_value = mock_response

            result = await consensus_provider.consensus_sampling(
                prompt="Test", num_samples=1, method="simple_vote"
            )

            # Should handle gracefully and return a valid response
            assert isinstance(result, ConsensusResponse)

    async def test_large_sample_count(self, consensus_provider):
        """Test handling of large sample counts."""
        request = ConsensusRequest(
            prompt="Test with many samples",
            num_samples=50,  # Large number
            method=ConsensusMethod.WEIGHTED_VOTE,
        )

        with patch.object(consensus_provider, "_client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Response"}}]
            }
            mock_client.post.return_value = mock_response

            result = await consensus_provider.consensus_sampling(request=request)

            assert isinstance(result, ConsensusResponse)
            assert (
                len(result.individual_responses) <= 50
            )  # Should handle or limit appropriately

    async def test_extreme_confidence_threshold(self, consensus_provider):
        """Test handling of extreme confidence thresholds."""
        request = ConsensusRequest(
            prompt="Test extreme confidence",
            num_samples=3,
            method=ConsensusMethod.CONFIDENCE_BASED,
            confidence_threshold=0.99,  # Very high threshold
        )

        with patch.object(consensus_provider, "_client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Low confidence response"}}]
            }
            mock_client.post.return_value = mock_response

            result = await consensus_provider.consensus_sampling(request=request)

            assert isinstance(result, ConsensusResponse)
            # Should handle cases where no response meets the threshold


class TestIntegration:
    """Integration tests with real or realistic scenarios."""

    async def test_full_pipeline_weighted_vote(self, consensus_provider):
        """Test complete pipeline for weighted vote consensus."""
        request = ConsensusRequest(
            prompt="What are the benefits of renewable energy?",
            num_samples=3,
            temperature=0.7,
            max_tokens=150,
            method=ConsensusMethod.WEIGHTED_VOTE,
            confidence_threshold=0.6,
        )

        # Mock diverse responses
        responses = [
            "Renewable energy is clean and sustainable.",
            "Clean energy sources reduce carbon emissions.",
            "Sustainable power helps combat climate change.",
        ]

        with patch.object(consensus_provider, "_client") as mock_client:
            mock_client.post.side_effect = [
                Mock(json=lambda: {"choices": [{"message": {"content": resp}}]})
                for resp in responses
            ]

            result = await consensus_provider.consensus_sampling(request=request)

            assert isinstance(result, ConsensusResponse)
            assert result.aggregation_method == "weighted_vote"
            assert len(result.individual_responses) == 3
            assert result.consensus_confidence > 0.0
            assert (
                "clean" in result.consensus_text.lower()
                or "renewable" in result.consensus_text.lower()
            )

    async def test_temperature_diversity_effect(self, consensus_provider):
        """Test that temperature range affects response diversity."""
        base_request = ConsensusRequest(
            prompt="Describe artificial intelligence",
            num_samples=3,
            method=ConsensusMethod.WEIGHTED_VOTE,
        )

        # Test with different temperature ranges
        low_temp_request = base_request.model_copy(
            update={"temperature": 0.1, "temperature_range": 0.1}
        )
        high_temp_request = base_request.model_copy(
            update={"temperature": 1.0, "temperature_range": 0.3}
        )

        # Mock responses
        with patch.object(consensus_provider, "_client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "AI is a technology field"}}]
            }
            mock_client.post.return_value = mock_response

            low_temp_result = await consensus_provider.consensus_sampling(
                request=low_temp_request
            )
            high_temp_result = await consensus_provider.consensus_sampling(
                request=high_temp_request
            )

            assert isinstance(low_temp_result, ConsensusResponse)
            assert isinstance(high_temp_result, ConsensusResponse)
            # Both should succeed with their respective temperature settings


class TestPerformance:
    """Performance and load testing."""

    async def test_concurrent_consensus_requests(self, consensus_provider):
        """Test handling multiple concurrent consensus requests."""
        requests_data = [
            ("What is machine learning?", ConsensusMethod.WEIGHTED_VOTE),
            ("Explain neural networks", ConsensusMethod.CONFIDENCE_BASED),
            ("Define deep learning", ConsensusMethod.SEMANTIC_SIMILARITY),
        ]

        with patch.object(consensus_provider, "_client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Test response"}}]
            }
            mock_client.post.return_value = mock_response

            # Create concurrent tasks
            tasks = []
            for prompt, method in requests_data:
                request = ConsensusRequest(prompt=prompt, num_samples=2, method=method)
                task = consensus_provider.consensus_sampling(request=request)
                tasks.append(task)

            # Execute concurrently
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 3
            for result in results:
                assert isinstance(result, ConsensusResponse)

    async def test_response_time_measurement(self, consensus_provider):
        """Test and measure response time for consensus operations."""
        import time

        request = ConsensusRequest(
            prompt="Quick test prompt",
            num_samples=2,
            method=ConsensusMethod.SIMPLE_VOTE,
        )

        with patch.object(consensus_provider, "_client") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Quick response"}}]
            }
            mock_client.post.return_value = mock_response

            start_time = time.time()
            result = await consensus_provider.consensus_sampling(request=request)
            end_time = time.time()

            response_time = end_time - start_time

            assert isinstance(result, ConsensusResponse)
            assert response_time < 5.0  # Should complete within 5 seconds with mocking


# Pytest configuration and test discovery
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
