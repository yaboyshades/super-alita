"""Test the Constitutional SDD Framework integration."""

import asyncio
from pathlib import Path

import pytest

from src.sdd.constitutional_pipeline import ConstitutionalSDDPipeline
from src.sdd.models import SpecifyRequest


@pytest.mark.asyncio
async def test_constitutional_sdd_specify():
    """Test the /specify endpoint with constitutional validation."""
    # Create a test workspace
    test_workspace = Path("./test_workspace")
    test_workspace.mkdir(exist_ok=True)

    # Initialize the pipeline
    pipeline = ConstitutionalSDDPipeline(test_workspace)

    # Create a test request
    request = SpecifyRequest(
        user_input="Create a simple REST API for managing user accounts",
        constitutional_gates=True,
    )

    # Execute the specification
    response = await pipeline.specify(request)

    # Verify the response
    assert response.success is True
    assert response.specification is not None
    assert "Feature Specification" in response.specification
    assert response.feature_id is not None
    assert Path(response.feature_path).exists()
    assert response.overall_compliance_score >= 0.0
    assert response.overall_compliance_score <= 1.0

    # Check constitutional compliance
    assert isinstance(response.constitutional_compliance, dict)
    assert len(response.constitutional_compliance) > 0

    # Clean up
    import shutil

    if test_workspace.exists():
        shutil.rmtree(test_workspace)


def test_constitutional_scorer_integration():
    """Test that the constitutional scorer integrates correctly."""
    from src.constitutional import ConstitutionalScorer

    scorer = ConstitutionalScorer()

    # Test with a simple specification
    spec = """
    # Feature Specification

    Create a REST API for user management.

    ## Requirements
    - User registration
    - User authentication
    - Profile management
    """

    result = scorer.score_specification(spec)

    assert result.overall_score >= 0.0
    assert result.overall_score <= 1.0
    assert isinstance(result.violations, list)


def test_sdd_router_creation():
    """Test that the SDD router can be created."""
    from src.sdd.router import create_sdd_router

    router = create_sdd_router()
    assert router is not None
    assert router.prefix == "/sdd"
    assert "sdd" in router.tags


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_constitutional_sdd_specify())
    print("✅ Constitutional SDD specify test passed")

    # Run the scorer integration test
    test_constitutional_scorer_integration()
    print("✅ Constitutional scorer integration test passed")

    # Run the router test
    test_sdd_router_creation()
    print("✅ SDD router creation test passed")

    print("🎯 All Constitutional SDD integration tests passed!")
