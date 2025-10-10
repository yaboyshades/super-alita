"""
Unit tests for Alita MCP Brainstorming Ability.

Tests capability gap assessment, tool specification generation,
and constitutional validation.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.abilities.alita_mcp_brainstorming import (
    AlitaMCPBrainstormingAbility,
    BrainstormingResult,
    MCPToolSpecification,
)
from src.constitutional.scorer import ConstitutionalResult


@pytest.fixture
def brainstorming_ability():
    """Create a brainstorming ability instance for testing."""
    config = {
        "model": "claude-3-7-sonnet",
        "temperature": 0.7,
        "max_tokens": 2000,
        "constitutional_threshold": 0.75,
    }
    ability = AlitaMCPBrainstormingAbility(config)
    ability.llm_client = AsyncMock()
    ability.ability_registry = MagicMock()
    return ability


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for capability assessment."""
    return json.dumps(
        {
            "needs_new_mcp": True,
            "existing_abilities_sufficient": False,
            "assessment_confidence": 0.85,
            "capability_gaps": [
                "No ability to extract YouTube video subtitles"
            ],
            "tool_specifications": [
                {
                    "name": "youtube_subtitle_crawler",
                    "purpose": "Extract subtitles from YouTube videos",
                    "suggested_libraries": ["youtube-transcript-api"],
                    "input_schema": {
                        "type": "object",
                        "properties": {"video_id": {"type": "string"}},
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"transcript": {"type": "array"}},
                    },
                    "interface_description": "Call extract_transcript(video_id) to get subtitle data",
                    "estimated_complexity": "simple",
                    "constitutional_alignment": {
                        "article_i_compliance": 0.9,
                        "article_ii_compliance": 0.85,
                        "article_iii_compliance": 0.95,
                        "article_v_compliance": 0.88,
                    },
                }
            ],
            "recommendations": [
                "Use youtube-transcript-api (verified 100k+ downloads/month)",
                "Add error handling for unavailable transcripts",
            ],
        }
    )


@pytest.mark.asyncio
async def test_initialize(brainstorming_ability):
    """Test ability initialization."""
    result = await brainstorming_ability.initialize(
        None, llm_client=AsyncMock(), ability_registry=MagicMock()
    )
    assert result is True
    assert brainstorming_ability.llm_client is not None
    assert brainstorming_ability.ability_registry is not None


@pytest.mark.asyncio
async def test_assess_capability_gap_with_new_mcp(
    brainstorming_ability, mock_llm_response
):
    """Test capability gap assessment when new MCP is needed."""
    # Mock LLM client
    brainstorming_ability.llm_client.generate = AsyncMock(
        return_value={"text": mock_llm_response}
    )

    # Mock constitutional scorer
    with patch.object(
        brainstorming_ability.constitutional_scorer,
        "score_specification",
    ) as mock_scorer:
        mock_scorer.return_value = ConstitutionalResult(
            overall_score=0.87,
            article_scores={
                "Article I": 0.9,
                "Article II": 0.85,
                "Article III": 0.95,
            },
            violations=[],
            is_compliant=True,
        )

        task = "Extract subtitles from YouTube videos for analysis"
        result = await brainstorming_ability.assess_capability_gap(task)

        # Assertions
        assert result["needs_new_mcp"] is True
        assert result["existing_abilities_sufficient"] is False
        assert result["assessment_confidence"] == 0.85
        assert len(result["tool_specifications"]) == 1
        assert (
            result["tool_specifications"][0]["name"]
            == "youtube_subtitle_crawler"
        )
        assert result["constitutional_score"] == 0.87
        assert (
            result["metadata"]["constitutional_validation"]["is_compliant"]
            is True
        )


@pytest.mark.asyncio
async def test_assess_capability_gap_no_new_mcp_needed(
    brainstorming_ability,
):
    """Test when existing abilities are sufficient."""
    # Mock LLM response indicating no new MCP needed
    llm_response = json.dumps(
        {
            "needs_new_mcp": False,
            "existing_abilities_sufficient": True,
            "assessment_confidence": 0.95,
            "capability_gaps": [],
            "tool_specifications": [],
            "recommendations": [
                "Use existing web_agent ability for webpage access"
            ],
        }
    )

    brainstorming_ability.llm_client.generate = AsyncMock(
        return_value={"text": llm_response}
    )

    with patch.object(
        brainstorming_ability.constitutional_scorer,
        "score_specification",
    ) as mock_scorer:
        mock_scorer.return_value = ConstitutionalResult(
            overall_score=0.85,
            article_scores={},
            violations=[],
            is_compliant=True,
        )

        task = "Fetch content from a webpage"
        result = await brainstorming_ability.assess_capability_gap(task)

        assert result["needs_new_mcp"] is False
        assert result["existing_abilities_sufficient"] is True
        assert result["assessment_confidence"] == 0.95
        assert len(result["tool_specifications"]) == 0


@pytest.mark.asyncio
async def test_assess_capability_gap_llm_failure(
    brainstorming_ability,
):
    """Test graceful handling of LLM failures."""
    # Mock LLM client to raise exception
    brainstorming_ability.llm_client.generate = AsyncMock(
        side_effect=Exception("LLM service unavailable")
    )

    task = "Extract YouTube subtitles"
    result = await brainstorming_ability.assess_capability_gap(task)

    # Should return fallback result
    assert result["needs_new_mcp"] is False
    assert result["existing_abilities_sufficient"] is True
    assert result["assessment_confidence"] == 0.0
    assert "error" in result["metadata"]


def test_parse_brainstorming_response(
    brainstorming_ability, mock_llm_response
):
    """Test parsing of LLM response into BrainstormingResult."""
    result = brainstorming_ability._parse_brainstorming_response(
        mock_llm_response, "Extract YouTube subtitles"
    )

    assert isinstance(result, BrainstormingResult)
    assert result.needs_new_mcp is True
    assert len(result.tool_specifications) == 1
    assert isinstance(result.tool_specifications[0], MCPToolSpecification)
    assert result.tool_specifications[0].name == "youtube_subtitle_crawler"


def test_parse_brainstorming_response_with_markdown_fences(
    brainstorming_ability, mock_llm_response
):
    """Test parsing when LLM returns JSON wrapped in markdown fences."""
    wrapped_response = f"```json\n{mock_llm_response}\n```"

    result = brainstorming_ability._parse_brainstorming_response(
        wrapped_response, "Extract YouTube subtitles"
    )

    assert isinstance(result, BrainstormingResult)
    assert result.needs_new_mcp is True


def test_parse_brainstorming_response_invalid_json(
    brainstorming_ability,
):
    """Test graceful handling of invalid JSON responses."""
    invalid_response = "This is not valid JSON"

    result = brainstorming_ability._parse_brainstorming_response(
        invalid_response, "Some task"
    )

    # Should return fallback result
    assert isinstance(result, BrainstormingResult)
    assert result.needs_new_mcp is False
    assert result.assessment_confidence == 0.0
    assert len(result.recommendations) > 0


def test_get_current_capabilities(brainstorming_ability):
    """Test extraction of current capabilities from registry."""
    # Mock ability registry
    mock_tools = [
        {
            "tool_id": "web_agent",
            "description": "Fetch and parse webpages",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        },
        {
            "tool_id": "code_runner",
            "description": "Execute Python code in sandbox",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        },
    ]

    brainstorming_ability.ability_registry.list_tools = MagicMock(
        return_value=mock_tools
    )

    capabilities = brainstorming_ability._get_current_capabilities()

    assert len(capabilities) == 2
    assert "web_agent" in capabilities
    assert "code_runner" in capabilities
    assert (
        capabilities["web_agent"]["description"] == "Fetch and parse webpages"
    )


def test_build_brainstorming_prompt(brainstorming_ability):
    """Test prompt construction for LLM."""
    task = "Extract YouTube subtitles"
    capabilities = {
        "web_agent": {"description": "Fetch webpages", "input_schema": {}}
    }

    prompt = brainstorming_ability._build_brainstorming_prompt(
        task, capabilities
    )

    assert "Extract YouTube subtitles" in prompt
    assert "web_agent" in prompt
    assert "Fetch webpages" in prompt
    assert "capability gap" in prompt.lower()
    assert "constitutional" in prompt.lower()


def test_constitutional_validation(brainstorming_ability):
    """Test constitutional compliance validation of MCP specs."""
    result = BrainstormingResult(
        needs_new_mcp=True,
        tool_specifications=[
            MCPToolSpecification(
                name="test_tool",
                purpose="Test purpose",
                suggested_libraries=["pytest"],
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                interface_description="call test_tool(input)",
                estimated_complexity="simple",
                constitutional_alignment={
                    "article_i_compliance": 0.9,
                    "article_ii_compliance": 0.85,
                },
            )
        ],
        capability_gaps=["test gap"],
        existing_abilities_sufficient=False,
        assessment_confidence=0.8,
        constitutional_score=0.0,
        recommendations=[],
        metadata={},
    )

    with patch.object(
        brainstorming_ability.constitutional_scorer,
        "score_specification",
    ) as mock_scorer:
        mock_scorer.return_value = ConstitutionalResult(
            overall_score=0.82,
            article_scores={"Article I": 0.9, "Article II": 0.85},
            violations=[],
            is_compliant=True,
        )

        constitutional_result = (
            brainstorming_ability._validate_constitutional_compliance(result)
        )

        assert constitutional_result.overall_score == 0.82
        assert constitutional_result.is_compliant is True


@pytest.mark.asyncio
async def test_shutdown(brainstorming_ability):
    """Test ability shutdown."""
    await brainstorming_ability.shutdown()
    # Should complete without errors


@pytest.mark.asyncio
async def test_cleanup(brainstorming_ability):
    """Test ability cleanup."""
    await brainstorming_ability.cleanup()
    # Should complete without errors


@pytest.mark.asyncio
async def test_process_event(brainstorming_ability):
    """Test event processing."""
    result = await brainstorming_ability.process_event({"type": "test_event"})
    assert result is None  # Currently no event handling implemented


def test_to_dict(brainstorming_ability):
    """Test conversion of BrainstormingResult to dictionary."""
    result = BrainstormingResult(
        needs_new_mcp=True,
        tool_specifications=[
            MCPToolSpecification(
                name="test_tool",
                purpose="Test",
                suggested_libraries=["lib1"],
                input_schema={},
                output_schema={},
                interface_description="test interface",
                estimated_complexity="simple",
                constitutional_alignment={},
            )
        ],
        capability_gaps=["gap1"],
        existing_abilities_sufficient=False,
        assessment_confidence=0.8,
        constitutional_score=0.82,
        recommendations=["rec1"],
        metadata={"test": "data"},
    )

    result_dict = brainstorming_ability._to_dict(result)

    assert isinstance(result_dict, dict)
    assert result_dict["needs_new_mcp"] is True
    assert len(result_dict["tool_specifications"]) == 1
    assert result_dict["tool_specifications"][0]["name"] == "test_tool"
    assert result_dict["assessment_confidence"] == 0.8
    assert result_dict["constitutional_score"] == 0.82
