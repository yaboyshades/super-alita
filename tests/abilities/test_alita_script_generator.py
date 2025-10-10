"""
Unit tests for Alita Script Generator ability.

Tests cover:
- Script generation from MCP specifications
- GitHub search integration
- AST syntax validation
- Environment file creation
- Constitutional compliance scoring
- Error handling and edge cases
"""

import ast
from unittest.mock import MagicMock, patch

import pytest

from src.abilities.alita_script_generator import (
    AlitaScriptGeneratorAbility,
    GitHubSearchResult,
)
from src.core.event_bus import EventBus


@pytest.fixture
def sample_tool_spec():
    """Sample MCP tool specification from brainstorming."""
    return {
        "name": "youtube_subtitle_extractor",
        "purpose": "Extract subtitles from YouTube videos",
        "libraries": "youtube-transcript-api, requests",
        "complexity": "simple",
        "input_schema": {
            "video_url": "string",
            "language": "string (optional)",
        },
        "output_schema": {"subtitles": "list", "metadata": "dict"},
    }


@pytest.fixture
def mock_github_results():
    """Mock GitHub search results."""
    return [
        GitHubSearchResult(
            repository_url="https://github.com/jdepoix/youtube-transcript-api",
            repository_name="youtube-transcript-api",
            description="Python API for YouTube transcripts",
            stars=1500,
            readme_snippet="# YouTube Transcript API\nFetch YouTube transcripts",
            code_examples=[
                "from youtube_transcript_api import YouTubeTranscriptApi\n"
                "transcript = YouTubeTranscriptApi.get_transcript('video_id')"
            ],
            dependencies=["youtube-transcript-api", "requests"],
        )
    ]


@pytest.fixture
def script_generator():
    """Create script generator instance."""
    config = {
        "model": "claude-3-7-sonnet",
        "temperature": 0.7,
        "max_tokens": 4000,
        "max_github_results": 5,
        "constitutional_threshold": 0.75,
    }
    return AlitaScriptGeneratorAbility(config=config)


@pytest.fixture
async def initialized_generator(script_generator):
    """Create and initialize script generator."""
    event_bus = MagicMock(spec=EventBus)
    llm_client = None  # Use template generation
    ability_registry = MagicMock()

    await script_generator.initialize(
        event_bus=event_bus,
        llm_client=llm_client,
        ability_registry=ability_registry,
    )

    return script_generator


# ============================================================
# Initialization Tests
# ============================================================


def test_initialization_with_config():
    """Test initialization with custom configuration."""
    config = {
        "model": "gpt-4",
        "temperature": 0.8,
        "max_tokens": 3000,
        "max_github_results": 3,
        "constitutional_threshold": 0.80,
    }

    generator = AlitaScriptGeneratorAbility(config=config)

    assert generator.model == "gpt-4"
    assert generator.temperature == 0.8
    assert generator.max_tokens == 3000
    assert generator.max_github_results == 3
    assert generator.constitutional_threshold == 0.80
    assert not generator.initialized


def test_initialization_with_defaults():
    """Test initialization with default configuration."""
    generator = AlitaScriptGeneratorAbility()

    assert generator.model == "claude-3-7-sonnet"
    assert generator.temperature == 0.7
    assert generator.max_tokens == 4000
    assert generator.max_github_results == 5
    assert generator.constitutional_threshold == 0.75


@pytest.mark.asyncio
async def test_initialize_sets_dependencies():
    """Test that initialize() sets required dependencies."""
    generator = AlitaScriptGeneratorAbility()
    event_bus = MagicMock(spec=EventBus)
    llm_client = MagicMock()
    ability_registry = MagicMock()

    await generator.initialize(
        event_bus=event_bus,
        llm_client=llm_client,
        ability_registry=ability_registry,
    )

    assert generator.event_bus == event_bus
    assert generator.llm_client == llm_client
    assert generator.ability_registry == ability_registry
    assert generator.initialized


# ============================================================
# Script Generation Tests
# ============================================================


@pytest.mark.asyncio
async def test_generate_script_uninitialized():
    """Test that generate_script fails if not initialized."""
    generator = AlitaScriptGeneratorAbility()
    tool_spec = {"name": "test_tool", "purpose": "test"}

    result = await generator.generate_script(
        tool_specification=tool_spec, search_github=False
    )

    assert not result.success
    assert result.error_message == "Ability not initialized"


@pytest.mark.asyncio
async def test_generate_script_with_template(
    initialized_generator, sample_tool_spec
):
    """Test script generation using template (no LLM)."""
    result = await initialized_generator.generate_script(
        tool_specification=sample_tool_spec, search_github=False
    )

    assert result.success
    assert result.main_script is not None
    assert result.syntax_valid
    assert len(result.syntax_errors) == 0

    # Verify script content
    assert "import logging" in result.main_script
    assert "def execute_youtube_subtitle_extractor" in result.main_script
    assert 'if __name__ == "__main__"' in result.main_script

    # Verify metadata
    assert result.metadata["tool_name"] == "youtube_subtitle_extractor"
    assert result.metadata["model_used"] == "claude-3-7-sonnet"


@pytest.mark.asyncio
async def test_generate_script_with_github_search(
    initialized_generator, sample_tool_spec, mock_github_results
):
    """Test script generation with GitHub search enabled."""
    with patch.object(
        initialized_generator,
        "_search_github_examples",
        return_value=mock_github_results,
    ):
        result = await initialized_generator.generate_script(
            tool_specification=sample_tool_spec, search_github=True
        )

        assert result.success
        assert len(result.github_references) > 0
        assert result.metadata["github_search_enabled"]
        assert result.metadata["num_github_results"] == 1


@pytest.mark.asyncio
async def test_generate_script_creates_all_files(
    initialized_generator, sample_tool_spec
):
    """Test that all required files are generated."""
    result = await initialized_generator.generate_script(
        tool_specification=sample_tool_spec, search_github=False
    )

    assert result.success
    assert result.main_script is not None
    assert result.environment_yml is not None
    assert result.requirements_txt is not None
    assert result.cleanup_script is not None


@pytest.mark.asyncio
async def test_generate_script_constitutional_scoring(
    initialized_generator, sample_tool_spec
):
    """Test that constitutional score is calculated."""
    result = await initialized_generator.generate_script(
        tool_specification=sample_tool_spec, search_github=False
    )

    assert result.success
    assert 0.0 <= result.constitutional_score <= 1.0
    # Template scripts should score reasonably well
    assert result.constitutional_score >= 0.5


# ============================================================
# GitHub Search Tests
# ============================================================


@pytest.mark.asyncio
async def test_search_github_examples(initialized_generator, sample_tool_spec):
    """Test GitHub search functionality."""
    results = await initialized_generator._search_github_examples(
        sample_tool_spec
    )

    assert isinstance(results, list)
    assert len(results) > 0

    # Verify result structure
    for result in results:
        assert isinstance(result, GitHubSearchResult)
        assert result.repository_url
        assert result.repository_name
        assert result.description


@pytest.mark.asyncio
async def test_search_github_with_string_libraries(initialized_generator):
    """Test GitHub search with comma-separated library string."""
    tool_spec = {
        "name": "test",
        "purpose": "testing",
        "libraries": "requests, beautifulsoup4, lxml",
    }

    results = await initialized_generator._search_github_examples(tool_spec)

    assert len(results) > 0
    # Should respect max_github_results limit
    assert len(results) <= initialized_generator.max_github_results


# ============================================================
# Syntax Validation Tests
# ============================================================


def test_validate_script_syntax_valid(initialized_generator):
    """Test syntax validation with valid Python code."""
    valid_script = """
import logging

def hello_world():
    print("Hello, world!")

if __name__ == "__main__":
    hello_world()
"""

    is_valid, errors = initialized_generator._validate_script_syntax(
        valid_script
    )

    assert is_valid
    assert len(errors) == 0


def test_validate_script_syntax_invalid(initialized_generator):
    """Test syntax validation with invalid Python code."""
    invalid_script = """
def broken_function()
    print("Missing colon"
"""

    is_valid, errors = initialized_generator._validate_script_syntax(
        invalid_script
    )

    assert not is_valid
    assert len(errors) > 0
    assert "SyntaxError" in errors[0]


def test_validate_script_syntax_empty(initialized_generator):
    """Test syntax validation with empty script."""
    empty_script = ""

    is_valid, errors = initialized_generator._validate_script_syntax(
        empty_script
    )

    # Empty script is valid Python (no statements)
    assert is_valid
    assert len(errors) == 0


# ============================================================
# Environment File Creation Tests
# ============================================================


def test_create_conda_environment(
    initialized_generator, sample_tool_spec, mock_github_results
):
    """Test Conda environment.yml creation."""
    env_yml = initialized_generator._create_conda_environment(
        sample_tool_spec, mock_github_results
    )

    assert "name: youtube-subtitle-extractor" in env_yml
    assert "channels:" in env_yml
    assert "conda-forge" in env_yml
    assert "dependencies:" in env_yml
    assert "python=3.11" in env_yml
    assert "youtube-transcript-api" in env_yml
    assert "requests" in env_yml


def test_create_pip_requirements(
    initialized_generator, sample_tool_spec, mock_github_results
):
    """Test pip requirements.txt creation."""
    requirements = initialized_generator._create_pip_requirements(
        sample_tool_spec, mock_github_results
    )

    assert "youtube-transcript-api" in requirements
    assert "requests" in requirements
    assert "# Python dependencies" in requirements


def test_generate_cleanup_script(initialized_generator, sample_tool_spec):
    """Test cleanup script generation."""
    cleanup = initialized_generator._generate_cleanup_script(sample_tool_spec)

    assert "#!/usr/bin/env bash" in cleanup
    assert "youtube-subtitle-extractor" in cleanup
    assert "conda env remove" in cleanup
    assert "rm -rf" in cleanup


# ============================================================
# Code Quality Assessment Tests
# ============================================================


def test_assess_code_quality_comprehensive(initialized_generator):
    """Test code quality assessment with comprehensive script."""
    script = '''"""
Module docstring.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def process_data(data: str) -> Dict[str, Any]:
    """
    Process data with error handling.
    
    Args:
        data: Input data
        
    Returns:
        Processed result
    """
    try:
        logger.info("Processing data")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    result = process_data("test")
'''

    metrics = initialized_generator._assess_code_quality(script)

    assert metrics["has_docstrings"]
    assert metrics["has_type_hints"]
    assert metrics["has_error_handling"]
    assert metrics["has_logging"]
    assert metrics["has_main_guard"]
    assert metrics["has_imports"]
    assert metrics["function_count"] >= 1


def test_assess_code_quality_minimal(initialized_generator):
    """Test code quality assessment with minimal script."""
    script = "print('hello')"

    metrics = initialized_generator._assess_code_quality(script)

    assert not metrics["has_docstrings"]
    assert not metrics["has_type_hints"]
    assert not metrics["has_error_handling"]
    assert not metrics["has_logging"]
    assert not metrics["has_main_guard"]
    assert not metrics["has_imports"]


# ============================================================
# Constitutional Scoring Tests
# ============================================================


def test_constitutional_score_article_i(
    initialized_generator, sample_tool_spec, mock_github_results
):
    """Test Article I (Library-First) scoring."""
    script_with_imports = """
import youtube_transcript_api
import requests

def main():
    pass
"""

    score = initialized_generator._calculate_constitutional_score(
        script_with_imports, sample_tool_spec, mock_github_results
    )

    # Should score well on Article I (has imports + GitHub results)
    assert score > 0.0


def test_constitutional_score_article_ii(
    initialized_generator, sample_tool_spec
):
    """Test Article II (Test-First) scoring."""
    script_with_examples = '''
def process(data: str) -> dict:
    """
    Process data.
    
    Args:
        data: Input data
        
    Returns:
        Result dictionary
        
    Example:
        >>> result = process("test")
        >>> assert result["status"] == "success"
    """
    return {"status": "success"}
'''

    score = initialized_generator._calculate_constitutional_score(
        script_with_examples, sample_tool_spec, []
    )

    # Should score well on Article II (has docstring examples)
    assert score > 0.0


def test_constitutional_score_article_iii(
    initialized_generator, sample_tool_spec
):
    """Test Article III (Simplicity) scoring."""
    simple_script = """
import logging

def main():
    logging.info("Simple function")
    return True

if __name__ == "__main__":
    main()
"""

    score = initialized_generator._calculate_constitutional_score(
        simple_script, sample_tool_spec, []
    )

    # Simple script with 1 function should score well
    assert score > 0.0


def test_constitutional_score_comprehensive(
    initialized_generator, sample_tool_spec, mock_github_results
):
    """Test overall constitutional scoring."""
    comprehensive_script = '''"""
YouTube Subtitle Extractor.
"""

import logging
from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)


def extract_subtitles(video_id: str) -> dict:
    """
    Extract subtitles from YouTube video.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Subtitle data
        
    Example:
        >>> result = extract_subtitles("dQw4w9WgXcQ")
        >>> assert "subtitles" in result
    """
    try:
        logger.info(f"Extracting subtitles for {video_id}")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return {"subtitles": transcript, "status": "success"}
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    result = extract_subtitles("dQw4w9WgXcQ")
    print(result)
'''

    score = initialized_generator._calculate_constitutional_score(
        comprehensive_script, sample_tool_spec, mock_github_results
    )

    # Comprehensive script should score high
    assert score >= 0.70


# ============================================================
# Error Handling Tests
# ============================================================


@pytest.mark.asyncio
async def test_generate_script_handles_exception(initialized_generator):
    """Test that exceptions during generation are handled."""
    tool_spec = {"name": "test", "purpose": "test"}

    # Patch method to raise exception
    with patch.object(
        initialized_generator,
        "_generate_python_code",
        side_effect=Exception("Test error"),
    ):
        result = await initialized_generator.generate_script(
            tool_specification=tool_spec, search_github=False
        )

        assert not result.success
        assert result.error_message == "Test error"


@pytest.mark.asyncio
async def test_shutdown(initialized_generator):
    """Test shutdown cleanup."""
    await initialized_generator.shutdown()

    assert not initialized_generator.initialized


# ============================================================
# Template Generation Tests
# ============================================================


def test_generate_template_script_structure(
    initialized_generator, sample_tool_spec, mock_github_results
):
    """Test template script has correct structure."""
    script = initialized_generator._generate_template_script(
        sample_tool_spec, mock_github_results
    )

    # Should have all required components
    assert '"""' in script  # Docstring
    assert "import logging" in script
    assert "def execute_youtube_subtitle_extractor" in script
    assert "logger.info" in script
    assert "try:" in script
    assert "except Exception" in script
    assert 'if __name__ == "__main__"' in script

    # Should be valid Python
    try:
        ast.parse(script)
        valid = True
    except:
        valid = False

    assert valid


def test_extract_code_from_response_with_markdown(initialized_generator):
    """Test code extraction from markdown code blocks."""
    response = """
Here's the code:

```python
import logging

def main():
    print("Hello")
```

That's the implementation.
"""

    code = initialized_generator._extract_code_from_response(response)

    assert "import logging" in code
    assert "def main():" in code
    assert "Here's the code" not in code
    assert "```" not in code


def test_extract_code_from_response_with_imports(initialized_generator):
    """Test code extraction from response with imports."""
    response = """
Some explanation text.
import logging
import requests

def process():
    pass
"""

    code = initialized_generator._extract_code_from_response(response)

    assert code.startswith("import logging")
    assert "def process():" in code


def test_extract_code_from_response_plain(initialized_generator):
    """Test code extraction from plain code."""
    response = "print('hello')\nprint('world')"

    code = initialized_generator._extract_code_from_response(response)

    assert code == "print('hello')\nprint('world')"
