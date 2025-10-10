"""Tests for MCP Box storage system."""

import json
import os
import shutil
import tempfile

import pytest

from src.mcp.mcp_box import MCPBox


@pytest.fixture
def temp_mcp_box():
    """Create temporary MCP Box for testing."""
    temp_dir = tempfile.mkdtemp()
    box = MCPBox(storage_dir=temp_dir)
    yield box
    shutil.rmtree(temp_dir)


def test_store_and_retrieve_mcp(temp_mcp_box):
    """Test storing and retrieving an MCP."""
    mcp_id = "test_tool"
    spec = {
        "name": "Test Tool",
        "purpose": "Testing",
        "libraries": "pytest",
    }
    script = "#!/usr/bin/env python\nprint('test')"

    # Store MCP
    success = temp_mcp_box.store_mcp(
        mcp_id=mcp_id,
        name="Test Tool",
        purpose="Testing MCP Box",
        spec=spec,
        main_script=script,
        environment_yml="name: test",
        requirements_txt="pytest",
        constitutional_score=0.85,
        tags=["test", "demo"],
    )

    assert success is True

    # Retrieve MCP
    mcp = temp_mcp_box.get_mcp(mcp_id)
    assert mcp is not None
    assert mcp["metadata"]["name"] == "Test Tool"
    assert mcp["metadata"]["purpose"] == "Testing MCP Box"
    assert mcp["metadata"]["constitutional_score"] == 0.85
    assert "test" in mcp["metadata"]["tags"]
    assert mcp["files"]["spec.json"] == spec
    assert "print('test')" in mcp["files"]["script.py"]


def test_list_mcps(temp_mcp_box):
    """Test listing stored MCPs."""
    # Store multiple MCPs
    for i in range(3):
        temp_mcp_box.store_mcp(
            mcp_id=f"tool_{i}",
            name=f"Tool {i}",
            purpose=f"Test tool {i}",
            spec={},
            main_script="print('test')",
            constitutional_score=0.7 + (i * 0.1),
            tags=[f"tag{i}"],
        )

    # List all
    all_mcps = temp_mcp_box.list_mcps()
    assert len(all_mcps) == 3

    # Filter by score
    high_score = temp_mcp_box.list_mcps(min_score=0.85)
    assert len(high_score) == 1
    assert high_score[0].name == "Tool 2"

    # Filter by tags
    tagged = temp_mcp_box.list_mcps(tags=["tag1"])
    assert len(tagged) == 1
    assert tagged[0].name == "Tool 1"


def test_search_by_purpose(temp_mcp_box):
    """Test semantic search by purpose."""
    # Store MCPs with different purposes
    temp_mcp_box.store_mcp(
        mcp_id="youtube_tool",
        name="YouTube Extractor",
        purpose="Extract subtitles from YouTube videos",
        spec={},
        main_script="print('youtube')",
    )
    temp_mcp_box.store_mcp(
        mcp_id="github_tool",
        name="GitHub Analyzer",
        purpose="Analyze GitHub repositories for code quality",
        spec={},
        main_script="print('github')",
    )

    # Search for YouTube-related tools
    results = temp_mcp_box.search_by_purpose("video subtitles")
    assert len(results) > 0
    assert results[0].name == "YouTube Extractor"

    # Search for GitHub-related tools
    results = temp_mcp_box.search_by_purpose("code analysis")
    assert len(results) > 0
    assert results[0].name == "GitHub Analyzer"


def test_record_usage(temp_mcp_box):
    """Test usage telemetry recording."""
    mcp_id = "test_tool"

    # Store MCP
    temp_mcp_box.store_mcp(
        mcp_id=mcp_id,
        name="Test Tool",
        purpose="Testing usage",
        spec={},
        main_script="print('test')",
    )

    # Record successful usage
    temp_mcp_box.record_usage(mcp_id, success=True, notes="Worked great")

    # Record failed usage
    temp_mcp_box.record_usage(mcp_id, success=False, notes="Error occurred")

    # Check usage log
    mcp_dir = temp_mcp_box.storage_dir / mcp_id
    usage_log_path = mcp_dir / "usage_log.json"
    assert usage_log_path.exists()

    with open(usage_log_path, encoding="utf-8") as f:
        usage_log = json.load(f)

    assert len(usage_log) == 2
    assert usage_log[0]["success"] is True
    assert usage_log[0]["notes"] == "Worked great"
    assert usage_log[1]["success"] is False


def test_success_rate(temp_mcp_box):
    """Test success rate calculation."""
    mcp_id = "test_tool"

    # Store MCP
    temp_mcp_box.store_mcp(
        mcp_id=mcp_id,
        name="Test Tool",
        purpose="Testing",
        spec={},
        main_script="print('test')",
    )

    # Record mixed usage
    temp_mcp_box.record_usage(mcp_id, success=True)
    temp_mcp_box.record_usage(mcp_id, success=True)
    temp_mcp_box.record_usage(mcp_id, success=True)
    temp_mcp_box.record_usage(mcp_id, success=False)

    # Check success rate
    mcp = temp_mcp_box.get_mcp(mcp_id)
    assert mcp["metadata"]["usage_count"] == 4
    assert mcp["metadata"]["success_rate"] == 0.75  # 3/4


def test_delete_mcp(temp_mcp_box):
    """Test deleting an MCP."""
    mcp_id = "test_tool"

    # Store MCP
    temp_mcp_box.store_mcp(
        mcp_id=mcp_id,
        name="Test Tool",
        purpose="Testing",
        spec={},
        main_script="print('test')",
    )

    # Verify it exists
    assert temp_mcp_box.get_mcp(mcp_id) is not None

    # Delete it
    success = temp_mcp_box.delete_mcp(mcp_id)
    assert success is True

    # Verify it's gone
    assert temp_mcp_box.get_mcp(mcp_id) is None


def test_update_mcp_metadata(temp_mcp_box):
    """Test updating MCP metadata."""
    mcp_id = "test_tool"

    # Store initial MCP
    temp_mcp_box.store_mcp(
        mcp_id=mcp_id,
        name="Test Tool",
        purpose="Testing",
        spec={},
        main_script="print('test')",
        constitutional_score=0.75,
        tags=["initial"],
    )

    # Update metadata
    success = temp_mcp_box.update_metadata(
        mcp_id=mcp_id,
        name="Updated Tool",
        purpose="Updated purpose",
        constitutional_score=0.90,
        tags=["updated", "improved"],
    )
    assert success is True

    # Verify updates
    mcp = temp_mcp_box.get_mcp(mcp_id)
    assert mcp["metadata"]["name"] == "Updated Tool"
    assert mcp["metadata"]["purpose"] == "Updated purpose"
    assert mcp["metadata"]["constitutional_score"] == 0.90
    assert "updated" in mcp["metadata"]["tags"]


def test_helper_functions():
    """Test helper functions for Copilot integration."""
    from src.mcp.mcp_box import get_mcp_summary, list_mcp_tools, store_mcp_tool

    # Create temp box
    temp_dir = tempfile.mkdtemp()
    os.environ["ALITA_MCP_BOX_PATH"] = temp_dir

    try:
        # Store via helper
        store_mcp_tool(
            mcp_id="test_helper",
            name="Helper Test",
            purpose="Testing helper functions",
            spec={},
            script="print('helper')",
            constitutional_score=0.88,
        )

        # List via helper
        output = list_mcp_tools(min_score=0.8)
        assert "Helper Test" in output
        assert "0.88" in output

        # Summary via helper
        summary = get_mcp_summary()
        assert "Total MCPs: 1" in summary
        assert "Average score:" in summary

    finally:
        shutil.rmtree(temp_dir)
        if "ALITA_MCP_BOX_PATH" in os.environ:
            del os.environ["ALITA_MCP_BOX_PATH"]


def test_constitutional_threshold_filtering(temp_mcp_box):
    """Test filtering by constitutional threshold."""
    # Store MCPs with varying scores
    temp_mcp_box.store_mcp(
        mcp_id="good_tool",
        name="Good Tool",
        purpose="High quality",
        spec={},
        main_script="print('good')",
        constitutional_score=0.92,
    )
    temp_mcp_box.store_mcp(
        mcp_id="ok_tool",
        name="OK Tool",
        purpose="Medium quality",
        spec={},
        main_script="print('ok')",
        constitutional_score=0.78,
    )
    temp_mcp_box.store_mcp(
        mcp_id="bad_tool",
        name="Bad Tool",
        purpose="Low quality",
        spec={},
        main_script="print('bad')",
        constitutional_score=0.62,
    )

    # Filter by 0.75 threshold (Super Alita standard)
    compliant = temp_mcp_box.list_mcps(min_score=0.75)
    assert len(compliant) == 2
    assert all(m.constitutional_score >= 0.75 for m in compliant)

    # Filter by 0.90 threshold (excellence)
    excellent = temp_mcp_box.list_mcps(min_score=0.90)
    assert len(excellent) == 1
    assert excellent[0].name == "Good Tool"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
