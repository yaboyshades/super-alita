#!/usr/bin/env python3
"""
Tests for Mangle integration with Super Alita.

This module contains tests for the Mangle deductive database integration
with the Super Alita system, verifying logical reasoning capabilities.
"""

import asyncio
import os
import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abilities.mangle.mangle_ability import MangleAbility


class MockProcess:
    """Mock subprocess.CompletedProcess for testing."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture
def mock_mangle_ability(monkeypatch):
    """Create a MangleAbility instance with mocked subprocess."""
    import subprocess

    # Mock the subprocess.run method
    def mock_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("cmd", [])

        # Default mock data
        stdout_data = "[]"
        stderr_data = ""

        # Mock different responses based on the command
        if "--explain" in cmd:
            stderr_data = "Explanation steps:\n1. Fact a(1) matches\n2. Rule r1 applies\n"

        if "vulnerable" in " ".join(cmd):
            stdout_data = '[{"Name": "log4j", "Version": "2.14.0"}]'

        return MockProcess(stdout=stdout_data, stderr=stderr_data, returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    # Create a temporary directory for knowledge base
    import tempfile
    temp_dir = tempfile.mkdtemp()

    # Create the ability with the temp dir
    ability = MangleAbility(config={"knowledge_base_dir": temp_dir})

    yield ability

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_add_fact(mock_mangle_ability):
    """Test adding facts to the knowledge base."""
    # Add a fact
    result = await mock_mangle_ability.add_fact("dependency('log4j', '2.14.0')")

    assert result["success"] is True
    assert "dependency('log4j', '2.14.0')." in mock_mangle_ability.facts
    assert result["total_facts"] == 1

    # Add another fact
    result = await mock_mangle_ability.add_fact("vulnerability('log4j', '2.14.0', 'CVE-2021-44228')")

    assert result["success"] is True
    assert "vulnerability('log4j', '2.14.0', 'CVE-2021-44228')." in mock_mangle_ability.facts
    assert result["total_facts"] == 2


@pytest.mark.asyncio
async def test_add_rule(mock_mangle_ability):
    """Test adding rules to the knowledge base."""
    # Add a rule
    rule = "vulnerable(Lib, Ver) :- dependency(Lib, Ver), vulnerability(Lib, Ver, _)"
    result = await mock_mangle_ability.add_rule("detect_vulnerable", rule)

    assert result["success"] is True
    assert rule + "." in mock_mangle_ability.rules["detect_vulnerable"]
    assert result["total_rules"] == 1


@pytest.mark.asyncio
async def test_query(mock_mangle_ability):
    """Test executing a query against the knowledge base."""
    # Add prerequisite data
    await mock_mangle_ability.add_fact("dependency('log4j', '2.14.0')")
    await mock_mangle_ability.add_fact("vulnerability('log4j', '2.14.0', 'CVE-2021-44228')")
    await mock_mangle_ability.add_rule(
        "detect_vulnerable",
        "vulnerable(Lib, Ver) :- dependency(Lib, Ver), vulnerability(Lib, Ver, _)"
    )

    # Execute query
    result = await mock_mangle_ability.query("vulnerable(Name, Version)")

    assert result["success"] is True
    assert len(result["results"]) > 0
    assert result["results"][0]["Name"] == "log4j"
    assert result["results"][0]["Version"] == "2.14.0"


@pytest.mark.asyncio
async def test_analyze_dependencies(mock_mangle_ability):
    """Test dependency analysis functionality."""
    # Define test dependencies
    dependencies = [
        {"name": "log4j", "version": "2.14.0"},
        {"name": "spring-core", "version": "5.3.20"}
    ]

    # Analyze dependencies
    result = await mock_mangle_ability.analyze_dependencies(dependencies)

    assert result["success"] is True
    assert len(result["results"]) > 0
    # Our mock always returns log4j as vulnerable
    assert any(r["Name"] == "log4j" for r in result["results"])


@pytest.mark.asyncio
async def test_knowledge_graph_query(mock_mangle_ability):
    """Test knowledge graph query capabilities."""
    # Add base facts
    await mock_mangle_ability.add_fact("depends_on('app', 'lib1')")
    await mock_mangle_ability.add_fact("depends_on('lib1', 'lib2')")

    # Define context for the query
    context = [
        {"relation": "vulnerable", "subject": "lib2", "object": "CVE-2023-1234"}
    ]

    # Execute knowledge graph query
    result = await mock_mangle_ability.knowledge_graph_query(
        "transitive_dependency(X, Y) :- depends_on(X, Y)",
        context
    )

    assert result["success"] is True


@pytest.mark.asyncio
async def test_explain_query_results(mock_mangle_ability):
    """Test query explanation functionality."""
    # Add test data
    await mock_mangle_ability.add_fact("dependency('log4j', '2.14.0')")
    await mock_mangle_ability.add_fact("vulnerability('log4j', '2.14.0', 'CVE-2021-44228')")
    await mock_mangle_ability.add_rule(
        "detect_vulnerable",
        "vulnerable(Lib, Ver) :- dependency(Lib, Ver), vulnerability(Lib, Ver, _)"
    )

    # Get explanation for a query
    result = await mock_mangle_ability.explain_query_results("vulnerable(Name, Version)")

    assert result["success"] is True
    assert "explanation" in result
    assert len(result["explanation"]) > 0
    assert "Explanation steps" in result["explanation"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
