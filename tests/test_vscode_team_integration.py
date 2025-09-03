# tests/test_vscode_team_integration.py
import pytest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os

from src.ecosystem.vscode_bridge import VSCodeBridgeSimulator
from src.ecosystem.team_orchestrator import TeamProductivityOrchestrator


@pytest.mark.asyncio
@patch('aiohttp.ClientSession.post')
async def test_vscode_bridge_calls_orchestrator(mock_post):
    """
    Tests that the VSCodeBridgeSimulator correctly detects a TODO and
    makes a properly formatted HTTP call to the orchestrator service.
    """
    # Configure the mock response from the server
    mock_response = MagicMock()
    mock_response.status = 200
    async def json_body():
        return {"copilot_prompt": "test prompt", "vscode_snippets": []}
    mock_response.json = json_body
    mock_post.return_value.__aenter__.return_value = mock_response

    bridge = VSCodeBridgeSimulator()
    
    # Create a fake file to scan
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("# TODO: A test to-do item\n")
        f.write("def example_function():\n    pass\n")
        temp_file = f.name

    try:
        await bridge.scan_and_process_file(temp_file)

        # Assert that the HTTP POST call was made
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        request_payload = call_args.kwargs['json']
        
        assert request_payload['action'] == 'todo_detected'
        assert request_payload['context']['todo_text'] == 'A test to-do item'
        assert request_payload['context']['line_number'] == 1
        assert temp_file in request_payload['context']['file_path']
    finally:
        os.unlink(temp_file)


@pytest.mark.asyncio
async def test_vscode_bridge_handles_file_not_found():
    """Tests that the VS Code bridge handles missing files gracefully."""
    bridge = VSCodeBridgeSimulator()
    
    # This should not raise an exception
    await bridge.scan_and_process_file("nonexistent_file.py")


@pytest.mark.asyncio 
async def test_vscode_bridge_handles_connection_error():
    """Tests that the VS Code bridge handles connection errors gracefully."""
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.side_effect = Exception("Connection failed")
        
        bridge = VSCodeBridgeSimulator()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# TODO: test todo\n")
            temp_file = f.name
        
        try:
            # This should not raise an exception
            await bridge.scan_and_process_file(temp_file)
        finally:
            os.unlink(temp_file)


def test_team_orchestrator_aggregation():
    """
    Tests that the TeamProductivityOrchestrator correctly aggregates
    data from events and generates a meaningful summary.
    """
    team_orch = TeamProductivityOrchestrator()

    # Simulate consuming several events
    team_orch.consume_event("workflow.todo_resolution.completed", 
                           {"context": {"todo_text": "Fix bug in user authentication"}})
    team_orch.consume_event("workflow.todo_resolution.completed", 
                           {"context": {"todo_text": "Improve authentication performance"}})
    team_orch.consume_event("workflow.todo_resolution.completed", 
                           {"context": {"todo_text": "Refactor authentication module"}})

    summary = team_orch.generate_team_health_summary()

    assert summary['total_workflows_analyzed'] == 3
    
    # Check for suggested snippets
    suggestions = summary['optimizations']['suggested_snippet_libraries']
    assert len(suggestions) > 0
    auth_suggestion = next((s for s in suggestions if s['keyword'] == 'authentication'), None)
    assert auth_suggestion is not None
    assert auth_suggestion['occurrences'] == 3


def test_team_orchestrator_keyword_extraction():
    """Tests the keyword extraction logic in the team orchestrator."""
    team_orch = TeamProductivityOrchestrator()
    
    # Test keyword extraction
    keywords = team_orch._extract_keywords("Fix the authentication system bug")
    assert "authentication" in keywords
    assert "system" in keywords
    assert "bug" not in keywords  # Too short
    assert "the" not in keywords  # Stop word


def test_team_orchestrator_empty_events():
    """Tests that the team orchestrator handles empty events gracefully."""
    team_orch = TeamProductivityOrchestrator()
    
    # Process empty events
    team_orch.consume_event("workflow.todo_resolution.completed", {})
    team_orch.consume_event("unknown_event", {"data": "test"})
    
    summary = team_orch.generate_team_health_summary()
    assert summary['total_workflows_analyzed'] == 1
    assert len(summary['optimizations']['suggested_snippet_libraries']) == 0


def test_vscode_bridge_todo_detection():
    """Tests TODO detection patterns in the VS Code bridge."""
    bridge = VSCodeBridgeSimulator()
    
    # Test different TODO formats
    test_content = """
# TODO: Standard format
# todo: lowercase
#TODO:No space
    # TODO: Indented
// TODO: Different comment style (won't match)
    """
    
    import re
    todo_pattern = re.compile(r'#\s*TODO[:\s]*(.*)', re.IGNORECASE)
    matches = []
    for line in test_content.split('\n'):
        match = todo_pattern.search(line)
        if match:
            matches.append(match.group(1).strip())
    
    assert len(matches) == 4  # Should find 4 TODO items
    assert "Standard format" in matches
    assert "lowercase" in matches
    assert "No space" in matches
    assert "Indented" in matches


@pytest.mark.asyncio
@patch('aiohttp.ClientSession.post')
async def test_vscode_bridge_multiple_todos(mock_post):
    """Tests that the VS Code bridge processes multiple TODOs in a file."""
    # Configure the mock response
    mock_response = MagicMock()
    mock_response.status = 200
    async def json_body():
        return {"copilot_prompt": "test prompt", "vscode_snippets": []}
    mock_response.json = json_body
    mock_post.return_value.__aenter__.return_value = mock_response

    bridge = VSCodeBridgeSimulator()
    
    # Create a file with multiple TODOs
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("# TODO: First todo item\n")
        f.write("def function1():\n")
        f.write("    # TODO: Second todo item\n")
        f.write("    pass\n")
        temp_file = f.name

    try:
        await bridge.scan_and_process_file(temp_file)
        
        # Should have made two HTTP calls (one for each TODO)
        assert mock_post.call_count == 2
    finally:
        os.unlink(temp_file)