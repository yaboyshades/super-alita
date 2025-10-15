"""Test configuration and shared fixtures."""

import pytest
import asyncio
import os
import tempfile
import shutil
from fastapi.testclient import TestClient

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_db():
    """Create a temporary database directory for tests."""
    temp_dir = tempfile.mkdtemp()
    os.environ["CHROMA_DB_PATH"] = temp_dir
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_chat_message():
    return {
        "message": "Hello, can you help me with Python?",
        "session_id": "test-session-123"
    }

@pytest.fixture
def sample_code_analysis():
    return {
        "code": "def hello_world():\n    print('Hello, World!')",
        "language": "python"
    }