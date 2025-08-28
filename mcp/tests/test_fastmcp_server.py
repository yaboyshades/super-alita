#
# /mcp/tests/test_fastmcp_server.py
#
# Description: Unit tests for the FastMCP server. Mocks the OpenAI client to test
# `search` and `fetch` tool logic without making real API calls.
#

# Ensure the module under test is imported after patching environment variables if needed
import os
import types

import pytest

os.environ["OPENAI_API_KEY"] = "fake-key"
os.environ["VECTOR_STORE_ID"] = "fake-vs-id"
os.environ["MCP_ALLOW_NO_AUTH"] = "true"  # Disable auth for tests

from mcp import fastmcp_server


# --- Fake OpenAI SDK Objects ---
class FakeChunk:
    def __init__(self, text):
        self.text = text


class FakeFile:
    def __init__(self, file_id, filename, content):
        self.file_id = file_id
        self.filename = filename
        self.content = content


class FakeSearchResp:
    def __init__(self, data):
        self.data = data


class FakeContentResp:
    def __init__(self, data):
        self.data = data


class FakeRetrieveResp:
    def __init__(self, filename, attributes=None):
        self.filename = filename
        self.attributes = attributes


class FakeClient:
    def __init__(self):
        self.vector_stores = types.SimpleNamespace(
            search=self._search,
            files=types.SimpleNamespace(content=self._content, retrieve=self._retrieve),
        )
        self._files = {"file-123": ("Cats.pdf", [FakeChunk("Meow.\nPurr.")])}

    def _search(self, vector_store_id, query):
        if not query:
            return FakeSearchResp([])
        fake_file = FakeFile(
            "file-123",
            "Cats.pdf",
            [types.SimpleNamespace(text="Cats are great companions.")],
        )
        return FakeSearchResp([fake_file])

    def _content(self, vector_store_id, file_id):
        _filename, chunks = self._files[file_id]
        return FakeContentResp(chunks)

    def _retrieve(self, vector_store_id, file_id):
        filename, _ = self._files[file_id]
        return FakeRetrieveResp(filename, attributes={"topic": "cats"})


@pytest.fixture(autouse=True)
def mock_openai_client(monkeypatch):
    """Fixture to replace the real OpenAI client with our fake one for all tests."""
    fake_client_instance = FakeClient()
    # Patch the function that returns the client
    monkeypatch.setattr(
        fastmcp_server, "get_openai_client", lambda: fake_client_instance
    )


@pytest.mark.asyncio
async def test_search_and_fetch():
    """Tests a successful search followed by a fetch, verifying data shapes."""
    srv = fastmcp_server.create_server()
    search_tool = srv.tools["search"]
    fetch_tool = srv.tools["fetch"]

    # Test search
    search_result = await search_tool("cat behavior")
    assert "results" in search_result
    assert len(search_result["results"]) == 1

    first_item = search_result["results"][0]
    assert first_item["id"] == "file-123"
    assert first_item["title"] == "Cats.pdf"
    assert "Cats are great companions" in first_item["text"]
    assert first_item["url"] == "https://platform.openai.com/storage/files/file-123"

    # Test fetch using the ID from search
    doc_id = first_item["id"]
    fetch_result = await fetch_tool(doc_id)

    assert fetch_result["id"] == "file-123"
    assert fetch_result["title"] == "Cats.pdf"
    assert "Meow.\nPurr." in fetch_result["text"]
    assert fetch_result["url"] == "https://platform.openai.com/storage/files/file-123"
    assert "metadata" in fetch_result
    assert fetch_result["metadata"]["topic"] == "cats"


@pytest.mark.asyncio
async def test_empty_query_returns_empty_results():
    """Ensures that an empty search query returns an empty result list."""
    srv = fastmcp_server.create_server()
    search_tool = srv.tools["search"]

    result = await search_tool("")
    assert "results" in result
    assert result["results"] == []

    result_whitespace = await search_tool("   ")
    assert "results" in result_whitespace
    assert result_whitespace["results"] == []


@pytest.mark.asyncio
async def test_fetch_with_empty_id_raises_error():
    """Ensures fetching with an empty ID raises a ValueError."""
    srv = fastmcp_server.create_server()
    fetch_tool = srv.tools["fetch"]
    with pytest.raises(ValueError, match="Document ID is required"):
        await fetch_tool("")
