from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ecosystem.master_orchestrator import GitHubExample
from src.ecosystem.tools import GitHubTool


@pytest.mark.asyncio
async def test_find_github_examples_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "items": [
            {
                "path": "src/main.py",
                "repository": {
                    "full_name": "test/repo1",
                    "license": {"name": "MIT"},
                },
                "text_matches": [{"fragment": "def my_func():"}],
            }
        ]
    }

    async_mock_client = MagicMock()
    async_mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient") as mock_async_client_cls:
        mock_async_client_cls.return_value.__aenter__.return_value = (
            async_mock_client
        )

        tool = GitHubTool(token="fake_token")
        results = await tool.find_github_examples("test query")

        assert len(results) == 1
        assert isinstance(results[0], GitHubExample)
        assert results[0].repo == "test/repo1"
        assert results[0].code_snippet == "def my_func():"


@pytest.mark.asyncio
async def test_find_github_examples_api_error():
    async_mock_client = MagicMock()
    async_mock_client.get = AsyncMock(side_effect=Exception("Network Error"))

    with patch("httpx.AsyncClient") as mock_async_client_cls:
        mock_async_client_cls.return_value.__aenter__.return_value = (
            async_mock_client
        )

        tool = GitHubTool(token="fake_token")
        results = await tool.find_github_examples("test query")

        assert results == []
