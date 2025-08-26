"""Tests for Puter plugin integration."""

import pytest
pytest.skip("legacy Puter plugin tests", allow_module_level=True)

from unittest.mock import AsyncMock, patch
import aiohttp
import json
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from typing import Any

from src.puter.plugin_registry import PluginRegistry
from src.puter.puter_plugin import PuterAPIError, PuterPlugin
from src.puter.tools_registry import PuterTool
from tests.runtime.puter_fakes import FakePuterServer


class TestPuterPlugin(AioHTTPTestCase):
    async def get_application(self) -> web.Application:  # type: ignore[override]

        self.fake_server = FakePuterServer()
        return self.fake_server.create_app()

        return FakePuterServer().create_app()

    async def setUpAsync(self) -> None:
        await super().setUpAsync()
        self.server_url = f"http://localhost:{self.server.port}"
        self.config = {
            "base_url": self.server_url,
            "api_key": "test-api-key",
            "timeout": 5,
        }
        self.plugin = PuterPlugin(self.config)
        await self.plugin.initialize()

    async def tearDownAsync(self) -> None:
        await self.plugin.cleanup()
        await super().tearDownAsync()

    @unittest_run_loop
    async def test_plugin_initialization(self) -> None:
        assert self.plugin.session is not None
        assert self.plugin.base_url == self.server_url
        assert self.plugin.api_key == "test-api-key"

    @unittest_run_loop
    async def test_read_file_success(self) -> None:
        content = await self.plugin.read_file("/test/file.txt")
        assert content == "test file content"

    @unittest_run_loop
    async def test_write_file_success(self) -> None:
        result = await self.plugin.write_file("/test/new_file.txt", "new content")
        assert result is True

    @unittest_run_loop
    async def test_execute_command_success(self) -> None:
        result = await self.plugin.execute_command("echo", ["hello world"])
        assert result["stdout"] == "hello world\n"
        assert result["exit_code"] == 0

    @unittest_run_loop
    async def test_list_directory_success(self) -> None:
        items = await self.plugin.list_directory("/test")
        assert isinstance(items, list)
        assert len(items) > 0

    @unittest_run_loop
    async def test_change_directory_success(self) -> None:
        new_dir = await self.plugin.change_directory("/test")
        assert new_dir == "/test"
        assert self.plugin.get_current_directory() == "/test"

    @unittest_run_loop
    async def test_api_error_handling(self) -> None:
        with pytest.raises(PuterAPIError):
            await self.plugin.read_file("/nonexistent/file.txt")


    @unittest_run_loop
    async def test_delete_file_no_content(self) -> None:
        result = await self.plugin.delete_file("/test/file.txt")
        assert result is True

    @unittest_run_loop
    async def test_retry_on_503(self) -> None:
        result = await self.plugin._make_request("GET", "/api/flaky")
        assert result["status"] == "ok"
        assert self.fake_server.flaky_calls == 3

class TestPuterTool(AioHTTPTestCase):
    async def get_application(self) -> web.Application:  # type: ignore[override]
        return FakePuterServer().create_app()

    async def setUpAsync(self) -> None:
        await super().setUpAsync()
        self.server_url = f"http://localhost:{self.server.port}"
        self.registry = PluginRegistry()
        await self.registry.initialize_plugin(
            "puter", {"base_url": self.server_url, "api_key": "test-api-key"}
        )
        self.tool = PuterTool(self.registry)
        await self.tool.initialize()

    async def tearDownAsync(self) -> None:
        await self.registry.cleanup_all()
        await super().tearDownAsync()

    @unittest_run_loop
    async def test_tool_initialization(self) -> None:
        assert self.tool.puter_plugin is not None

    @unittest_run_loop
    async def test_execute_command_with_events(self) -> None:
        with patch.object(self.tool, "_emit_event", new_callable=AsyncMock) as mock_emit:
            result = await self.tool.execute_command("echo", ["test"])
            assert result["exit_code"] == 0
            mock_emit.assert_called_once_with(
                "command_executed",
                {
                    "command": "echo",
                    "args": ["test"],
                    "exit_code": 0,
                    "execution_time": result["execution_time"],
                },
            )

    @unittest_run_loop
    async def test_file_operations_with_events(self) -> None:
        with patch.object(self.tool, "_emit_event", new_callable=AsyncMock) as mock_emit:
            await self.tool.write_file("/test/file.txt", "content")
            mock_emit.assert_called_with(
                "file_written", {"path": "/test/file.txt", "size": 7, "created_dirs": True}
            )
            mock_emit.reset_mock()
            content = await self.tool.read_file("/test/file.txt")
            mock_emit.assert_called_with(
                "file_read", {"path": "/test/file.txt", "size": len(content)}
            )


@pytest.mark.asyncio
async def test_plugin_registry_integration() -> None:
    registry = PluginRegistry()
    available = registry.list_available_plugins()
    assert "puter" in available
    config = {"base_url": "http://fake-server", "api_key": "test-key"}
    with patch.object(PuterPlugin, "initialize", new_callable=AsyncMock):
        await registry.initialize_plugin("puter", config)
        initialized = registry.list_initialized_plugins()
        assert "puter" in initialized
        plugin = registry.get_plugin("puter")
        assert isinstance(plugin, PuterPlugin)
    await registry.cleanup_all()


@pytest.mark.asyncio
async def test_error_handling_and_retries() -> None:
    config = {"base_url": "http://unreachable-server", "api_key": "test-key", "max_retries": 2}
    plugin = PuterPlugin(config)

    class FailingSession:
        def __init__(self) -> None:
            self.request_call_count = 0

        def request(self, *args: Any, **kwargs: Any):
            self.request_call_count += 1

            class Ctx:
                async def __aenter__(self_inner):
                    raise aiohttp.ClientError("Connection failed")

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

    session = FailingSession()
    plugin.session = session  # type: ignore[assignment]
    with pytest.raises(PuterAPIError):
        await plugin._make_request("GET", "/api/test")
    assert session.request_call_count == 3



@pytest.mark.asyncio
async def test_worker_hmac_signing() -> None:
    config = {
        "base_url": "https://api.example",
        "worker": {
            "enabled": True,
            "base_url": "https://worker.example",
            "shared_secret": "secret",
        },
        "skip_healthcheck": True,
    }
    plugin = PuterPlugin(config)
    await plugin.initialize()

    def fake_request(method: str, url: str, data=None, params=None, headers=None):
        expected_sig = plugin._hmac_sha256_hex("secret", json.dumps({"foo": "bar"}))
        assert url == "https://worker.example/api/test"
        assert headers["x-reug-sig"] == expected_sig

        class Resp:
            status = 200

            async def json(self, content_type=None):
                return {"ok": True}

            async def text(self):
                return ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return Resp()

    plugin.session.request = fake_request  # type: ignore[assignment]
    result = await plugin._make_request("POST", "/api/test", data={"foo": "bar"})
    assert result == {"ok": True}
    await plugin.cleanup()

