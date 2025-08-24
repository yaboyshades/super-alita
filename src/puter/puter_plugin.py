"""
Puter Plugin for Agent Framework Integration

This plugin provides seamless integration with Puter's cloud environment,
enabling file I/O operations and process execution through Puter's API.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import aiohttp

from .plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class PuterAPIError(Exception):
    """Custom exception for Puter API errors"""


class PuterPlugin(PluginInterface):
    """Plugin for integrating with Puter cloud environment."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://puter.com")
        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.session: Optional[aiohttp.ClientSession] = None
        self.current_directory = "/"

    async def initialize(self) -> None:
        """Initialize the plugin and establish connection to Puter."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = {"User-Agent": "PuterAgent/1.0", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.session = aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=headers
        )

        try:
            await self._make_request("GET", "/api/health")
            logger.info("Successfully connected to Puter instance")
            self.is_initialized = True
        except Exception as exc:  # pragma: no cover - network failure
            logger.error("Failed to connect to Puter: %s", exc)
            raise PuterAPIError(f"Connection failed: {exc}")

    async def cleanup(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None
            self.is_initialized = False

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        if not self.session:
            raise PuterAPIError("Plugin not initialized")

        url = urljoin(self.base_url, endpoint)
        try:
            async with self.session.request(
                method, url, json=data, params=params
            ) as response:
                response_data = await response.json()
                if response.status >= 400:
                    error_msg = response_data.get("error", f"HTTP {response.status}")
                    raise PuterAPIError(f"API Error: {error_msg}")
                return response_data
        except aiohttp.ClientError as exc:
            if retry_count < self.max_retries:
                await asyncio.sleep(2 ** retry_count)
                return await self._make_request(
                    method, endpoint, data, params, retry_count + 1
                )
            raise PuterAPIError(f"Network error: {exc}")

    def _resolve_path(self, path: str) -> str:
        if path.startswith("/"):
            return path
        if path == ".":
            return self.current_directory
        if path == "..":
            return str(Path(self.current_directory).parent)
        return str(Path(self.current_directory) / path)

    # File I/O operations
    async def read_file(self, path: str) -> str:
        full_path = self._resolve_path(path)
        response = await self._make_request(
            "GET", "/api/fs/read", params={"path": full_path}
        )
        return response.get("content", "")

    async def write_file(self, path: str, content: str, create_dirs: bool = True) -> bool:
        full_path = self._resolve_path(path)
        data = {"path": full_path, "content": content, "create_dirs": create_dirs}
        await self._make_request("POST", "/api/fs/write", data=data)
        return True

    async def list_directory(self, path: str = ".") -> List[Dict[str, Any]]:
        full_path = self._resolve_path(path)
        response = await self._make_request(
            "GET", "/api/fs/list", params={"path": full_path}
        )
        return response.get("items", [])

    async def delete_file(self, path: str) -> bool:
        full_path = self._resolve_path(path)
        await self._make_request("DELETE", "/api/fs/delete", params={"path": full_path})
        return True

    async def create_directory(self, path: str) -> bool:
        full_path = self._resolve_path(path)
        await self._make_request("POST", "/api/fs/mkdir", data={"path": full_path})
        return True

    # Process execution
    async def execute_command(
        self,
        command: str,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        working_dir = self._resolve_path(cwd) if cwd else self.current_directory
        data = {
            "command": command,
            "args": args or [],
            "cwd": working_dir,
            "env": env or {},
        }
        response = await self._make_request("POST", "/api/exec", data=data)
        return {
            "stdout": response.get("stdout", ""),
            "stderr": response.get("stderr", ""),
            "exit_code": response.get("exit_code", 0),
            "execution_time": response.get("execution_time", 0),
        }

    # Directory navigation
    async def change_directory(self, path: str) -> str:
        new_path = self._resolve_path(path)
        try:
            await self.list_directory(new_path)
            self.current_directory = new_path
            return self.current_directory
        except PuterAPIError as exc:
            raise PuterAPIError(f"Directory does not exist: {new_path}") from exc

    def get_current_directory(self) -> str:
        return self.current_directory

    async def get_file_info(self, path: str) -> Dict[str, Any]:
        full_path = self._resolve_path(path)
        return await self._make_request(
            "GET", "/api/fs/stat", params={"path": full_path}
        )

    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            "name": "PuterPlugin",
            "version": "1.0.0",
            "description": "Integration with Puter cloud environment",
            "capabilities": [
                "file_io",
                "process_execution",
                "directory_management",
            ],
        }
