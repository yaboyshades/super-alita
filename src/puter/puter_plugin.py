"""
Puter Plugin for Agent Framework Integration

This plugin provides seamless integration with Puter's cloud environment,
enabling file I/O operations and process execution through Puter's API.
"""

import asyncio
import json
import logging
import random
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urljoin

import aiohttp

from .plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class PuterAPIError(Exception):
    """Custom exception for Puter API errors"""


class PuterPlugin(PluginInterface):
    """Plugin for integrating with Puter cloud environment."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        # Two base URLs: raw API vs Worker bridge (optional)
        self.base_url = config.get("base_url", "https://puter.com")
        worker_cfg = config.get("worker") or {}
        self.worker_enabled: bool = bool(worker_cfg.get("enabled"))
        self.worker_base_url: str | None = worker_cfg.get("base_url")
        self.hmac_secret: str | None = worker_cfg.get("shared_secret")
        self.hmac_header: str = worker_cfg.get("hmac_header", "x-reug-sig")

        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.retriable_statuses = set(
            config.get("retriable_statuses", [502, 503, 504])
        )
        self.skip_healthcheck = bool(config.get("skip_healthcheck", False))

        self.current_directory = "/"
        self.session: aiohttp.ClientSession | None = None
        self.is_initialized = False

    def _get_base_url(self) -> str:
        """Return appropriate base URL (worker or direct)."""
        if self.worker_enabled and self.worker_base_url:
            return self.worker_base_url
        return self.base_url

    async def initialize(self) -> None:
        """Initialize the plugin and establish connection."""
        if self.is_initialized:
            return

        # Create session with appropriate headers
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Configure connector and timeout
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        self.session = aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=headers
        )

        # Health check unless skipped
        if not self.skip_healthcheck:
            try:
                await self._make_request(
                    "GET", "/api/health", expect_json=True
                )
                logger.info("Successfully connected to Puter instance")
            except Exception as exc:  # pragma: no cover - network failure
                logger.error("Failed to connect to Puter: %s", exc)
                raise PuterAPIError(f"Connection failed: {exc}") from exc

        self.is_initialized = True

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_initialized = False

    def _sign_request(self, body: str) -> str:
        """Generate HMAC signature for worker authentication."""
        if not self.hmac_secret:
            raise ValueError("HMAC secret not configured")
        return self._hmac_sha256_hex(self.hmac_secret, body)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        retry_count: int = 0,
        expect_json: bool = False,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic and error handling."""
        if not self.session:
            raise RuntimeError("Plugin not initialized")

        # Use worker URL if available, otherwise direct API
        base = self._get_base_url()
        url = urljoin(base, endpoint)

        # Prepare headers
        headers = {}
        body_for_sig = ""

        if data:
            body_for_sig = json.dumps(data, sort_keys=True)
            if self.worker_enabled and self.hmac_secret:
                sig = self._hmac_sha256_hex(self.hmac_secret, body_for_sig)
                headers[self.hmac_header] = sig

        try:
            async with self.session.request(
                method, url, json=data, params=params, headers=headers
            ) as response:
                status = response.status
                raw_text = None
                parsed = None

                # Try to parse JSON if content type suggests it
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        parsed = await response.json()
                    except Exception:
                        raw_text = await response.text()
                else:
                    raw_text = await response.text()

                # Handle errors with retry logic
                if status >= 400:
                    if (
                        status in self.retriable_statuses
                        and retry_count < self.max_retries
                    ):
                        await asyncio.sleep(
                            min(2**retry_count + random.random(), 8)
                        )
                        return await self._make_request(
                            method,
                            endpoint,
                            data,
                            params,
                            retry_count + 1,
                            expect_json=expect_json,
                        )

                    # Extract error message
                    message = ""
                    if parsed and isinstance(parsed, dict):
                        message = parsed.get("error", "")
                    if not message:
                        message = raw_text or f"HTTP {status}"
                    raise PuterAPIError(f"API Error: {message}")

                # Return JSON object if available, otherwise wrap text
                if parsed is not None:
                    return (
                        parsed
                        if isinstance(parsed, dict)
                        else {"data": parsed}
                    )
                # Non-JSON success
                txt = (
                    raw_text if raw_text is not None else await response.text()
                )
                return {"data": txt}
        except aiohttp.ClientError as exc:
            if retry_count < self.max_retries:
                await asyncio.sleep(min(2**retry_count + random.random(), 8))
                return await self._make_request(
                    method,
                    endpoint,
                    data,
                    params,
                    retry_count + 1,
                    expect_json=expect_json,
                )
            raise PuterAPIError(f"Network error: {exc}") from exc

    def _resolve_path(self, path: str) -> str:
        """Resolve relative paths to absolute paths."""
        if path.startswith("/"):
            return path
        if path == ".":
            return self.current_directory
        if path == "..":
            return str(PurePosixPath(self.current_directory).parent) or "/"
        return str(PurePosixPath(self.current_directory) / path)

    # File I/O operations
    async def read_file(self, path: str) -> str:
        """Read file content from Puter."""
        full_path = self._resolve_path(path)
        response = await self._make_request(
            "GET", "/api/fs/read", params={"path": full_path}, expect_json=True
        )
        return response.get("content", "")

    async def write_file(
        self, path: str, content: str, create_dirs: bool = True
    ) -> bool:
        """Write file content to Puter."""
        full_path = self._resolve_path(path)
        data = {
            "path": full_path,
            "content": content,
            "create_dirs": create_dirs,
        }
        await self._make_request("POST", "/api/fs/write", data=data)
        return True

    async def list_directory(self, path: str = ".") -> list[dict[str, Any]]:
        """List directory contents."""
        full_path = self._resolve_path(path)
        response = await self._make_request(
            "GET", "/api/fs/list", params={"path": full_path}, expect_json=True
        )
        return response.get("items", [])

    async def delete_file(self, path: str) -> bool:
        """Delete a file."""
        full_path = self._resolve_path(path)
        await self._make_request(
            "DELETE",
            "/api/fs/delete",
            params={"path": full_path},
            expect_json=True,
        )
        return True

    async def create_directory(self, path: str) -> bool:
        """Create a directory."""
        full_path = self._resolve_path(path)
        await self._make_request(
            "POST", "/api/fs/mkdir", data={"path": full_path}
        )
        return True

    # Process execution
    async def execute_command(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Execute a command in the Puter environment."""
        working_dir = self._resolve_path(cwd or ".")
        data = {
            "command": command,
            "args": args or [],
            "cwd": working_dir,
        }
        return await self._make_request("POST", "/api/exec", data=data)

    # Directory navigation
    async def change_directory(self, path: str) -> str:
        """Change current working directory."""
        new_path = self._resolve_path(path)
        try:
            await self.list_directory(new_path)
            self.current_directory = new_path
            return self.current_directory
        except PuterAPIError as exc:
            raise PuterAPIError(
                f"Directory does not exist: {new_path}"
            ) from exc

    def get_current_directory(self) -> str:
        """Get current working directory."""
        return self.current_directory

    async def get_file_info(self, path: str) -> dict[str, Any]:
        """Get file/directory information."""
        full_path = self._resolve_path(path)
        return await self._make_request(
            "GET", "/api/fs/stat", params={"path": full_path}, expect_json=True
        )

    def get_plugin_info(self) -> dict[str, Any]:
        """Get plugin metadata."""
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

    @staticmethod
    def _hmac_sha256_hex(secret: str, body: str) -> str:
        """Generate HMAC-SHA256 signature."""
        import hashlib
        import hmac

        return hmac.new(
            key=secret.encode("utf-8"),
            msg=body.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
