"""
Puter Plugin for Agent Framework Integration

This plugin provides seamless integration with Puter's cloud environment,
enabling file I/O operations and process execution through Puter's API.
"""

import asyncio
import logging

import json
import random
from pathlib import PurePosixPath

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

        # Two base URLs: raw API vs Worker bridge (optional)
        self.base_url = config.get("base_url", "https://puter.com")
        worker_cfg = config.get("worker") or {}
        self.worker_enabled: bool = bool(worker_cfg.get("enabled"))
        self.worker_base_url: Optional[str] = worker_cfg.get("base_url")
        self.hmac_secret: Optional[str] = worker_cfg.get("shared_secret")
        self.hmac_header: str = worker_cfg.get("hmac_header", "x-reug-sig")

        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.retriable_statuses = set(config.get("retriable_statuses", [502, 503, 504]))
        self.skip_healthcheck = bool(config.get("skip_healthcheck", False))

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

        # NOTE: don't set JSON Content-Type globally; set it per request if needed
        headers = {"User-Agent": "PuterAgent/1.0"}

        headers = {"User-Agent": "PuterAgent/1.0", "Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers,
        )

        # Health check (optional; some deployments may not expose it)
        if not self.skip_healthcheck:
            try:
                await self._make_request("GET", "/api/health", expect_json=True)
                logger.info("Successfully connected to Puter instance")
            except Exception as exc:  # pragma: no cover - network failure
                logger.error("Failed to connect to Puter: %s", exc)
                raise PuterAPIError(f"Connection failed: {exc}")

        self.is_initialized = True

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

        expect_json: bool = True,

    ) -> Dict[str, Any]:
        if not self.session:
            raise PuterAPIError("Plugin not initialized")


        # Decide which base to use
        base = self.worker_base_url if (self.worker_enabled and self.worker_base_url) else self.base_url
        url = urljoin(base, endpoint)
        json_body: Optional[str] = None
        headers: Dict[str, str] = {}
        # For JSON requests, build body string and set content-type
        if method.upper() in {"POST", "PUT", "PATCH"}:
            json_body = json.dumps(data or {})
            headers["Content-Type"] = "application/json"
        # HMAC signing for Worker mode
        if self.worker_enabled and self.hmac_secret:
            body_for_sig = json_body if json_body is not None else ""
            sig = self._hmac_sha256_hex(self.hmac_secret, body_for_sig)
            headers[self.hmac_header] = sig

        try:
            async with self.session.request(
                method,
                url,
                data=json_body if json_body is not None else None,
                params=params,
                headers=headers or None,
            ) as response:
                status = response.status
                # Quick exit for 204
                if status == 204:
                    return {}
                # Prefer JSON, but be resilient to wrong Content-Type
                raw_text: Optional[str] = None
                parsed: Optional[Dict[str, Any]] = None
                try:
                    parsed = await response.json(content_type=None)
                except aiohttp.ContentTypeError:
                    raw_text = await response.text()
                    if expect_json:
                        # Best-effort JSON parse even if content-type is wrong
                        try:
                            parsed = json.loads(raw_text)
                        except json.JSONDecodeError:
                            parsed = None
                except json.JSONDecodeError:
                    raw_text = await response.text()
                    parsed = None

                if status >= 400:
                    # Retry on configured retriable statuses
                    if status in self.retriable_statuses and retry_count < self.max_retries:
                        await asyncio.sleep(min(2 ** retry_count + random.random(), 8))
                        return await self._make_request(
                            method,
                            endpoint,
                            data,
                            params,
                            retry_count + 1,
                            expect_json=expect_json,
                        )
                    message = None
                    if parsed and isinstance(parsed, dict):
                        message = parsed.get("error") or parsed.get("message")
                    if not message:
                        message = raw_text or f"HTTP {status}"
                    raise PuterAPIError(f"API Error: {message}")

                # Return JSON object if available, otherwise wrap text
                if parsed is not None:
                    return parsed if isinstance(parsed, dict) else {"data": parsed}
                # Non-JSON success
                txt = raw_text if raw_text is not None else await response.text()
                return {"data": txt}
        except aiohttp.ClientError as exc:
            if retry_count < self.max_retries:
                await asyncio.sleep(min(2 ** retry_count + random.random(), 8))
                return await self._make_request(
                    method,
                    endpoint,
                    data,
                    params,
                    retry_count + 1,
                    expect_json=expect_json,

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

            return str(PurePosixPath(self.current_directory).parent) or "/"
        return str(PurePosixPath(self.current_directory) / path)

            return str(Path(self.current_directory).parent)
        return str(Path(self.current_directory) / path)


    # File I/O operations
    async def read_file(self, path: str) -> str:
        full_path = self._resolve_path(path)
        response = await self._make_request(

            "GET", "/api/fs/read", params={"path": full_path}, expect_json=True

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

            "GET", "/api/fs/list", params={"path": full_path}, expect_json=True

            "GET", "/api/fs/list", params={"path": full_path}

        )
        return response.get("items", [])

    async def delete_file(self, path: str) -> bool:
        full_path = self._resolve_path(path)

        await self._make_request(
            "DELETE", "/api/fs/delete", params={"path": full_path}, expect_json=True
        )

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

            "GET", "/api/fs/stat", params={"path": full_path}, expect_json=True

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


    @staticmethod
    def _hmac_sha256_hex(secret: str, body: str) -> str:
        import hmac
        import hashlib

        digest = hmac.new(
            key=secret.encode("utf-8"),
            msg=body.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        return digest
