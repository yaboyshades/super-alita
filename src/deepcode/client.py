"""DeepCode HTTP client for real service integration.

Uses environment variables:
  DEEPCODE_API_URL - Base URL of DeepCode service (e.g. https://deepcode.example.com)
  DEEPCODE_API_KEY - API key / token for authentication (optional)

Provides a minimal interface compatible with DeepCodeClientInterface used by the
orchestrator plugin. Each method returns structured dicts matching the stub
client so downstream pipeline remains unchanged.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_TIMEOUT = 30.0


class DeepCodeHTTPError(RuntimeError):
    pass


@dataclass
class _APIConfig:
    base_url: str
    api_key: str | None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)


def _load_config() -> _APIConfig:
    return _APIConfig(
        base_url=os.getenv("DEEPCODE_API_URL", "").strip().rstrip("/"),
        api_key=os.getenv("DEEPCODE_API_KEY"),
    )


class DeepCodeHTTPClient:
    """HTTP client implementing the DeepCodeClientInterface.

    The remote API is assumed to expose endpoints:
      POST /plan               -> plan dict
      POST /collect_references -> references dict
      POST /generate           -> implementation (diffs/tests/docs)
      POST /validate           -> validation dict

    If any endpoint fails the client raises DeepCodeHTTPError. The orchestrator
    catches generic exceptions and emits failure events, so we keep surface
    narrow here.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self._cfg = _load_config()
        self._timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    async def close(self) -> None:  # pragma: no cover - lifecycle convenience
        with contextlib.suppress(Exception):  # type: ignore[name-defined]
            await self._client.aclose()

    async def _post(
        self, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if not self._cfg.enabled:
            raise DeepCodeHTTPError(
                "DeepCode HTTP client not enabled (missing DEEPCODE_API_URL)"
            )
        url = f"{self._cfg.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self._cfg.api_key:
            # simple bearer model
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"
        try:
            resp = await self._client.post(
                url, content=json.dumps(payload), headers=headers
            )
        except Exception as e:  # pragma: no cover - network failure
            raise DeepCodeHTTPError(f"Request failed: {e}") from e
        if resp.status_code >= 400:
            raise DeepCodeHTTPError(
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            return resp.json()
        except Exception as e:  # pragma: no cover - malformed JSON
            raise DeepCodeHTTPError(f"Invalid JSON response: {e}") from e

    # Interface methods -------------------------------------------------
    async def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/plan", request)

    async def collect_references(self, plan: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/collect_references", {"plan": plan})

    async def generate_code(
        self, plan: dict[str, Any], references: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._post(
            "/generate", {"plan": plan, "references": references}
        )

    async def validate(self, implementation: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/validate", implementation)


__all__ = ["DeepCodeHTTPClient", "DeepCodeHTTPError"]
