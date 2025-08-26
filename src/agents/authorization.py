"""OAuth authorization utilities for MCP models.

This module provides a minimal OAuth token exchange implementation
compatible with the MCP specification used by VS Code's `vscode.lm`
API.  It also exposes helpers to gate model access based on the
scopes returned in the access token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableSet

import httpx


@dataclass(slots=True)
class OAuthConfig:
    """Configuration for exchanging an OAuth authorization code."""

    token_url: str
    client_id: str
    redirect_uri: str | None = None


class AuthorizationManager:
    """Handle OAuth token exchange and scope based gating."""

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        model_preferences: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self._client = http_client or httpx.AsyncClient()
        self._model_prefs: dict[str, MutableSet[str]] = {
            model: set(scopes) for model, scopes in (model_preferences or {}).items()
        }
        self.access_token: str | None = None
        self.scopes: set[str] = set()

    async def exchange_token(self, config: OAuthConfig, code: str) -> str:
        """Exchange an authorization code for an access token.

        Parameters
        ----------
        config:
            OAuthConfig containing token URL and client id.
        code:
            The authorization code returned from the OAuth grant flow.
        """

        payload: dict[str, str] = {
            "grant_type": "authorization_code",
            "client_id": config.client_id,
            "code": code,
        }
        if config.redirect_uri:
            payload["redirect_uri"] = config.redirect_uri

        response = await self._client.post(config.token_url, data=payload)
        response.raise_for_status()
        data = response.json()
        token = data["access_token"]
        self.access_token = token
        scope_str = data.get("scope") or ""
        self.scopes = set(scope_str.split()) if scope_str else set()
        return token

    def model_allowed(self, model: str) -> bool:
        """Check whether the current token permits access to a model."""

        required = self._model_prefs.get(model)
        if not required:
            return True
        return bool(self.scopes.intersection(required))
