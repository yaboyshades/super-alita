"""Feature flag adapter for the consolidation engine."""

from __future__ import annotations

import os

from domain.consolidation_engine.service import ConsolidationFeatureFlagProvider


class EnvironmentFeatureFlagProvider(ConsolidationFeatureFlagProvider):
    """Simple environment-backed feature flag provider."""

    def __init__(self, *, overrides: dict[str, bool] | None = None) -> None:
        self._overrides = overrides or {}

    def is_enabled(self, key: str, default: bool = False) -> bool:
        if key in self._overrides:
            return bool(self._overrides[key])
        env_key = key.replace(".", "_").upper()
        raw: str | None = os.getenv(env_key)
        if raw is None:
            return default
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "on", "yes"}:
            return True
        if normalized in {"0", "false", "off", "no"}:
            return False
        return default

    def set_override(self, key: str, value: bool) -> None:
        """Allow tests to set deterministic overrides."""

        self._overrides[key] = bool(value)

    def clear_override(self, key: str) -> None:
        """Remove an override returning control to environment lookup."""

        self._overrides.pop(key, None)
