from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatModeConfig(BaseModel):
    """
    Schema for a Copilot custom mode. Supports both `applyTo` (canonical)
    and `apply_to` (alias).
    """

    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, str_strip_whitespace=True
    )

    description: str = Field(..., max_length=512)
    instructions: str
    # Accept arbitrary tool identifiers to preserve compatibility with
    # existing modes in this repository.
    tools: list[str] = Field(default_factory=list)
    applyTo: list[str] = Field(default_factory=list, alias="apply_to")
    enabled: bool = True

    @field_validator("tools", mode="before")
    @classmethod
    def normalize_tools(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)


class EnvironmentSettings(BaseSettings):
    """
    Per-environment overrides (shallow or nested). Example:
      DEV_CONFIG__tools__0="terminal"  (env var for nested override)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )
    dev_config: dict[str, Any] = Field(default_factory=dict)
    staging_config: dict[str, Any] = Field(default_factory=dict)
    prod_config: dict[str, Any] = Field(default_factory=dict)

    def for_env(self, env: str) -> dict[str, Any]:
        mapping = {
            "dev": self.dev_config,
            "staging": self.staging_config,
            "prod": self.prod_config,
        }
        return mapping.get(env, {})


def deep_merge(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out
