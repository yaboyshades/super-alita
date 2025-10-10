"""Configuration primitives for the Quality Gauntlet pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, validator  # type: ignore[import-not-found]


class QualityThresholds(BaseModel):
    """Threshold configuration for each verification dimension."""

    security: float = Field(0.90, ge=0.0, le=1.0)
    quality: float = Field(0.85, ge=0.0, le=1.0)
    compliance: float = Field(0.95, ge=0.0, le=1.0)
    constitutional: float = Field(0.75, ge=0.0, le=1.0)

    @validator("security", "quality", "compliance", "constitutional")
    def _validate_precision(cls, value: float) -> float:  # noqa: D401,N805
        """Ensure scores are rounded to two decimals for stability."""

        return round(value, 4)


class GauntletConfig(BaseModel):
    """Top level configuration for orchestrator runs."""

    thresholds: QualityThresholds = Field(default_factory=QualityThresholds)
    max_iterations: int = Field(3, ge=1, le=5)

    constitution_path: Path = Field(default=Path(".github/CONSTITUTION.md"))
    agents_md_path: Path = Field(default=Path("AGENTS.md"))
    project_root: Path = Field(default_factory=Path.cwd)

    enable_snyk: bool = Field(default=True)
    enable_codeql: bool = Field(default=False)
    enable_bandit: bool = Field(default=True)
    enable_ruff: bool = Field(default=True)
    enable_mypy: bool = Field(default=True)

    model: str = Field(default="gpt-4")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=300, ge=30, le=900)
    enable_parallel_analysis: bool = Field(default=False)

    class Config:
        env_prefix = "GAUNTLET_"
        validate_assignment = True
