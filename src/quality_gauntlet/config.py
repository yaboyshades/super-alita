"""Configuration primitives for the Quality Gauntlet pipeline."""

from __future__ import annotations

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

    max_iterations: int = Field(3, ge=1, le=10)
    thresholds: QualityThresholds = Field(default_factory=QualityThresholds)

    class Config:
        validate_assignment = True
        frozen = True
