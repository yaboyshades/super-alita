"""Guidance persistence helpers for SDD FeatureSession."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.yaml_utils import safe_dump, safe_load

from .models import NextStepGuidance


@dataclass(slots=True)
class ArtifactRecord:
    """Lightweight view of an artifact on disk."""

    path: Path
    content: str


class GuidanceRepository:
    """Persistence facade for artifacts and guidance files."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root.resolve()
        self.specs_dir = self.workspace_root / "specs"

    def find_feature_dir(self, feature_id: str) -> Path | None:
        """Locate a feature directory matching the prefix."""
        if not self.specs_dir.exists():
            return None
        for child in self.specs_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith(feature_id):
                return child
        return None

    def load_artifact(self, path: Path) -> ArtifactRecord:
        data = path.read_text(encoding="utf-8")
        return ArtifactRecord(path=path, content=data)

    def load_guidance(self, feature_dir: Path) -> NextStepGuidance | None:
        guidance_path = feature_dir / "next_steps.yaml"
        if not guidance_path.exists():
            return None
        raw = safe_load(guidance_path.read_text(encoding="utf-8")) or {}
        try:
            return NextStepGuidance(**raw)
        except Exception:  # pragma: no cover - guard against legacy content
            return None

    def save_guidance(
        self, feature_dir: Path, guidance: NextStepGuidance
    ) -> Path:
        guidance_path = feature_dir / "next_steps.yaml"
        guidance_path.write_text(
            safe_dump(guidance.model_dump()), encoding="utf-8"
        )
        return guidance_path

    def ensure_feature_dir(self, feature_id: str, slug: str) -> Path:
        feature_dir = self.specs_dir / f"{feature_id}-{slug}"
        feature_dir.mkdir(parents=True, exist_ok=True)
        return feature_dir

    def artifact_path(self, feature_dir: Path, name: str) -> Path:
        return feature_dir / name
