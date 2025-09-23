"""Feature Session Factory for creating and managing SDD sessions."""

from pathlib import Path

from ..constitutional_pipeline import ConstitutionalSDDPipeline
from . import FeatureSession, GuidanceRepository


class FeatureSessionFactory:
    """Factory for creating FeatureSession instances."""

    def __init__(self, workspace_root: Path | None = None):
        """Initialize factory with workspace root."""
        self.workspace_root = workspace_root or Path.cwd()
        self.pipeline = ConstitutionalSDDPipeline(self.workspace_root)
        self.repository = GuidanceRepository(self.workspace_root)

    def for_feature_id(self, feature_id: str) -> FeatureSession:
        """Create session for existing feature by ID."""
        # Find feature directory
        specs_dir = self.workspace_root / "specs"
        feature_dir = None

        # Look for directory starting with feature_id
        for d in specs_dir.iterdir():
            if d.is_dir() and d.name.startswith(f"{feature_id:03d}"):
                feature_dir = d
                break

        if not feature_dir:
            # Create new directory if it doesn't exist
            feature_dir = specs_dir / f"{feature_id:03d}-unknown"
            feature_dir.mkdir(parents=True, exist_ok=True)

        return FeatureSession(feature_id, feature_dir, self.pipeline, self.repository)

    def for_description(
        self, description: str, context: dict | None = None
    ) -> FeatureSession:
        """Create session for new feature from description.

        Args:
            description: Feature description
            context: Additional context (reserved for future use)
        """
        # Generate feature ID using pipeline logic
        specs_dir = self.workspace_root / "specs"
        existing_features = [
            d for d in specs_dir.iterdir() if d.is_dir() and d.name[:3].isdigit()
        ]
        next_num = len(existing_features) + 1
        feature_id = f"{next_num:03d}"

        # Simple slugify
        import re

        feature_name = re.sub(r"[^\w\s-]", "", description.lower())
        feature_name = re.sub(r"[-\s]+", "-", feature_name).strip("-")[:50]

        feature_dir = specs_dir / f"{feature_id}-{feature_name}"
        feature_dir.mkdir(parents=True, exist_ok=True)

        return FeatureSession(feature_id, feature_dir, self.pipeline, self.repository)
