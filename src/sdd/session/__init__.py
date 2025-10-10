"""Feature Session layer for unified SDD pipeline access.

This module provides a unified interface for SDD operations that maintains
consistent guidance, artifacts, and constitutional status across all consumers
(CLI, orchestrator, tests).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.yaml_utils import safe_dump, safe_load

from ..constitutional_pipeline import ConstitutionalSDDPipeline
from ..models import (
    NextStepGuidance,
    PlanRequest,
    SpecifyRequest,
    TasksRequest,
)


@dataclass
class ArtifactResult:
    """Result of a session phase operation containing artifacts
    and guidance.
    """

    feature_id: str
    artifact_path: str
    artifact_content: str
    guidance: NextStepGuidance
    constitutional_compliance: dict[str, Any]
    overall_compliance_score: float
    compliance_threshold_met: bool
    next_steps: list[str]
    metadata_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Serialize constitutional compliance
        compliance_dict = {}
        for key, value in self.constitutional_compliance.items():
            if hasattr(value, "model_dump"):
                compliance_dict[key] = value.model_dump(mode="json")
            else:
                compliance_dict[key] = value

        return {
            "feature_id": self.feature_id,
            "artifact_path": self.artifact_path,
            "artifact_content": self.artifact_content,
            "guidance": (
                self.guidance.model_dump(mode="json")
                if self.guidance
                else None
            ),
            "constitutional_compliance": compliance_dict,
            "overall_compliance_score": self.overall_compliance_score,
            "compliance_threshold_met": self.compliance_threshold_met,
            "next_steps": self.next_steps,
            "metadata_path": self.metadata_path,
        }


class GuidanceRepository:
    """Handles persistence of guidance and artifacts."""

    def __init__(self, workspace_root: Path):
        """Initialize repository with workspace root."""
        self.workspace_root = workspace_root
        self.specs_dir = workspace_root / "specs"

    def load_guidance(self, feature_dir: Path) -> NextStepGuidance | None:
        """Load guidance from feature directory."""
        metadata_path = feature_dir / "next_steps.yaml"
        if not metadata_path.exists():
            return None

        try:
            raw_data = safe_load(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if not raw_data:
            return None

        if "feature_id" not in raw_data:
            # Derive from directory name
            feature_id = self._derive_feature_id_from_path(feature_dir)
            raw_data["feature_id"] = feature_id

        try:
            return NextStepGuidance(**raw_data)
        except Exception:
            return None

    def save_guidance(
        self, feature_dir: Path, guidance: NextStepGuidance
    ) -> str:
        """Save guidance to feature directory and return relative path."""
        metadata_path = feature_dir / "next_steps.yaml"
        metadata_path.write_text(
            safe_dump(guidance.model_dump()), encoding="utf-8"
        )
        return str(metadata_path.relative_to(self.workspace_root))

    def load_artifact(
        self, feature_dir: Path, artifact_name: str
    ) -> str | None:
        """Load artifact content from feature directory."""
        artifact_path = feature_dir / artifact_name
        if not artifact_path.exists():
            return None
        return artifact_path.read_text(encoding="utf-8")

    def save_artifact(
        self, feature_dir: Path, artifact_name: str, content: str
    ) -> str:
        """Save artifact to feature directory and return relative path."""
        artifact_path = feature_dir / artifact_name
        artifact_path.write_text(content, encoding="utf-8")
        return str(artifact_path.relative_to(self.workspace_root))

    def _derive_feature_id_from_path(self, feature_dir: Path) -> str:
        """Derive feature ID from directory path."""
        dir_name = feature_dir.name
        if len(dir_name) >= 3 and dir_name[:3].isdigit():
            return dir_name[:3]
        return dir_name.split("-", 1)[0] if "-" in dir_name else "unknown"


class FeatureSession:
    """Unified session for SDD operations with consistent guidance
    and artifacts.
    """

    def __init__(
        self,
        feature_id: str,
        feature_dir: Path,
        pipeline: ConstitutionalSDDPipeline,
        repository: GuidanceRepository,
    ):
        """Initialize session with feature context."""
        self.feature_id = feature_id
        self.feature_dir = feature_dir
        self.pipeline = pipeline
        self.repository = repository

        # Load existing state
        self.guidance = self.repository.load_guidance(feature_dir)
        self._cached_artifacts: dict[str, str] = {}

    async def specify(self, request: SpecifyRequest) -> ArtifactResult:
        """Execute specify phase and return unified result."""
        # Run pipeline
        response = await self.pipeline.specify(request)

        # Update guidance if it exists
        if response.next_step_guidance:
            self.guidance = response.next_step_guidance
            metadata_path = self.repository.save_guidance(
                self.feature_dir, self.guidance
            )
        else:
            metadata_path = None

        # Cache the artifact
        self._cached_artifacts["spec.md"] = response.specification

        return ArtifactResult(
            feature_id=self.feature_id,
            artifact_path=response.spec_file_path or response.feature_path,
            artifact_content=response.specification,
            guidance=self.guidance,
            constitutional_compliance=response.constitutional_compliance,
            overall_compliance_score=response.overall_compliance_score,
            compliance_threshold_met=response.compliance_threshold_met,
            next_steps=response.next_steps,
            metadata_path=metadata_path,
        )

    async def plan(self, request: PlanRequest) -> ArtifactResult:
        """Execute plan phase and return unified result."""
        # Run pipeline
        response = await self.pipeline.plan(request)

        # Update guidance if it exists
        if response.next_step_guidance:
            self.guidance = response.next_step_guidance
            metadata_path = self.repository.save_guidance(
                self.feature_dir, self.guidance
            )
        else:
            metadata_path = None

        # Cache the artifact
        self._cached_artifacts["implementation-plan.md"] = (
            response.implementation_plan
        )

        return ArtifactResult(
            feature_id=self.feature_id,
            artifact_path=response.plan_path,
            artifact_content=response.implementation_plan,
            guidance=self.guidance,
            constitutional_compliance=response.constitutional_compliance,
            overall_compliance_score=response.overall_compliance_score,
            compliance_threshold_met=response.compliance_threshold_met,
            next_steps=response.next_steps,
            metadata_path=metadata_path,
        )

    async def tasks(self, request: TasksRequest) -> ArtifactResult:
        """Execute tasks phase and return unified result."""
        # Run pipeline
        response = await self.pipeline.tasks(request)

        # Update guidance if it exists
        if response.next_step_guidance:
            self.guidance = response.next_step_guidance
            metadata_path = self.repository.save_guidance(
                self.feature_dir, self.guidance
            )
        else:
            metadata_path = None

        # Cache the artifact
        self._cached_artifacts["tasks.md"] = response.tasks_breakdown

        return ArtifactResult(
            feature_id=self.feature_id,
            artifact_path=response.tasks_path,
            artifact_content=response.tasks_breakdown,
            guidance=self.guidance,
            constitutional_compliance=response.constitutional_compliance,
            overall_compliance_score=response.overall_compliance_score,
            compliance_threshold_met=response.compliance_threshold_met,
            next_steps=response.next_steps,
            metadata_path=metadata_path,
        )

    def get_guidance_summary(self) -> dict[str, Any]:
        """Get summary of current guidance state."""
        if not self.guidance:
            return {"status": "no_guidance"}

        return {
            "clarifications": len(self.guidance.clarifications),
            "artefacts": len(self.guidance.artefacts),
            "commands": len(self.guidance.commands),
            "constitutional_alignment": len(
                self.guidance.constitutional_alignment
            ),
        }

    def get_cached_artifact(self, name: str) -> str | None:
        """Get cached artifact content."""
        return self._cached_artifacts.get(name)
