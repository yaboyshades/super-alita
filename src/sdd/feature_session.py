"""FeatureSession abstraction for the SDD pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constitutional_pipeline import ConstitutionalSDDPipeline
from .guidance_repository import ArtifactRecord, GuidanceRepository
from .models import (
    NextStepGuidance,
    PlanRequest,
    PlanResponse,
    SpecifyRequest,
    SpecifyResponse,
    TasksRequest,
    TasksResponse,
)


@dataclass(slots=True)
class SessionArtifactResult:
    """Return payload for a session phase invocation."""

    phase: str
    artifact: ArtifactRecord
    response: SpecifyResponse | PlanResponse | TasksResponse
    guidance: NextStepGuidance | None
    guidance_path: Path | None

    @property
    def feature_id(self) -> str | None:
        return getattr(self.response, "feature_id", None)

    @property
    def artifact_path(self) -> str:
        return str(self.artifact.path)

    @property
    def next_steps(self) -> list[str]:
        return getattr(self.response, "next_steps", []) or []

    @property
    def overall_compliance_score(self) -> float:
        return getattr(self.response, "overall_compliance_score", 0.0)

    @property
    def compliance_threshold_met(self) -> bool:
        return getattr(self.response, "compliance_threshold_met", True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "artifact_path": self.artifact_path,
            "guidance_path": str(self.guidance_path) if self.guidance_path else None,
            "response": self.response.model_dump(mode="python"),
            "guidance": self.guidance.model_dump(mode="python") if self.guidance else None,
        }


class FeatureSession:
    """Unified entrypoint for SDD specify/plan/tasks phases."""

    def __init__(
        self,
        pipeline: ConstitutionalSDDPipeline,
        repository: GuidanceRepository,
        *,
        workspace_root: Path,
        feature_id: str | None = None,
        feature_dir: Path | None = None,
        guidance: NextStepGuidance | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._repo = repository
        self.workspace_root = workspace_root
        self.feature_id = feature_id
        self.feature_dir = feature_dir
        self.guidance = guidance
        self._spec_path: Path | None = None
        self._plan_path: Path | None = None

    async def specify(
        self, description: str, *, context: dict[str, Any] | None = None
    ) -> SessionArtifactResult:
        request = SpecifyRequest(user_input=description, context=context or {})
        response = await self._pipeline.specify(request)
        feature_dir = Path(response.feature_dir or Path(response.feature_path).parent)
        self.feature_id = response.feature_id
        self.feature_dir = feature_dir
        self.guidance = response.next_step_guidance
        guidance_path = None
        if response.next_step_guidance:
            guidance_path = self._repo.save_guidance(feature_dir, response.next_step_guidance)
        spec_path = Path(response.spec_file_path or response.feature_path)
        self._spec_path = spec_path
        artifact = self._repo.load_artifact(spec_path)
        return SessionArtifactResult(
            phase="specify",
            artifact=artifact,
            response=response,
            guidance=response.next_step_guidance,
            guidance_path=guidance_path,
        )

    async def plan(
        self,
        *,
        technology_stack: str = "",
        constraints: dict[str, Any] | None = None,
    ) -> SessionArtifactResult:
        self._ensure_feature_dir()
        spec_path = self._spec_path or self.feature_dir / "spec.md"
        request = PlanRequest(
            specification_path=str(spec_path),
            technology_stack=technology_stack,
            constraints=constraints or {},
            feature_id=self.feature_id,
        )
        response = await self._pipeline.plan(request)
        self.guidance = response.next_step_guidance or self.guidance
        guidance_path = None
        if response.next_step_guidance and self.feature_dir:
            guidance_path = self._repo.save_guidance(self.feature_dir, response.next_step_guidance)
        plan_path = Path(response.plan_path)
        self._plan_path = plan_path
        artifact = self._repo.load_artifact(plan_path)
        return SessionArtifactResult(
            phase="plan",
            artifact=artifact,
            response=response,
            guidance=response.next_step_guidance,
            guidance_path=guidance_path,
        )

    async def tasks(
        self,
        *,
        priority_focus: str = "test-first",
        team_size: int = 1,
    ) -> SessionArtifactResult:
        self._ensure_feature_dir()
        plan_path = self._plan_path or self.feature_dir / "implementation-plan.md"
        request = TasksRequest(
            plan_path=str(plan_path),
            feature_id=self.feature_id,
            priority_focus=priority_focus,
            team_size=team_size,
        )
        response = await self._pipeline.tasks(request)
        self.guidance = response.next_step_guidance or self.guidance
        guidance_path = None
        if response.next_step_guidance and self.feature_dir:
            guidance_path = self._repo.save_guidance(self.feature_dir, response.next_step_guidance)
        tasks_path = Path(response.tasks_path)
        artifact = self._repo.load_artifact(tasks_path)
        return SessionArtifactResult(
            phase="tasks",
            artifact=artifact,
            response=response,
            guidance=response.next_step_guidance,
            guidance_path=guidance_path,
        )

    def _ensure_feature_dir(self) -> None:
        if not self.feature_dir:
            raise RuntimeError("Feature directory is not set; run specify first or load session.")


class FeatureSessionFactory:
    """Factory for creating FeatureSession instances."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root.resolve()
        self._pipeline = ConstitutionalSDDPipeline(workspace_root=self.workspace_root)
        self._repository = GuidanceRepository(self.workspace_root)

    def create(self) -> FeatureSession:
        return FeatureSession(
            self._pipeline,
            self._repository,
            workspace_root=self.workspace_root,
        )

    def for_description(self, description: str, context: dict[str, object] | None | None = None) -> FeatureSession:
        # Currently identical to create(); reserved for future context-aware initialization.
        return self.create()

    def load(self, feature_id: str) -> FeatureSession:
        feature_dir = self._repository.find_feature_dir(feature_id)
        if feature_dir is None:
            raise FileNotFoundError(f"Feature directory not found for id '{feature_id}'.")
        guidance = self._repository.load_guidance(feature_dir)
        return FeatureSession(
            self._pipeline,
            self._repository,
            workspace_root=self.workspace_root,
            feature_id=feature_id,
            feature_dir=feature_dir,
            guidance=guidance,
        )

    def for_feature_id(self, feature_id: str) -> FeatureSession:
        return self.load(feature_id)

