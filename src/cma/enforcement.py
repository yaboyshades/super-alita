from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, TypedDict

from src.constitutional import ConstitutionalScorer
from src.core.yaml_utils import safe_load


class EnforcementReport(TypedDict, total=False):
    ok: bool
    reasons: list[str]
    warnings: list[str]
    constitutional_score: float
    article_scores: dict[str, float]
    metadata: dict[str, Any]


@dataclass
class CMAConfig:
    """CMA enforcement configuration."""

    min_constitutional_score: float = 0.75
    require_phases: tuple[str, ...] = (
        "alignment_and_scoping",  # Phase 0
        "blueprint_finalization",  # Phase 1
    )
    enabled: bool = True


class CMAEnforcementError(RuntimeError):
    pass


class CMAEnforcer:
    """Validates CMA workflow phases and guardrails before generation.

    This is a pure checker (no side effects). It validates:
    - Blueprint presence and structure (mandated components)
    - Phase completion for Phase 0 and Phase 1
    - Constitutional guardrails (mapped to Article I–VI)
    - Creative guardrails presence in blueprint (persona, sensory, arc, etc.)
    """

    def __init__(self, config: CMAConfig | None = None) -> None:
        self.config = config or CMAConfig()
        self.scorer = ConstitutionalScorer()

    def validate_pre_generation(
        self,
        *,
        blueprint_text: str,
        phases_completed: list[str] | None = None,
    ) -> EnforcementReport:
        reasons: list[str] = []
        warnings: list[str] = []

        # Phase enforcement
        phases_completed = [p.strip().lower() for p in (phases_completed or [])]
        for req in self.config.require_phases:
            if req.lower() not in phases_completed:
                reasons.append(
                    f"Phase missing: '{req}' must be completed before generation"
                )

        # Blueprint presence/shape
        bp_obj = self._parse_blueprint(blueprint_text, reasons)
        if bp_obj is None:
            return EnforcementReport(
                ok=False,
                reasons=reasons or ["Invalid or missing blueprint YAML"],
                warnings=warnings,
                constitutional_score=0.0,
                article_scores={},
                metadata={"parsed": False},
            )

        # Creative guardrails presence checks (structure-level, not semantics)
        self._verify_creative_components(bp_obj, reasons, warnings)

        # Map to constitutional guardrails via spec scoring
        score_result = self.scorer.score_specification(blueprint_text)
        if score_result.overall_score < self.config.min_constitutional_score:
            reasons.append(
                f"Constitutional score {score_result.overall_score:.2f} < minimum "
                f"{self.config.min_constitutional_score:.2f}"
            )

        return EnforcementReport(
            ok=not reasons,
            reasons=reasons,
            warnings=warnings,
            constitutional_score=score_result.overall_score,
            article_scores=score_result.article_scores,
            metadata={
                "total_violations": len(score_result.violations),
                "phases_checked": list(self.config.require_phases),
                "phases_completed": phases_completed,
            },
        )

    def enforce_or_raise(
        self,
        *,
        blueprint_text: str,
        phases_completed: list[str] | None = None,
    ) -> None:
        report = self.validate_pre_generation(
            blueprint_text=blueprint_text, phases_completed=phases_completed
        )
        if not report["ok"]:
            parts = ["CMA enforcement failed:"] + [f"- {r}" for r in report["reasons"]]
            raise CMAEnforcementError("\n".join(parts))

    # ------------------------
    # Internal helpers
    # ------------------------
    def _parse_blueprint(self, text: str, reasons: list[str]) -> dict[str, Any] | None:
        try:
            obj = safe_load(text)
            if not isinstance(obj, dict):
                reasons.append("Blueprint is not a YAML mapping (dict)")
                return None
            return obj
        except Exception as e:  # noqa: BLE001
            reasons.append(f"YAML parse error: {type(e).__name__}: {e}")
            return None

    def _verify_creative_components(
        self, bp: dict[str, Any], reasons: list[str], warnings: list[str]
    ) -> None:
        # Accept snake_case or Title Case variants
        def has_any(*keys: str) -> bool:
            lowered = {k.lower(): k for k in bp}
            return any(k.lower() in lowered for k in keys)

        def ensure(key_group: tuple[str, ...], label: str) -> None:
            if not has_any(*key_group):
                reasons.append(f"Blueprint missing required component: {label}")

        ensure(("persona_profile", "Persona Profile"), "Persona Profile")
        ensure(
            ("narrative_state_machine", "Narrative State Machine"),
            "Narrative State Machine",
        )
        ensure(("sensory_palette", "Sensory Palette"), "Sensory Palette")
        ensure(("thematic_blueprint", "Thematic Blueprint"), "Thematic Blueprint")

        # Library-First signal: expect explicit library/template usage record
        if not has_any(
            "master_template_library",
            "template_library",
            "master_template_usage",
            "library_usage",
        ):
            warnings.append(
                "No explicit Master Template Library usage recorded (Library-First)"
            )

        # Test-Driven Generation signal
        if not has_any("tests", "test_cases", "acceptance_criteria"):
            warnings.append(
                "No tests/acceptance criteria present (Test-Driven Generation)"
            )


def load_blueprint_from_env() -> tuple[str | None, list[str]]:
    """Helper to load blueprint text and phases from environment.

    - CMA_BLUEPRINT_PATH: file path to YAML blueprint
    - CMA_BLUEPRINT: inline YAML string
    - CMA_PHASES_COMPLETED: CSV of phases
    """
    phases_csv = os.getenv("CMA_PHASES_COMPLETED", "")
    phases = [p.strip() for p in phases_csv.split(",") if p.strip()]

    bp_path = os.getenv("CMA_BLUEPRINT_PATH")
    if bp_path and os.path.exists(bp_path):
        try:
            with open(bp_path, encoding="utf-8") as f:
                return f.read(), phases
        except Exception:
            return None, phases

    inline = os.getenv("CMA_BLUEPRINT")
    if inline:
        return inline, phases

    return None, phases

