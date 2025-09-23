"""
Hardened Unified Intelligence Orchestrator

Implements the governed pipeline with:
- Explicit contracts and score fusion
- Failure semantics and graceful degradation
- Telemetry and authority rules
- Canonical orchestration flow
"""

import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Decision(Enum):
    """Final decision outcomes."""

    PROCEED = "proceed"
    REVISE = "revise"
    BLOCK = "block"


@dataclass
class FusionConfig:
    """Configuration for score fusion math."""

    # Base weights (must sum to 1.0)
    mangle_base: float = 0.35
    constitution_base: float = 0.45
    workflow_base: float = 0.20

    # Routing modifiers (added to weights, then renormalized)
    code_task_boost: float = 0.15  # for new_feature, refactor, debug
    constitutional_boost: float = 0.15  # for constitutional workflow

    # Confidence gating
    confidence_floor: float = 0.5

    # Decision thresholds
    proceed_threshold: float = 0.7
    revise_threshold: float = 0.5

    # Hard gates
    constitution_gate: float = 0.4  # below this = cannot proceed


@dataclass
class Request:
    """Standardized request format."""

    request_id: str
    ts: str
    intent_text: str
    code_refs: list[str] | None = None
    context: dict[str, Any] | None = None


@dataclass
class MangleResult:
    """Mangle Bridge result."""

    ok: bool
    facts: list[dict[str, Any]]
    metrics: dict[str, float | None]
    findings: list[dict[str, Any]]
    confidence: float
    errors: list[str]


@dataclass
class ConstitutionResult:
    """Constitutional Engine result."""

    ok: bool
    article_scores: dict[str, float]
    overall: float
    infractions: list[dict[str, Any]]
    confidence: float
    errors: list[str]


@dataclass
class WorkflowResult:
    """Workflow Detector result."""

    label: str
    confidence: float
    features: list[str]
    errors: list[str]


@dataclass
class CopilotEnhancement:
    """Copilot Enhancer result."""

    ok: bool
    templates_applied: list[str]
    guidance: list[dict[str, str]]
    errors: list[str]


@dataclass
class UnifiedAdvice:
    """Final orchestrated advice."""

    ok: bool
    decision: str
    reasons: list[str]
    recommendations: list[dict[str, Any]]
    scores: dict[str, Any]
    telemetry: dict[str, Any]
    errors: list[str]


@dataclass
class CodeAnalysisResult:
    """Code Analysis result from mangle-style reasoning."""

    ok: bool
    repo_path: str
    total_files: int
    total_symbols: int
    findings: dict[str, list[dict[str, Any]]]
    summary: dict[str, int]
    analysis_time: float
    confidence: float
    errors: list[str]


class HardenedOrchestrator:
    """
    Hardened orchestrator with explicit contracts and fusion math.

    Implements the governed pipeline with authority rules and telemetry.
    """

    def __init__(self, config: FusionConfig | None = None):
        """Initialize with fusion configuration."""
        self.config = config or FusionConfig()
        self._validate_config()

    def _validate_config(self):
        """Validate fusion configuration."""
        base_sum = (
            self.config.mangle_base
            + self.config.constitution_base
            + self.config.workflow_base
        )
        if abs(base_sum - 1.0) > 0.001:
            raise ValueError(f"Base weights must sum to 1.0, got {base_sum}")

    async def orchestrate(self, request: Request) -> UnifiedAdvice:
        """
        Execute the canonical orchestration flow.

        Request -> WorkflowDetector ->
        [MangleBridge, ConstitutionalEngine, CodeAnalysis] (parallel) ->
        CopilotEnhancer -> Fuse -> UnifiedAdvice
        """
        start_time = time.time()
        timings = {}
        components_used = []
        all_errors = []

        try:
            # Step 1: Detect workflow pattern
            workflow_start = time.time()
            workflow_result = await self._detect_workflow(request)
            timings["workflow"] = (time.time() - workflow_start) * 1000
            components_used.append("workflow")

            if workflow_result.errors:
                all_errors.extend(workflow_result.errors)

            # Step 2: Parallel execution of Mangle, Constitution, and Code Analysis
            mangle_task = self._analyze_mangle(request)
            constitution_task = self._analyze_constitution(request)
            code_analysis_task = self._analyze_code(request)

            mangle_start = time.time()
            constitution_start = time.time()
            code_start = time.time()

            mangle_result, constitution_result, code_result = (
                await self._gather_parallel(
                    mangle_task, constitution_task, code_analysis_task
                )
            )

            timings["mangle"] = (time.time() - mangle_start) * 1000
            timings["constitution"] = (time.time() - constitution_start) * 1000
            timings["code_analysis"] = (time.time() - code_start) * 1000

            if mangle_result.ok:
                components_used.append("mangle")
            if constitution_result.ok:
                components_used.append("constitution")
            if code_result.ok:
                components_used.append("code_analysis")

            all_errors.extend(mangle_result.errors)
            all_errors.extend(constitution_result.errors)
            all_errors.extend(code_result.errors)

            # Step 3: Copilot enhancement (after workflow)
            copilot_start = time.time()
            copilot_result = await self._enhance_copilot(request, workflow_result)
            timings["copilot"] = (time.time() - copilot_start) * 1000
            components_used.append("copilot")

            if copilot_result.errors:
                all_errors.extend(copilot_result.errors)

            # Step 4: Fuse scores and make decision
            advice = self._fuse_and_decide(
                request,
                workflow_result,
                mangle_result,
                constitution_result,
                code_result,
                copilot_result,
            )

            # Step 5: Add telemetry
            total_time = (time.time() - start_time) * 1000
            timings["total"] = total_time

            advice.telemetry = {
                "request_id": request.request_id,
                "components": components_used,
                "timings_ms": timings,
                "version": "1",
            }

            advice.errors = all_errors
            advice.ok = len(all_errors) == 0

            return advice

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return UnifiedAdvice(
                ok=False,
                decision=Decision.BLOCK.value,
                reasons=["Orchestration failure"],
                recommendations=[],
                scores={
                    "fused": 0.0,
                    "contributors": {
                        "mangle": 0.0,
                        "constitution": 0.0,
                        "code_analysis": 0.0,
                        "workflow": 0.0,
                    },
                },
                telemetry={"request_id": request.request_id, "error": str(e)},
                errors=[str(e)],
            )

    async def _detect_workflow(self, request: Request) -> WorkflowResult:
        """Detect workflow pattern."""
        try:
            # Import here to avoid circular dependencies
            from .workflow_detector import WorkflowDetector

            detector = WorkflowDetector()
            detection = detector.detect_with_confidence(request.intent_text)

            return WorkflowResult(
                label=detection["pattern"],
                confidence=detection["confidence"],
                features=detection["matches"],
                errors=[],
            )
        except Exception as e:
            logger.warning(f"Workflow detection failed: {e}")
            return WorkflowResult(
                label="general", confidence=0.0, features=[], errors=[str(e)]
            )

    async def _analyze_mangle(self, request: Request) -> MangleResult:
        """Analyze with Mangle Bridge."""
        try:
            from .mangle_bridge import MangleBridge

            bridge = MangleBridge()
            insights = bridge.generate_code_insights(request.intent_text)

            if insights.get("available", False):
                return MangleResult(
                    ok=True,
                    facts=insights.get("facts", []),
                    metrics={
                        "complexity": None,  # TODO: extract from facts
                        "coverage_gap": None,
                    },
                    findings=[],  # TODO: extract issues
                    confidence=0.8,  # TODO: calculate based on facts
                    errors=[],
                )
            else:
                return MangleResult(
                    ok=False,
                    facts=[],
                    metrics={"complexity": None, "coverage_gap": None},
                    findings=[],
                    confidence=0.0,
                    errors=["Mangle not available"],
                )
        except Exception as e:
            logger.warning(f"Mangle analysis failed: {e}")
            return MangleResult(
                ok=False,
                facts=[],
                metrics={"complexity": None, "coverage_gap": None},
                findings=[],
                confidence=0.0,
                errors=[str(e)],
            )

    async def _analyze_constitution(self, request: Request) -> ConstitutionResult:
        """Analyze constitutional compliance."""
        try:
            from .constitutional_engine import ConstitutionalEngine

            engine = ConstitutionalEngine()
            analysis = engine.analyze_compliance(request.intent_text)

            return ConstitutionResult(
                ok=True,
                article_scores=analysis.get("article_scores", {}),
                overall=analysis.get("overall_score", 0.0),
                infractions=analysis.get("violations", []),
                confidence=0.8,  # TODO: calculate based on analysis
                errors=[],
            )
        except Exception as e:
            logger.warning(f"Constitutional analysis failed: {e}")
            return ConstitutionResult(
                ok=False,
                article_scores={},
                overall=0.0,
                infractions=[],
                confidence=0.0,
                errors=[str(e)],
            )

    async def _enhance_copilot(
        self, request: Request, workflow: WorkflowResult
    ) -> CopilotEnhancement:
        """Enhance with Copilot guidance."""
        try:
            from .copilot_enhancer import CopilotEnhancer

            enhancer = CopilotEnhancer()
            enhancement = enhancer.enhance_response(request.intent_text)

            return CopilotEnhancement(
                ok=True,
                templates_applied=[],  # TODO: extract from enhancement
                guidance=enhancement.get("constitutional_notes", []),
                errors=[],
            )
        except Exception as e:
            logger.warning(f"Copilot enhancement failed: {e}")
            return CopilotEnhancement(
                ok=False, templates_applied=[], guidance=[], errors=[str(e)]
            )

    async def _gather_parallel(self, *tasks):
        """Gather parallel tasks (simplified for now)."""
        # TODO: Implement proper async gathering
        results = []
        for task in tasks:
            if hasattr(task, "__await__"):
                results.append(await task)
            else:
                results.append(task)
        return results

    async def _analyze_code(self, request: Request) -> CodeAnalysisResult:
        """Analyze code using mangle-style reasoning."""
        try:
            import os
            import tempfile

            from .code_reasoning import CodeIngester, RuleEngine

            # Determine repo path from request
            repo_path = "."
            if request.code_refs:
                # Use the first code ref as repo path if it's a directory
                for ref in request.code_refs:
                    if os.path.isdir(ref):
                        repo_path = ref
                        break

            # Create temporary database for analysis
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                db_path = tmp.name

            try:
                # Ingest code
                ingester = CodeIngester(db_path)
                ingest_stats = ingester.ingest_repository(repo_path)

                # Run rules
                rule_engine = RuleEngine(db_path)
                findings = rule_engine.run_all_rules()
                summary = rule_engine.get_summary(findings)

                # Convert findings to dict format
                findings_dict = {}
                for rule_name, finding_list in findings.items():
                    findings_dict[rule_name] = [asdict(f) for f in finding_list]

                # Calculate confidence based on findings
                total_findings = sum(summary.values())
                confidence = min(
                    0.9, 0.5 + (total_findings * 0.1)
                )  # Base 0.5, +0.1 per finding

                return CodeAnalysisResult(
                    ok=True,
                    repo_path=repo_path,
                    total_files=ingest_stats["files_processed"],
                    total_symbols=ingest_stats["symbols_extracted"],
                    findings=findings_dict,
                    summary=summary,
                    analysis_time=0.0,  # TODO: track timing
                    confidence=confidence,
                    errors=[],
                )

            finally:
                # Clean up temporary database
                try:
                    os.unlink(db_path)
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Code analysis failed: {e}")
            return CodeAnalysisResult(
                ok=False,
                repo_path=".",
                total_files=0,
                total_symbols=0,
                findings={},
                summary={},
                analysis_time=0.0,
                confidence=0.0,
                errors=[str(e)],
            )

    def _fuse_and_decide(
        self,
        request: Request,
        workflow: WorkflowResult,
        mangle: MangleResult,
        constitution: ConstitutionResult,
        code_analysis: CodeAnalysisResult,
        copilot: CopilotEnhancement,
    ) -> UnifiedAdvice:
        """Fuse scores and make final decision."""
        # Calculate adjusted weights based on workflow routing
        weights = self._calculate_weights(workflow)

        # Apply confidence gating
        adj_mangle = mangle.confidence * (
            self.config.confidence_floor
            + (1 - self.config.confidence_floor) * mangle.confidence
        )
        adj_const = constitution.confidence * (
            self.config.confidence_floor
            + (1 - self.config.confidence_floor) * constitution.confidence
        )
        adj_code = code_analysis.confidence * (
            self.config.confidence_floor
            + (1 - self.config.confidence_floor) * code_analysis.confidence
        )
        adj_work = workflow.confidence

        # Fuse scores (adjust weights to include code analysis)
        total_weight = weights["mangle"] + weights["constitution"] + weights["workflow"]
        code_weight = 1.0 - total_weight  # Give remaining weight to code analysis

        fused = (
            weights["mangle"] * adj_mangle
            + weights["constitution"] * adj_const
            + weights["workflow"] * adj_work
            + code_weight * adj_code
        )

        # Make decision with authority rules
        decision = self._make_decision(fused, constitution, workflow, code_analysis)

        # Generate reasons and recommendations
        reasons = self._generate_reasons(
            workflow, mangle, constitution, code_analysis, copilot
        )
        recommendations = self._generate_recommendations(
            workflow, mangle, constitution, code_analysis, copilot
        )
        return UnifiedAdvice(
            ok=True,
            decision=decision.value,
            reasons=reasons,
            recommendations=recommendations,
            scores={
                "fused": fused,
                "contributors": {
                    "mangle": adj_mangle,
                    "constitution": adj_const,
                    "code_analysis": adj_code,
                    "workflow": adj_work,
                },
            },
            telemetry={},  # Will be filled by caller
            errors=[],
        )

    def _calculate_weights(self, workflow: WorkflowResult) -> dict[str, float]:
        """Calculate adjusted weights based on workflow routing."""
        weights = {
            "mangle": self.config.mangle_base,
            "constitution": self.config.constitution_base,
            "workflow": self.config.workflow_base,
        }
        # Apply routing modifiers
        code_tasks = {"new_feature", "refactor", "debug"}
        if workflow.label in code_tasks:
            weights["mangle"] += self.config.code_task_boost
        if workflow.label in code_tasks:
            weights["constitution"] += self.config.constitutional_boost
        if workflow.label == "constitutional":
            weights["constitution"] += self.config.constitutional_boost

        # Renormalize to sum = 1.0
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total

        return weights

    def _make_decision(
        self,
        fused: float,
        constitution: ConstitutionResult,
        workflow: WorkflowResult,
        code_analysis: CodeAnalysisResult,
    ) -> Decision:
        """Make final decision with authority rules."""
        if (
            constitution.ok
            and constitution.overall < self.config.constitution_gate
            and constitution.confidence > 0.6
        ):
            return Decision.REVISE

        # Check for critical code analysis findings
        if code_analysis.ok and code_analysis.summary.get("cycle", 0) > 0:
            # Circular dependencies are a hard block
            return Decision.REVISE

        # Standard thresholds
        if fused >= self.config.proceed_threshold:
            return Decision.PROCEED
        elif fused >= self.config.revise_threshold:
            return Decision.REVISE
        else:
            return Decision.BLOCK

    def _generate_reasons(
        self,
        workflow: WorkflowResult,
        mangle: MangleResult,
        constitution: ConstitutionResult,
        code_analysis: CodeAnalysisResult,
        copilot: CopilotEnhancement,
    ) -> list[str]:
        """Generate human-readable reasons for the decision."""
        reasons = []

        if constitution.ok and constitution.infractions:
            reasons.append(
                f"Constitutional infractions: {len(constitution.infractions)} issues"
            )

        if mangle.ok and mangle.findings:
            reasons.append(f"Code analysis found {len(mangle.findings)} issues")

        if code_analysis.ok:
            total_findings = sum(code_analysis.summary.values())
            if total_findings > 0:
                reasons.append(f"Code reasoning found {total_findings} issues")
                # Add specific high-priority findings
                if code_analysis.summary.get("cycle", 0) > 0:
                    reasons.append("Circular dependencies detected")
                if code_analysis.summary.get("untested_function", 0) > 0:
                    reasons.append("Untested complex functions found")

        if workflow.confidence < 0.5:
            reasons.append("Low confidence in workflow detection")

        if not reasons:
            reasons.append("Analysis completed successfully")

        return reasons

    def _generate_recommendations(
        self,
        workflow: WorkflowResult,
        mangle: MangleResult,
        constitution: ConstitutionResult,
        code_analysis: CodeAnalysisResult,
        copilot: CopilotEnhancement,
    ) -> list[dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []

        # Constitutional recommendations
        if constitution.ok:
            for infraction in constitution.infractions[:3]:  # Top 3
                recommendations.append(
                    {
                        "action": f"Address {infraction.get('article', 'unknown')} "
                        "compliance",
                        "rationale": infraction.get(
                            "note", "Improve constitutional compliance"
                        ),
                        "refs": [],
                    }
                )

        # Mangle-based recommendations
        if mangle.ok:
            for finding in mangle.findings[:2]:  # Top 2
                recommendations.append(
                    {
                        "action": f"Fix {finding.get('severity', 'medium')} "
                        "priority issue",
                        "rationale": finding.get("note", "Improve code quality"),
                        "refs": [],
                    }
                )

        # Code analysis recommendations
        if code_analysis.ok:
            # Prioritize critical findings
            if code_analysis.summary.get("cycle", 0) > 0:
                recommendations.append(
                    {
                        "action": "Resolve circular dependencies",
                        "rationale": "Circular imports create maintenance issues",
                        "refs": [],
                    }
                )

            if code_analysis.summary.get("untested_function", 0) > 0:
                recommendations.append(
                    {
                        "action": "Add tests for complex functions",
                        "rationale": "Untested complex code increases risk",
                        "refs": [],
                    }
                )

            if code_analysis.summary.get("hot_path", 0) > 0:
                recommendations.append(
                    {
                        "action": "Review hot path functions",
                        "rationale": "Complex functions called frequently need attention",
                        "refs": [],
                    }
                )

        # Workflow-specific recommendations
        if workflow.label == "new_feature":
            recommendations.append(
                {
                    "action": "Use SDD feature specification template",
                    "rationale": "Ensure comprehensive requirements definition",
                    "refs": ["templates/sdd/feature-spec-template.md"],
                }
            )

        return recommendations


# Convenience function for easy usage
async def orchestrate_request(request_data: dict[str, Any]) -> dict[str, Any]:
    """Convenience function to orchestrate a request from dict."""
    request = Request(**request_data)
    orchestrator = HardenedOrchestrator()
    advice = await orchestrator.orchestrate(request)
    return asdict(advice)
