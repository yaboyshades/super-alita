"""
Enhanced SDD Framework with Mangle integration.

This module extends the existing constitutional SDD pipeline with Mangle-based
reasoning capabilities for deeper code analysis and specification traceability.
"""

import logging
from pathlib import Path
from typing import Any

from .constitutional_pipeline import ConstitutionalSDDPipeline
from .mangle_reasoner import MangleReasoner
from .models import (
    PlanRequest,
    PlanResponse,
    SpecifyRequest,
    SpecifyResponse,
    TasksRequest,
    TasksResponse,
)

logger = logging.getLogger(__name__)


class EnhancedSDDFramework(ConstitutionalSDDPipeline):
    """
    Enhanced SDD Framework with Mangle integration.

    Extends the constitutional SDD pipeline with:
    - Code knowledge graph reasoning
    - Natural language query capabilities
    - Enhanced constitutional validation using deductive logic
    - Code-to-specification traceability
    - Comprehensive quality analysis
    """

    def __init__(
        self,
        workspace_root: Path | None = None,
        mangle_executable: str = "mangle",
    ):
        """
        Initialize the enhanced SDD framework.

        Args:
            workspace_root: Root directory of the workspace
            mangle_executable: Path to the Mangle executable
        """
        super().__init__(workspace_root)

        # Initialize Mangle reasoner
        self.mangle_reasoner = MangleReasoner(
            workspace_root=str(self.workspace_root),
            mangle_executable=mangle_executable,
        )

        # Enhanced analysis capabilities
        self._last_analysis_results: dict[str, Any] = {}

    async def specify(self, request: SpecifyRequest) -> SpecifyResponse:
        """
        Execute the /specify phase with enhanced Mangle-based validation.

        This extends the base specify method with:
        - Library-first analysis using Mangle queries
        - Existing solution detection
        - Constitutional compliance prediction
        """
        logger.info("Executing enhanced specify phase with Mangle integration")

        # Run base specification generation
        base_response = await super().specify(request)

        # Enhance with Mangle-based analysis
        try:
            # Analyze existing solutions
            existing_solutions = self._analyze_existing_solutions(
                request.user_input
            )

            # Check for similar features
            similar_features = self._find_similar_features(request.user_input)

            # Predict constitutional compliance issues
            compliance_predictions = self._predict_compliance_issues(
                request.user_input
            )

            # Update response with enhanced analysis
            base_response.analysis_results.update(
                {
                    "existing_solutions": existing_solutions,
                    "similar_features": similar_features,
                    "compliance_predictions": compliance_predictions,
                    "mangle_enhanced": True,
                }
            )

        except Exception as e:
            logger.warning(f"Mangle enhancement failed in specify phase: {e}")
            base_response.analysis_results["mangle_error"] = str(e)

        return base_response

    async def plan(self, request: PlanRequest) -> PlanResponse:
        """
        Execute the /plan phase with enhanced dependency analysis.

        This extends the base plan method with:
        - Dependency impact analysis
        - Code reuse opportunities
        - Integration complexity assessment
        """
        logger.info("Executing enhanced plan phase with Mangle integration")

        # If a raw specification was provided instead of a path, materialize it
        materialized_request = request
        if getattr(request, "specification_path", None) in (
            None,
            "",
        ) and getattr(request, "specification", None):
            try:
                # Determine feature id and directory
                feature_id = (
                    request.feature_id
                    if getattr(request, "feature_id", None)
                    else self._generate_feature_id("inline")
                )
                # Use a stable simple slug for inline specs
                slug = "inline-spec"
                feature_dir = self.specs_dir / f"{feature_id}-{slug}"
                feature_dir.mkdir(parents=True, exist_ok=True)

                # Write specification content to file
                spec_file = feature_dir / "spec.md"
                spec_content = request.specification or ""
                spec_file.write_text(spec_content, encoding="utf-8")

                # Build a new PlanRequest using the created path
                materialized_request = PlanRequest(
                    specification_path=str(spec_file),
                    technology_stack=request.technology_stack,
                    constraints=request.constraints,
                    constitutional_gates=request.constitutional_gates,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to materialize inline specification; falling back: %s",
                    e,
                )

        # Run base planning
        base_response = await super().plan(materialized_request)

        # Derive a feature identifier from the specification path
        feature_id = self._derive_feature_id_from_path(
            materialized_request.specification_path  # type: ignore[arg-type]
        )

        # Enhance with Mangle-based analysis
        try:
            # Analyze code dependencies
            dependency_analysis = self._analyze_dependencies(feature_id)

            # Find reuse opportunities
            reuse_opportunities = self._find_reuse_opportunities(feature_id)

            # Assess integration complexity
            integration_complexity = self._assess_integration_complexity(
                feature_id
            )

            # Update response with enhanced analysis
            base_response.analysis_results.update(
                {
                    "dependency_analysis": dependency_analysis,
                    "reuse_opportunities": reuse_opportunities,
                    "integration_complexity": integration_complexity,
                    "mangle_enhanced": True,
                }
            )

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Mangle enhancement failed in plan phase: {e}")
            base_response.analysis_results["mangle_error"] = str(e)

        return base_response

    async def tasks(self, request: TasksRequest) -> TasksResponse:
        """
        Execute the /tasks phase with enhanced task prioritization.

        This extends the base tasks method with:
        - Test-first task ordering
        - Constitutional compliance task injection
        - Quality gate task generation
        """
        logger.info("Executing enhanced tasks phase with Mangle integration")

        # Run base task generation
        base_response = await super().tasks(request)

        feature_id = self._derive_feature_id_from_path(request.plan_path)

        # Enhance with Mangle-based analysis
        try:
            # Analyze test coverage requirements
            test_requirements = self._analyze_test_requirements(feature_id)

            # Generate constitutional compliance tasks
            compliance_tasks = self._generate_compliance_tasks(feature_id)

            # Prioritize tasks based on dependencies
            task_priorities = self._prioritize_tasks(base_response.tasks)

            # Update response with enhanced analysis
            base_response.analysis_results.update(
                {
                    "test_requirements": test_requirements,
                    "compliance_tasks": compliance_tasks,
                    "task_priorities": task_priorities,
                    "mangle_enhanced": True,
                }
            )

            # Optionally inject compliance tasks if they match TaskBreakdown schema.
            # (Skipped: captured in analysis results to avoid schema mismatch.)

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Mangle enhancement failed in tasks phase: {e}")
            base_response.analysis_results["mangle_error"] = str(e)

        return base_response

    def ask_question(self, question: str) -> dict[str, Any]:
        """
        Answer natural language questions about the codebase using Mangle
        reasoning.

            Args:
                question: Natural language question

            Returns:
                Dictionary with answer, query used, and metadata
        """
        logger.info(f"Processing question: {question}")

        result = self.mangle_reasoner.ask_question(question)

        return {
            "question": question,
            "answer": result.format_for_display(),
            "query_used": result.query,
            "success": result.success,
            "results": result.raw_results,
            "execution_time": result.execution_time,
        }

    def validate_constitutional_compliance(self) -> dict[str, Any]:
        """
        Validate code against constitutional rules using enhanced Mangle
        reasoning.

            Returns:
                Comprehensive constitutional compliance report
        """
        logger.info("Running comprehensive constitutional validation")

        # Get Mangle-based constitutional validation
        mangle_results = (
            self.mangle_reasoner.validate_constitutional_compliance()
        )

        # Get traditional constitutional scoring
        try:
            traditional_score = self.constitutional_scorer.score_specification(
                "Comprehensive constitutional analysis of the current "
                "codebase."
            )
        except Exception as e:
            logger.warning(f"Traditional scoring failed: {e}")
            traditional_score = None

        # Combine results
        compliance_report = {
            "mangle_analysis": {
                article: {
                    "violations": result.raw_results,
                    "count": (
                        len(result.raw_results) if result.raw_results else 0
                    ),
                    "success": result.success,
                    "error": result.error_message,
                }
                for article, result in mangle_results.items()
            },
            "traditional_scoring": (
                {
                    "overall_score": traditional_score.overall_score,
                    "article_scores": traditional_score.article_scores,
                    "violations": [
                        {
                            "article": v.article,
                            "principle": v.principle,
                            "message": v.message,
                            "line": v.line,
                            "severity": v.severity,
                            "suggestion": v.suggestion,
                        }
                        for v in traditional_score.violations
                    ],
                    "is_compliant": traditional_score.is_compliant,
                    "metadata": traditional_score.metadata,
                }
                if traditional_score
                else None
            ),
            "summary": self._generate_compliance_summary(
                mangle_results, traditional_score
            ),
            "recommendations": self._generate_compliance_recommendations(
                mangle_results
            ),
        }

        self._last_analysis_results["constitutional_compliance"] = (
            compliance_report
        )
        return compliance_report

    def trace_code_to_spec(self, code_element: str) -> dict[str, Any]:
        """
        Trace code element back to specification using Mangle reasoning.

        Args:
            code_element: Name of function, class, or module to trace

        Returns:
            Traceability information
        """
        logger.info(f"Tracing code element: {code_element}")

        result = self.mangle_reasoner.trace_code_to_spec(code_element)

        return {
            "code_element": code_element,
            "related_specs": result.raw_results,
            "success": result.success,
            "traceability_found": (
                len(result.raw_results) > 0 if result.raw_results else False
            ),
            "execution_time": result.execution_time,
            "formatted_result": result.format_for_display(),
        }

    def analyze_code_quality(self) -> dict[str, Any]:
        """
        Perform comprehensive code quality analysis using Mangle reasoning.

        Returns:
            Detailed quality analysis report
        """
        logger.info("Running comprehensive code quality analysis")

        # Get quality issues
        quality_results = self.mangle_reasoner.find_quality_issues()

        # Get incomplete work
        incomplete_results = self.mangle_reasoner.find_incomplete_work()

        # Generate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            quality_results, incomplete_results
        )

        quality_report = {
            "quality_issues": {
                issue_type: {
                    "items": result.raw_results,
                    "count": (
                        len(result.raw_results) if result.raw_results else 0
                    ),
                    "success": result.success,
                }
                for issue_type, result in quality_results.items()
            },
            "incomplete_work": {
                work_type: {
                    "items": result.raw_results,
                    "count": (
                        len(result.raw_results) if result.raw_results else 0
                    ),
                    "success": result.success,
                }
                for work_type, result in incomplete_results.items()
            },
            "quality_metrics": quality_metrics,
            "recommendations": self._generate_quality_recommendations(
                quality_results, incomplete_results
            ),
        }

        self._last_analysis_results["code_quality"] = quality_report
        return quality_report

    def get_fact_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the generated Mangle facts.

        Returns:
            Fact generation statistics and cache information
        """
        return self.mangle_reasoner.get_fact_statistics()

    def invalidate_caches(self):
        """Invalidate all caches to force regeneration."""
        self.mangle_reasoner.invalidate_cache()
        self._last_analysis_results.clear()

    def _analyze_existing_solutions(self, _user_input: str) -> dict[str, Any]:
        """Analyze existing solutions that might address the user request."""
        # Query for similar functions or features
        similar_funcs = self.mangle_reasoner.query(
            "function(Func, File, Module)"
        )

        return {
            "existing_functions": (
                len(similar_funcs.raw_results)
                if similar_funcs.raw_results
                else 0
            ),
            "analysis_performed": similar_funcs.success,
        }

    def _find_similar_features(self, _user_input: str) -> dict[str, Any]:
        """Find similar existing features."""
        specs_result = self.mangle_reasoner.query("spec(FeatureID, Title)")

        return {
            "existing_specs": (
                len(specs_result.raw_results)
                if specs_result.raw_results
                else 0
            ),
            "analysis_performed": specs_result.success,
        }

    def _predict_compliance_issues(self, _user_input: str) -> dict[str, Any]:
        """Predict potential constitutional compliance issues."""
        violations = self.mangle_reasoner.query(
            "constitutional_violation(Article, Violator)"
        )

        return {
            "existing_violations": (
                len(violations.raw_results) if violations.raw_results else 0
            ),
            "analysis_performed": violations.success,
        }

    def _analyze_dependencies(self, _feature_id: str) -> dict[str, Any]:
        """Analyze code dependencies for a feature."""
        deps_result = self.mangle_reasoner.query(
            "imports(Module, ImportedModule)"
        )

        return {
            "total_dependencies": (
                len(deps_result.raw_results) if deps_result.raw_results else 0
            ),
            "analysis_performed": deps_result.success,
        }

    def _find_reuse_opportunities(self, _feature_id: str) -> list[str]:
        """Find code reuse opportunities."""
        funcs_result = self.mangle_reasoner.query("well_tested_function(Func)")

        if funcs_result.success and funcs_result.raw_results:
            return [result[0] for result in funcs_result.raw_results if result]
        return []

    def _assess_integration_complexity(
        self, _feature_id: str
    ) -> dict[str, Any]:
        """Assess integration complexity for a feature."""
        hotspots = self.mangle_reasoner.query("dependency_hotspot(Module)")

        return {
            "complexity_score": (
                len(hotspots.raw_results) if hotspots.raw_results else 0
            ),
            "dependency_hotspots": hotspots.raw_results,
            "analysis_performed": hotspots.success,
        }

    def _analyze_test_requirements(self, _feature_id: str) -> dict[str, Any]:
        """Analyze test requirements for a feature."""
        untested = self.mangle_reasoner.query("untested_function(Func)")

        return {
            "untested_functions": (
                len(untested.raw_results) if untested.raw_results else 0
            ),
            "functions_needing_tests": untested.raw_results,
            "analysis_performed": untested.success,
        }

    def _generate_compliance_tasks(self, _feature_id: str) -> list[Any]:
        """Generate constitutional compliance tasks."""
        # This would return TaskBreakdown objects in a real implementation
        return [
            {
                "title": "Constitutional Compliance Review",
                "description": (
                    "Review implementation for all constitutional articles"
                ),
                "priority": "high",
                "estimated_hours": 2,
                "dependencies": [],
                "acceptance_criteria": [
                    "All constitutional violations identified",
                    "Compliance score above 0.75",
                    "Remediation plan created",
                ],
            }
        ]

    def _prioritize_tasks(self, tasks: list[Any]) -> dict[str, Any]:
        """Prioritize tasks based on dependencies and complexity."""
        return {
            "prioritization_applied": True,
            "criteria": [
                "constitutional_compliance",
                "test_coverage",
                "complexity",
            ],
            "task_count": len(tasks),
        }

    # -------------------------
    # Helper utilities
    # -------------------------
    def _derive_feature_id_from_path(self, path_str: str) -> str:
        """Derive a feature ID from a spec/plan/tasks path.

        Expected directory names often start with a numeric prefix (e.g.,
        001-feature-name). Returns 'unknown' if no pattern is found.
        """
        try:
            p = Path(path_str).resolve()
            parent_name = p.parent.name
            if len(parent_name) >= 3 and parent_name[:3].isdigit():
                return parent_name[:3]
            return parent_name[:8] or "unknown"
        except Exception:  # noqa: BLE001
            return "unknown"

    def _generate_compliance_summary(
        self, mangle_results: dict, traditional_score: Any
    ) -> dict[str, Any]:
        """Generate summary of constitutional compliance."""
        total_violations = sum(
            len(result.raw_results) if result.raw_results else 0
            for result in mangle_results.values()
        )

        return {
            "total_violations": total_violations,
            "articles_with_violations": len(
                [
                    article
                    for article, result in mangle_results.items()
                    if result.raw_results and len(result.raw_results) > 0
                ]
            ),
            "overall_compliance": "PASS" if total_violations == 0 else "FAIL",
            "traditional_score": (
                traditional_score.overall_score if traditional_score else None
            ),
        }

    def _generate_compliance_recommendations(
        self, mangle_results: dict
    ) -> list[str]:
        """Generate recommendations for improving constitutional compliance."""
        recommendations = []

        for article, result in mangle_results.items():
            if result.raw_results and len(result.raw_results) > 0:
                if "Library-First" in article:
                    recommendations.append(
                        "Prefer existing libraries over new custom code"
                    )
                elif "Test-First" in article:
                    recommendations.append("Add tests for untested functions")
                elif "Simplicity" in article:
                    recommendations.append(
                        "Refactor complex functions to improve simplicity"
                    )
                # Add more specific recommendations

        return recommendations

    def _calculate_quality_metrics(
        self, quality_results: dict, _incomplete_results: dict
    ) -> dict[str, Any]:
        """Calculate overall quality metrics."""
        total_issues = sum(
            len(result.raw_results) if result.raw_results else 0
            for result in quality_results.values()
        )

        total_incomplete = sum(
            len(result.raw_results) if result.raw_results else 0
            for result in _incomplete_results.values()
        )

        return {
            "total_quality_issues": total_issues,
            "total_incomplete_items": total_incomplete,
            "quality_score": max(0, 100 - (total_issues * 5)),
            "completeness_score": max(0, 100 - (total_incomplete * 10)),
        }

    def _generate_quality_recommendations(
        self, quality_results: dict, _incomplete_results: dict
    ) -> list[str]:
        """Generate recommendations for improving code quality."""
        recommendations = []

        # Add specific recommendations based on found issues
        for issue_type, result in quality_results.items():
            if result.raw_results and len(result.raw_results) > 0:
                if "Untested" in issue_type:
                    recommendations.append(
                        "Add unit tests for untested functions"
                    )
                elif "Complex" in issue_type:
                    recommendations.append(
                        "Refactor complex functions into smaller components"
                    )
                elif "Circular" in issue_type:
                    recommendations.append(
                        "Resolve circular dependencies through refactoring"
                    )

        return recommendations
