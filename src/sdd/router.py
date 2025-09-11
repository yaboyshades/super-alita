"""FastAPI router for SDD endpoints with Mangle integration."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .enhanced_sdd_framework import EnhancedSDDFramework
from .models import (
    PlanRequest,
    PlanResponse,
    SpecifyRequest,
    SpecifyResponse,
    TasksRequest,
    TasksResponse,
)


# Pydantic models for Mangle endpoints
class QuestionRequest(BaseModel):
    """Request model for natural language questions."""

    question: str


class QuestionResponse(BaseModel):
    """Response model for natural language questions."""

    question: str
    answer: str
    success: bool
    results: list[Any]
    execution_time: float
    query_used: str


class ConstitutionalComplianceResponse(BaseModel):
    """Response model for constitutional compliance validation."""

    mangle_analysis: dict[str, Any]
    traditional_scoring: dict[str, Any] | None
    summary: dict[str, Any]
    recommendations: list[str]


class CodeTraceRequest(BaseModel):
    """Request model for code traceability."""

    code_element: str


class CodeTraceResponse(BaseModel):
    """Response model for code traceability."""

    code_element: str
    related_specs: list[Any]
    success: bool
    traceability_found: bool
    execution_time: float


class QualityAnalysisResponse(BaseModel):
    """Response model for code quality analysis."""

    quality_issues: dict[str, Any]
    incomplete_work: dict[str, Any]
    quality_metrics: dict[str, Any]
    recommendations: list[str]


class FactStatisticsResponse(BaseModel):
    """Response model for fact statistics."""

    total_facts: int
    fact_types: dict[str, int]
    cache_valid: bool
    cached_queries: int
    facts_size_bytes: int


def create_sdd_router(workspace_root: Path | None = None) -> APIRouter:
    """Create enhanced SDD router with Mangle reasoning capabilities."""
    router = APIRouter(prefix="/sdd", tags=["sdd"])
    framework = EnhancedSDDFramework(workspace_root)

    # Enhanced SDD endpoints (existing functionality with Mangle integration)
    @router.post("/specify", response_model=SpecifyResponse)
    async def specify_endpoint(request: SpecifyRequest) -> SpecifyResponse:
        """
        Create feature specification with enhanced constitutional validation.

        This endpoint implements the /specify phase of the SDD workflow with Mangle enhancements:
        1. Generate comprehensive feature specification
        2. Apply constitutional validation gates with deductive reasoning
        3. Analyze existing solutions and similar features
        4. Provide compliance scoring and recommendations
        """
        try:
            result = await framework.specify(request)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/plan", response_model=PlanResponse)
    async def plan_endpoint(request: PlanRequest) -> PlanResponse:
        """
        Generate implementation plan with enhanced dependency analysis.

        This endpoint implements the /plan phase with Mangle enhancements:
        1. Create detailed implementation plan
        2. Analyze code dependencies and reuse opportunities
        3. Assess integration complexity
        4. Apply constitutional validation
        """
        try:
            result = await framework.plan(request)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/tasks", response_model=TasksResponse)
    async def tasks_endpoint(request: TasksRequest) -> TasksResponse:
        """
        Generate implementation tasks with enhanced prioritization.

        This endpoint implements the /tasks phase with Mangle enhancements:
        1. Break down implementation into specific tasks
        2. Prioritize based on dependencies and complexity
        3. Inject constitutional compliance tasks
        4. Provide test-first task ordering
        """
        try:
            result = await framework.tasks(request)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    # New Mangle reasoning endpoints
    @router.post("/ask", response_model=QuestionResponse)
    async def ask_question(request: QuestionRequest) -> QuestionResponse:
        """
        Ask natural language questions about the codebase.

        Uses Mangle deductive reasoning to answer questions such as:
        - "What functions are untested?"
        - "What features are incomplete?"
        - "What violates constitutional principles?"
        - "What are the quality issues?"

        The system maps natural language to Mangle queries and provides
        structured results with execution metadata.
        """
        try:
            result = framework.ask_question(request.question)
            return QuestionResponse(**result)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Question processing failed: {str(e)}"
            ) from e

    @router.get("/validate", response_model=ConstitutionalComplianceResponse)
    async def validate_constitutional_compliance() -> ConstitutionalComplianceResponse:
        """
        Validate code against all constitutional rules using Mangle reasoning.

        Performs comprehensive constitutional compliance analysis:
        - Article I: Library-First Development violations
        - Article II: Test-First Development violations
        - Article III: Simplicity Gate violations
        - Article IV: Integration-First Testing violations
        - Article V: Clarity and Unambiguity violations
        - Article VI: Counterfactual Justification violations

        Returns detailed violation reports, compliance scores, and recommendations.
        """
        try:
            result = framework.validate_constitutional_compliance()
            return ConstitutionalComplianceResponse(**result)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Constitutional validation failed: {str(e)}"
            ) from e

    @router.post("/trace", response_model=CodeTraceResponse)
    async def trace_code_to_spec(request: CodeTraceRequest) -> CodeTraceResponse:
        """
        Trace code elements back to their specifications.

        Uses Mangle reasoning to establish traceability between:
        - Functions and their originating feature specifications
        - Classes and their design requirements
        - Modules and their architectural decisions

        Helps ensure that all code can be traced back to explicit requirements
        and design decisions, supporting constitutional Article V (Clarity).
        """
        try:
            result = framework.trace_code_to_spec(request.code_element)
            return CodeTraceResponse(**result)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Code tracing failed: {str(e)}"
            )

    @router.get("/analyze/quality", response_model=QualityAnalysisResponse)
    async def analyze_code_quality() -> QualityAnalysisResponse:
        """
        Perform comprehensive code quality analysis using Mangle reasoning.

        Analyzes multiple quality dimensions:
        - Untested functions (Article II violations)
        - Complex functions (Article III violations)
        - Large classes and god files
        - Circular dependencies
        - Dependency hotspots
        - Incomplete features and orphaned specifications

        Provides quality metrics, violation counts, and actionable recommendations.
        """
        try:
            result = framework.analyze_code_quality()
            return QualityAnalysisResponse(**result)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Quality analysis failed: {str(e)}"
            )

    @router.get("/stats", response_model=FactStatisticsResponse)
    async def get_knowledge_graph_statistics() -> FactStatisticsResponse:
        """
        Get statistics about the generated knowledge graph.

        Provides insights into:
        - Total number of facts in the knowledge graph
        - Distribution of fact types (functions, classes, specs, etc.)
        - Cache status and performance metrics
        - Knowledge graph size and complexity

        Useful for debugging, performance monitoring, and understanding
        the scope of analysis being performed.
        """
        try:
            result = framework.get_fact_statistics()
            return FactStatisticsResponse(**result)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Statistics retrieval failed: {str(e)}"
            )

    @router.post("/cache/invalidate")
    async def invalidate_caches() -> dict[str, str]:
        """
        Invalidate all caches to force regeneration of facts and queries.

        Use this endpoint when:
        - Code or specifications have been modified
        - You want to ensure fresh analysis results
        - Debugging cache-related issues
        - Preparing for a comprehensive re-analysis

        This will clear both fact generation caches and query result caches.
        """
        try:
            framework.invalidate_caches()
            return {"status": "success", "message": "All caches invalidated"}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Cache invalidation failed: {str(e)}"
            )

    # Convenience endpoints for common queries
    @router.get("/untested-functions")
    async def get_untested_functions() -> dict[str, Any]:
        """Get list of untested functions."""
        try:
            result = framework.ask_question("what functions are untested")
            return {
                "untested_functions": result["results"],
                "count": len(result["results"]) if result["results"] else 0,
                "success": result["success"],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/incomplete-features")
    async def get_incomplete_features() -> dict[str, Any]:
        """Get list of incomplete features."""
        try:
            result = framework.ask_question("what features are incomplete")
            return {
                "incomplete_features": result["results"],
                "count": len(result["results"]) if result["results"] else 0,
                "success": result["success"],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/constitutional-violations")
    async def get_constitutional_violations(
        article: str | None = Query(
            None, description="Specific article to check (I, II, III, IV, V, VI)"
        )
    ) -> dict[str, Any]:
        """Get constitutional violations, optionally filtered by article."""
        try:
            if article:
                # Query specific article
                query_map = {
                    "I": "violates_library_first(Module)",
                    "II": "violates_test_first(Func)",
                    "III": "violates_simplicity(Func)",
                    "IV": "violates_integration_first(FeatureID)",
                    "V": "violates_clarity(Func)",
                    "VI": "violates_counterfactual(Decision)",
                }
                if article.upper() in query_map:
                    from .mangle_reasoner import MangleReasoner

                    reasoner = MangleReasoner(str(framework.workspace_root))
                    result = reasoner.query(query_map[article.upper()])
                    return {
                        "article": article.upper(),
                        "violations": result.raw_results,
                        "count": len(result.raw_results) if result.raw_results else 0,
                        "success": result.success,
                    }
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid article. Use I, II, III, IV, V, or VI",
                    )
            else:
                # Get all violations
                result = framework.ask_question("what violates constitution")
                return {
                    "all_violations": result["results"],
                    "count": len(result["results"]) if result["results"] else 0,
                    "success": result["success"],
                }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Health check endpoint
    @router.get("/health")
    async def health_check() -> dict[str, str]:
        """
        Health check endpoint for the enhanced SDD system.

        Verifies that:
        - Enhanced SDD framework is initialized
        - Mangle reasoner is accessible
        - Workspace is readable

        Returns system status and basic configuration information.
        """
        try:
            stats = framework.get_fact_statistics()
            return {
                "status": "healthy",
                "workspace": str(framework.workspace_root),
                "mangle_integration": "enabled",
                "total_facts": str(stats.get("total_facts", 0)),
                "cache_valid": str(stats.get("cache_valid", False)),
            }
        except Exception as e:
            return {
                "status": "degraded",
                "error": str(e),
                "workspace": str(framework.workspace_root) if framework else "unknown",
                "mangle_integration": "error",
            }

    return router
