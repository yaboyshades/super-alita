# mcp_server.py
"""
FastAPI MCP Server for AI Agent Integration.
Provides HTTP endpoints for SDD workflow automation.
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scripts.common import Project
from scripts.constitution_manager import ConstitutionManager
from scripts.plan_manager import PlanManager
from scripts.spec_manager import SpecManager
from scripts.tasks_manager import TasksManager


class ConstitutionRequest(BaseModel):
    """Request to create a constitution."""

    principles: str
    force: bool = False


class SpecificationRequest(BaseModel):
    """Request to create a specification."""

    requirements: str
    feature_name: str
    context: str | None = None


class PlanRequest(BaseModel):
    """Request to create a plan."""

    feature_name: str


class TasksRequest(BaseModel):
    """Request to create tasks."""

    feature_name: str


class ValidationRequest(BaseModel):
    """Request to validate a feature."""

    feature_name: str


class TaskUpdateRequest(BaseModel):
    """Request to update task status."""

    feature_name: str
    task_id: str
    status: str


# Global managers
project: Project | None = None
constitution_mgr: ConstitutionManager | None = None
spec_mgr: SpecManager | None = None
plan_mgr: PlanManager | None = None
tasks_mgr: TasksManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Initialize managers on startup."""
    global project, constitution_mgr, spec_mgr, plan_mgr, tasks_mgr

    # Initialize project and managers
    project = Project()
    constitution_mgr = ConstitutionManager(project)
    spec_mgr = SpecManager(project)
    plan_mgr = PlanManager(project)
    tasks_mgr = TasksManager(project)

    yield

    # Cleanup if needed
    pass


app = FastAPI(
    title="Spec Kit MCP Server",
    description="AI Agent Integration for SDD Workflows",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "spec-kit-mcp", "version": "1.0.0"}


@app.post("/constitution")
async def create_constitution(request: ConstitutionRequest) -> dict[str, Any]:
    """Create a project constitution."""
    if not constitution_mgr:
        raise HTTPException(status_code=500, detail="Server not initialized")

    try:
        path = constitution_mgr.create_constitution(request.principles, request.force)
        return {
            "success": True,
            "message": f"Constitution created at {path}",
            "path": str(path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/specify")
async def create_specification(request: SpecificationRequest) -> dict[str, Any]:
    """Create a feature specification."""
    if not spec_mgr:
        raise HTTPException(status_code=500, detail="Server not initialized")

    try:
        path = spec_mgr.create_specification(
            request.feature_name, request.requirements, request.context
        )
        return {
            "success": True,
            "message": f"Specification created at {path}",
            "path": str(path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/plan")
async def create_plan(request: PlanRequest) -> dict[str, Any]:
    """Create a technical implementation plan."""
    if not plan_mgr:
        raise HTTPException(status_code=500, detail="Server not initialized")

    try:
        path = plan_mgr.create_plan(request.feature_name)
        return {
            "success": True,
            "message": f"Plan created at {path}",
            "path": str(path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/tasks")
async def create_tasks(request: TasksRequest) -> dict[str, Any]:
    """Create executable tasks from a plan."""
    if not tasks_mgr:
        raise HTTPException(status_code=500, detail="Server not initialized")

    try:
        path = tasks_mgr.create_tasks(request.feature_name)
        return {
            "success": True,
            "message": f"Tasks created at {path}",
            "path": str(path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/validate")
async def validate_feature(request: ValidationRequest) -> dict[str, Any]:
    """Validate a feature against the constitution."""
    if not constitution_mgr or not project:
        raise HTTPException(status_code=500, detail="Server not initialized")

    # Find feature directory
    feature_path = project.get_feature_path(request.feature_name)
    if not feature_path:
        raise HTTPException(
            status_code=404, detail=f"Feature '{request.feature_name}' not found"
        )

    # Get feature context
    context_parts = []
    for file_name in ["spec.md", "plan.md", "tasks.md"]:
        file_path = feature_path / file_name
        if file_path.exists():
            content = file_path.read_text()
            context_parts.append(f"## {file_name.upper()}\n{content}")

    feature_context = "\n\n".join(context_parts)

    try:
        result = constitution_mgr.validate_constitution(feature_context)
        return {
            "success": True,
            "feature_name": request.feature_name,
            "score": result.overall_score,
            "passed": result.passed,
            "recommendations": result.recommendations,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/task/update")
async def update_task_status(request: TaskUpdateRequest) -> dict[str, Any]:
    """Update the status of a task."""
    if not tasks_mgr:
        raise HTTPException(status_code=500, detail="Server not initialized")

    try:
        tasks_mgr.update_task_status(
            request.feature_name, request.task_id, request.status
        )
        return {
            "success": True,
            "message": f"Task {request.task_id} status updated to {request.status}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


class EvolveSpecPlanRequest(BaseModel):
    """Request to evolve a spec and plan using knowledge base."""

    feature_name: str
    evolution_context: str
    max_iterations: int = 5


@app.post("/evolve_spec_plan")
async def evolve_spec_plan(request: EvolveSpecPlanRequest) -> dict[str, Any]:
    """
    Evolve a specification and plan using knowledge base insights.
    This is the core evolutionary loop that Super Alita uses to improve
    its understanding and generate better implementations.
    """
    if not spec_mgr or not plan_mgr or not project:
        raise HTTPException(status_code=500, detail="Server not initialized")

    try:
        # Phase 1: Load current spec and plan
        feature_path = project.get_feature_path(request.feature_name)
        if not feature_path:
            raise HTTPException(
                status_code=404, detail=f"Feature '{request.feature_name}' not found"
            )

        current_spec = ""
        current_plan = ""

        spec_file = feature_path / "spec.md"
        plan_file = feature_path / "plan.md"

        if spec_file.exists():
            current_spec = spec_file.read_text()
        if plan_file.exists():
            current_plan = plan_file.read_text()

        # Phase 2: Initialize evolutionary context
        evolution_history = []
        best_score = 0.0
        best_spec = current_spec
        best_plan = current_plan

        # Phase 3: Evolutionary loop
        for iteration in range(request.max_iterations):
            # Generate improved spec using AI and context
            improved_spec = await _evolve_specification(
                request.feature_name,
                current_spec,
                request.evolution_context,
                evolution_history,
            )

            # Generate improved plan based on evolved spec
            improved_plan = await _evolve_plan(
                request.feature_name, improved_spec, current_plan, evolution_history
            )

            # Evaluate the improvements
            evaluation = await _evaluate_evolution(
                improved_spec, improved_plan, request.evolution_context
            )

            evolution_history.append(
                {
                    "iteration": iteration + 1,
                    "spec": improved_spec,
                    "plan": improved_plan,
                    "score": evaluation["score"],
                    "feedback": evaluation["feedback"],
                }
            )

            # Keep best version
            if evaluation["score"] > best_score:
                best_score = evaluation["score"]
                best_spec = improved_spec
                best_plan = improved_plan

            # Check for convergence
            if evaluation["score"] >= 0.95:  # High quality threshold
                break

        # Phase 4: Save evolved artifacts
        if best_spec != current_spec:
            spec_file.write_text(best_spec)
        if best_plan != current_plan:
            plan_file.write_text(best_plan)

        return {
            "success": True,
            "feature_name": request.feature_name,
            "final_score": best_score,
            "iterations_completed": len(evolution_history),
            "evolution_history": evolution_history,
            "improved": best_spec != current_spec or best_plan != current_plan,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _evolve_specification(
    feature_name: str,
    current_spec: str,
    evolution_context: str,
    history: list[dict[str, Any]],
) -> str:
    """Evolve the specification using AI and knowledge base insights."""
    # This would integrate with the knowledge base to generate
    # improved specifications based on similar features, patterns, etc.
    # For now, this is a placeholder that uses the existing spec manager

    evolution_prompt = f"""
    Evolve the specification for feature: {feature_name}

    Current specification:
    {current_spec}

    Evolution context: {evolution_context}

    Previous evolution attempts: {len(history)}

    Generate an improved specification that addresses the evolution context
    and learns from previous attempts.
    """

    # Use the AI generation from common.py
    from scripts.common import invoke_ai_generation

    try:
        improved_spec = await invoke_ai_generation(evolution_prompt)
        return improved_spec if improved_spec else current_spec
    except Exception:
        return current_spec


async def _evolve_plan(
    feature_name: str,
    evolved_spec: str,
    current_plan: str,
    history: list[dict[str, Any]],
) -> str:
    """Evolve the plan based on the improved specification."""
    evolution_prompt = f"""
    Evolve the implementation plan for feature: {feature_name}

    Evolved specification:
    {evolved_spec}

    Current plan:
    {current_plan}

    Previous evolution attempts: {len(history)}

    Generate an improved implementation plan that better addresses
    the evolved specification and incorporates lessons from previous attempts.
    """

    from scripts.common import invoke_ai_generation

    try:
        improved_plan = await invoke_ai_generation(evolution_prompt)
        return improved_plan if improved_plan else current_plan
    except Exception:
        return current_plan


async def _evaluate_evolution(spec: str, plan: str, context: str) -> dict[str, Any]:
    """Evaluate the quality of the evolved spec and plan."""
    evaluation_prompt = f"""
    Evaluate the quality of this evolved specification and plan.

    Specification:
    {spec}

    Plan:
    {plan}

    Evolution context: {context}

    Provide a quality score (0.0-1.0) and feedback for improvement.
    Format: SCORE: <number>
    FEEDBACK: <text>
    """

    from scripts.common import invoke_ai_generation

    try:
        response = await invoke_ai_generation(evaluation_prompt)
        if response:
            # Parse score and feedback from response
            lines = response.split("\n")
            score = 0.5  # Default
            feedback = "Evaluation completed"

            for line in lines:
                if line.startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                        score = max(0.0, min(1.0, score))  # Clamp to 0-1
                    except ValueError:
                        pass
                elif line.startswith("FEEDBACK:"):
                    feedback = line.split(":", 1)[1].strip()

            return {"score": score, "feedback": feedback}
        else:
            return {"score": 0.5, "feedback": "Evaluation failed"}
    except Exception:
        return {"score": 0.5, "feedback": "Evaluation error"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
