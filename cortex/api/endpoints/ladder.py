"""FastAPI endpoints for the Enhanced LADDER Planner."""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cortex.config.planner_config import get_planner_config, update_planner_config
from cortex.planner.ladder_enhanced import EnhancedLadderPlanner

logger = logging.getLogger(__name__)


class PlanRequest(BaseModel):
    """Request model for creating a plan."""

    goal: str
    context: str = ""
    mode: str = "shadow"  # "shadow" or "active"


class PlanResponse(BaseModel):
    """Response model for plan creation."""

    plan_id: str
    title: str
    tasks: list[dict[str, Any]]
    total_energy: float
    mode: str
    created_at: datetime


class ExecutionRequest(BaseModel):
    """Request model for plan execution."""

    plan_id: str
    force_mode: str | None = None  # Override planner mode for this execution


class ExecutionResponse(BaseModel):
    """Response model for plan execution."""

    success: bool
    plan_id: str
    results: dict[str, Any]
    final_state: dict[str, str]
    completion_rate: float
    total_reward: float
    execution_time: float


class StatsResponse(BaseModel):
    """Response model for planner statistics."""

    bandit_stats: dict[str, dict[str, float]]
    knowledge_base_summary: dict[str, Any]
    configuration: dict[str, Any]
    current_mode: str


class ModeRequest(BaseModel):
    """Request model for setting planner mode."""

    mode: str


class ConfigRequest(BaseModel):
    """Request model for updating configuration."""

    config: dict[str, Any]


def create_ladder_router(planner: EnhancedLadderPlanner) -> APIRouter:
    """Create the LADDER planner API router."""
    router = APIRouter(prefix="/api/planner", tags=["LADDER Planner"])

    @router.post("/create-plan", response_model=PlanResponse)
    async def create_plan(request: PlanRequest):
        """Create a LADDER plan for a given goal."""
        try:
            logger.info(
                f"Creating plan for goal: {request.goal} (mode: {request.mode})"
            )

            # Set planner mode if specified
            if request.mode and request.mode != planner.mode:
                planner.set_mode(request.mode)

            # Create user event structure
            user_event = type(
                "UserEvent",
                (),
                {"payload": {"query": request.goal, "context": request.context}},
            )()

            # Create the plan
            start_time = datetime.now()
            root_todo = await planner.plan_from_user_event(user_event)

            # Get all children
            all_tasks = []
            if root_todo.children_ids:
                for child_id in root_todo.children_ids:
                    child = planner.store.get(child_id)
                    if child:
                        all_tasks.append(
                            {
                                "id": child.id,
                                "title": child.title,
                                "description": child.description,
                                "energy": child.energy,
                                "priority": child.priority,
                                "status": child.status.value,
                                "tool_hint": child.tool_hint,
                            }
                        )

            # Calculate total energy
            total_energy = sum(task["energy"] for task in all_tasks)

            return PlanResponse(
                plan_id=root_todo.id,
                title=root_todo.title,
                tasks=all_tasks,
                total_energy=total_energy,
                mode=planner.mode,
                created_at=start_time,
            )

        except Exception as e:
            logger.error(f"Failed to create plan: {e}")
            raise HTTPException(
                status_code=500, detail=f"Plan creation failed: {str(e)}"
            )

    @router.post("/execute-plan", response_model=ExecutionResponse)
    async def execute_plan(request: ExecutionRequest):
        """Execute a LADDER plan."""
        try:
            logger.info(f"Executing plan: {request.plan_id}")

            # Override mode if requested
            original_mode = planner.mode
            if request.force_mode and request.force_mode != planner.mode:
                planner.set_mode(request.force_mode)

            # Get the root todo
            root_todo = planner.store.get(request.plan_id)
            if not root_todo:
                raise HTTPException(status_code=404, detail="Plan not found")

            # Get all children
            children = []
            if root_todo.children_ids:
                children = [
                    planner.store.get(child_id) for child_id in root_todo.children_ids
                ]
                children = [child for child in children if child]  # Filter None values

            if not children:
                raise HTTPException(
                    status_code=400, detail="Plan has no executable tasks"
                )

            # Execute the plan
            start_time = datetime.now()
            await planner._enhanced_execute(root_todo, children)
            await planner._enhanced_review(root_todo, children)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Calculate results
            successful_tasks = sum(
                1
                for child in children
                if planner.store.get(child.id).status.value == "done"
            )
            completion_rate = successful_tasks / len(children) if children else 0.0

            # Get final state
            final_state = {}
            total_reward = 0.0
            for child in children:
                current_child = planner.store.get(child.id)
                final_state[child.id] = current_child.status.value
                # Simple reward calculation
                if current_child.status.value == "done":
                    total_reward += 1.0

            # Restore original mode if it was overridden
            if request.force_mode and request.force_mode != original_mode:
                planner.set_mode(original_mode)

            return ExecutionResponse(
                success=completion_rate > 0.5,  # 50% success threshold
                plan_id=request.plan_id,
                results={
                    "executed_tasks": len(children),
                    "successful_tasks": successful_tasks,
                },
                final_state=final_state,
                completion_rate=completion_rate,
                total_reward=total_reward,
                execution_time=execution_time,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to execute plan: {e}")
            raise HTTPException(
                status_code=500, detail=f"Plan execution failed: {str(e)}"
            )

    @router.get("/stats", response_model=StatsResponse)
    async def get_planner_stats():
        """Get planner statistics and bandit learning data."""
        try:
            return StatsResponse(
                bandit_stats=planner.get_bandit_stats(),
                knowledge_base_summary=planner.get_knowledge_base_summary(),
                configuration=get_planner_config().to_dict(),
                current_mode=planner.mode,
            )
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get stats: {str(e)}"
            )

    @router.post("/set-mode")
    async def set_planner_mode(request: ModeRequest):
        """Set planner mode (shadow/active)."""
        try:
            if request.mode not in ["shadow", "active"]:
                raise HTTPException(
                    status_code=400, detail="Mode must be 'shadow' or 'active'"
                )

            planner.set_mode(request.mode)
            logger.info(f"Planner mode set to: {request.mode}")

            return {
                "message": f"Planner mode set to {request.mode}",
                "mode": request.mode,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to set mode: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to set mode: {str(e)}")

    @router.post("/config")
    async def update_configuration(request: ConfigRequest):
        """Update planner configuration."""
        try:
            errors = update_planner_config(request.config)

            if errors:
                raise HTTPException(
                    status_code=400,
                    detail=f"Configuration validation failed: {', '.join(errors)}",
                )

            # Update planner exploration rate if changed
            config = get_planner_config()
            planner.exploration_rate = config.exploration_rate

            logger.info("Planner configuration updated successfully")
            return {
                "message": "Configuration updated successfully",
                "config": config.to_dict(),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to update configuration: {str(e)}"
            )

    @router.get("/config")
    async def get_configuration():
        """Get current planner configuration."""
        try:
            config = get_planner_config()
            return {
                "config": config.to_dict(),
                "current_mode": planner.mode,
                "exploration_rate": planner.exploration_rate,
            }
        except Exception as e:
            logger.error(f"Failed to get configuration: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get configuration: {str(e)}"
            )

    @router.post("/plan-and-execute")
    async def plan_and_execute(request: PlanRequest):
        """Create and immediately execute a plan (convenience endpoint)."""
        try:
            logger.info(f"Planning and executing goal: {request.goal}")

            # Create plan
            plan_response = await create_plan(request)

            # Execute plan
            execution_request = ExecutionRequest(plan_id=plan_response.plan_id)
            execution_response = await execute_plan(execution_request)

            return {"plan": plan_response, "execution": execution_response}

        except Exception as e:
            logger.error(f"Failed to plan and execute: {e}")
            raise HTTPException(
                status_code=500, detail=f"Plan and execute failed: {str(e)}"
            )

    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "planner_mode": planner.mode,
            "timestamp": datetime.now().isoformat(),
            "bandit_tools": len(planner.bandit_stats),
            "knowledge_base_size": len(planner.knowledge_base),
        }

    return router
