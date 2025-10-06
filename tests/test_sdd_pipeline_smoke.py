from pathlib import Path

import pytest

from src.sdd.constitutional_pipeline import ConstitutionalSDDPipeline
from src.sdd.models import PlanRequest, SpecifyRequest, TasksRequest


@pytest.mark.asyncio
async def test_sdd_pipeline_spec_plan_tasks(tmp_path: Path):
    pipeline = ConstitutionalSDDPipeline(tmp_path)

    # specify
    spec_res = await pipeline.specify(
        SpecifyRequest(
            user_input="Implement a simple notes API with search",
            context={"description": "Test context"},
            constitutional_gates=True,
        )
    )

    assert spec_res.success is True
    spec_file = Path(spec_res.feature_path)
    assert spec_file.exists()
    assert spec_res.overall_compliance_score >= 0.0

    # plan
    plan_res = await pipeline.plan(
        PlanRequest(
            specification_path=str(spec_file),
            technology_stack="FastAPI + SQLite",
            constraints={},
            constitutional_gates=True,
        )
    )

    assert plan_res.success is True
    plan_file = Path(plan_res.plan_path)
    assert plan_file.exists()
    assert plan_res.overall_compliance_score >= 0.0

    # tasks
    tasks_res = await pipeline.tasks(
        TasksRequest(
            plan_path=str(plan_file),
            priority_focus="test-first",
            team_size=1,
            constitutional_gates=True,
        )
    )

    assert tasks_res.success is True
    tasks_file = Path(tasks_res.tasks_path)
    assert tasks_file.exists()
    assert tasks_res.estimated_total_hours >= 0

    # Agent-specific task expectations
    expected_task_ids = {
        "ARCH-001",
        "SEC-001",
        "IMPL-001",
        "TEST-001",
        "DOC-001",
        "INT-001",
        "VAL-001",
    }

    actual_task_ids = {task.id for task in tasks_res.tasks}
    assert actual_task_ids == expected_task_ids

    # Ensure markdown breakdown includes agent context
    assert "Architecture Agent" in tasks_res.tasks_breakdown
    assert "Security Agent" in tasks_res.tasks_breakdown
    assert "Priority Focus" in tasks_res.tasks_breakdown

    # Critical path should include all critical-priority tasks
    critical_ids = {task.id for task in tasks_res.tasks if task.priority == "critical"}
    assert set(tasks_res.critical_path) == critical_ids

    # Acceptance criteria should be present for every task definition
    assert all(task.acceptance_criteria for task in tasks_res.tasks)

