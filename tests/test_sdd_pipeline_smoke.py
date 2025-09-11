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

