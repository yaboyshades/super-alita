from pathlib import Path

import pytest

from src.core.yaml_utils import safe_load
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
    assert spec_res.next_step_guidance is not None
    assert spec_res.next_step_metadata_path is not None

    metadata_file = (
        Path(pipeline.workspace_root) / spec_res.next_step_metadata_path
    )
    assert metadata_file.exists()

    metadata = safe_load(metadata_file.read_text(encoding="utf-8"))
    assert metadata
    assert metadata.get("clarifications") is not None
    assert metadata.get("artefacts") is not None
    assert metadata.get("commands") is not None

    spec_text = spec_file.read_text(encoding="utf-8")
    assert "## Next Steps Guidance" in spec_text
    assert "Outstanding Clarifications" in spec_text
    assert "Command Checklist" in spec_text
    assert "- [" in spec_text

    assert metadata["commands"]
    assert metadata["commands"][0]["source"] == "command"

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
    assert plan_res.feature_id == spec_res.feature_id
    assert plan_res.next_step_guidance is not None
    assert plan_res.next_step_metadata_path == spec_res.next_step_metadata_path
    assert plan_res.next_steps

    plan_metadata = safe_load(metadata_file.read_text(encoding="utf-8"))
    assert plan_metadata
    assert all(
        item["status"] == "complete"
        for item in plan_metadata["clarifications"]
    )
    plan_command = next(
        item
        for item in plan_metadata["commands"]
        if "/plan" in item["action"].lower()
    )
    assert plan_command["status"] == "complete"

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
    assert tasks_res.feature_id == plan_res.feature_id
    assert tasks_res.next_step_guidance is not None
    assert (
        tasks_res.next_step_metadata_path == spec_res.next_step_metadata_path
    )
    assert tasks_res.next_steps
    assert any(task.id.startswith("NS-") for task in tasks_res.tasks)
    assert "Guidance Follow-ups" in tasks_file.read_text(encoding="utf-8")

    post_metadata = safe_load(metadata_file.read_text(encoding="utf-8"))
    tasks_command = next(
        item
        for item in post_metadata["commands"]
        if "/tasks" in item["action"].lower()
    )
    assert tasks_command["status"] in {"in_progress", "complete"}

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
    critical_ids = {
        task.id for task in tasks_res.tasks if task.priority == "critical"
    }
    assert set(tasks_res.critical_path) == critical_ids

    # Acceptance criteria should be present for every task definition
    assert all(task.acceptance_criteria for task in tasks_res.tasks)
