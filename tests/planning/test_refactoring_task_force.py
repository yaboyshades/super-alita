import pytest

from src.planning.refactoring_task_force import (
    RefactoringFocus,
    RefactoringObjective,
    RefactoringTaskForce,
    RefactoringTaskState,
    TaskForceMember,
    TaskStage,
)


def build_sample_force() -> RefactoringTaskForce:
    objectives = [
        RefactoringObjective(
            objective_id="runtime-stability",
            description="Harden the runtime streaming reliability gates.",
            focus_area=RefactoringFocus.RELIABILITY,
            priority=1,
            success_metrics={"error_budget": 0.1},
        ),
        RefactoringObjective(
            objective_id="coverage-upgrade",
            description="Increase runtime coverage for streaming router.",
            focus_area=RefactoringFocus.TEST_COVERAGE,
            priority=2,
            success_metrics={"coverage": 0.9},
        ),
    ]
    members = [
        TaskForceMember("Rhea", RefactoringFocus.RELIABILITY, capacity=2.0),
        TaskForceMember("Tomas", RefactoringFocus.TEST_COVERAGE, capacity=1.5),
        TaskForceMember("Priya", RefactoringFocus.PERFORMANCE, capacity=1.0),
    ]
    return RefactoringTaskForce(objectives, members)


def test_initial_task_decomposition_and_assignment():
    force = build_sample_force()

    stability_tasks = force.get_tasks_for_objective("runtime-stability")
    assert {task.stage for task in stability_tasks} == {
        TaskStage.ANALYSIS,
        TaskStage.IMPLEMENTATION,
        TaskStage.VALIDATION,
        TaskStage.INTEGRATION,
    }

    integration_task = next(
        task for task in stability_tasks if task.stage == TaskStage.INTEGRATION
    )
    expected_dependencies = {
        "runtime-stability:analysis",
        "runtime-stability:implementation",
        "runtime-stability:validation",
    }
    assert set(integration_task.dependencies) == expected_dependencies
    assert integration_task.state == RefactoringTaskState.BLOCKED

    analysis_task = next(
        task for task in stability_tasks if task.stage == TaskStage.ANALYSIS
    )
    validation_task = next(
        task for task in stability_tasks if task.stage == TaskStage.VALIDATION
    )

    assert analysis_task.assigned_to == "Rhea"
    assert validation_task.assigned_to == "Tomas"


def test_convergence_plan_transitions_with_progress():
    force = build_sample_force()

    stability_tasks = force.get_tasks_for_objective("runtime-stability")
    integration_task = next(
        task for task in stability_tasks if task.stage == TaskStage.INTEGRATION
    )

    for task in stability_tasks:
        if task.stage != TaskStage.INTEGRATION:
            force.update_task_state(task.task_id, state=RefactoringTaskState.COMPLETE)

    updated_integration = force.update_task_state(integration_task.task_id)
    assert updated_integration.state == RefactoringTaskState.READY

    plan = force.generate_convergence_plan()
    assert plan[0]["objective_id"] == "runtime-stability"
    stability_entry = next(
        item for item in plan if item["objective_id"] == "runtime-stability"
    )
    assert stability_entry["ready_for_integration"] is True
    assert stability_entry["converged"] is False
    assert stability_entry["loop_alignment"]["bonds"] == integration_task.dependencies
    assert stability_entry["loop_alignment"]["energy"] > 0

    force.update_task_state(
        integration_task.task_id, state=RefactoringTaskState.COMPLETE
    )
    plan = force.generate_convergence_plan()
    stability_entry = next(
        item for item in plan if item["objective_id"] == "runtime-stability"
    )
    assert stability_entry["converged"] is True
    assert stability_entry["pending_tasks"] == []
    assert stability_entry["loop_alignment"]["reward"] == {"error_budget": 0.1}


def test_progress_updates_require_valid_range():
    force = build_sample_force()
    tasks = force.get_tasks_for_objective("coverage-upgrade")
    analysis_task = next(task for task in tasks if task.stage == TaskStage.ANALYSIS)

    updated = force.update_task_state(analysis_task.task_id, progress=0.3)
    assert updated.progress == pytest.approx(0.3)
    assert updated.state == RefactoringTaskState.IN_PROGRESS

    with pytest.raises(ValueError):
        force.update_task_state(analysis_task.task_id, progress=1.2)


def test_status_board_tracks_assignments_and_states():
    force = build_sample_force()

    board = force.generate_status_board()
    stability_summary = next(
        entry for entry in board["objectives"] if entry["objective_id"] == "runtime-stability"
    )
    assert stability_summary["states"][RefactoringTaskState.READY.value] == 3
    assert stability_summary["states"][RefactoringTaskState.BLOCKED.value] == 1
    assert stability_summary["progress"] == pytest.approx(0.0)
    assert "runtime-stability:validation" in board["ready_next"]

    force.update_task_state("runtime-stability:analysis", progress=0.5)
    force.update_task_state(
        "runtime-stability:implementation", state=RefactoringTaskState.IN_PROGRESS
    )

    updated_board = force.generate_status_board()
    stability_summary = next(
        entry
        for entry in updated_board["objectives"]
        if entry["objective_id"] == "runtime-stability"
    )
    assert stability_summary["states"][RefactoringTaskState.IN_PROGRESS.value] == 2
    assert stability_summary["progress"] > 0
    assert "runtime-stability:analysis" in stability_summary["active"]["in_progress"]

    rhea_summary = next(
        member for member in updated_board["members"] if member["name"] == "Rhea"
    )
    assert any(
        assignment["task_id"] == "runtime-stability:analysis"
        for assignment in rhea_summary["assignments"]
    )
    assert rhea_summary["available_capacity"] == pytest.approx(
        rhea_summary["capacity"] - rhea_summary["allocation"]
    )
