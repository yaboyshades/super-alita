"""Tests for the SDD automation scripts."""

from unittest.mock import patch

import pytest

from mcp_server import (
    ConstitutionRequest,
    PlanRequest,
    SpecificationRequest,
    TasksRequest,
    TaskUpdateRequest,
    ValidationRequest,
)
from scripts.common import Project
from scripts.constitution_manager import ConstitutionManager
from scripts.plan_manager import PlanManager
from scripts.sdd_models import (
    ConstitutionResult,
    FeatureSpec,
    Plan,
    Task,
    UserScenario,
)
from scripts.spec_manager import SpecManager
from scripts.tasks_manager import TasksManager


@pytest.fixture(autouse=True)
def mock_openai_api_key(monkeypatch):
    """Mock the OpenAI API key for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class TestProject:
    """Test the Project class."""

    def test_project_initialization(self, tmp_path):
        """Test Project class initialization."""
        project = Project(root=tmp_path)

        assert project.root == tmp_path
        assert project.specs_dir == tmp_path / "specs"
        assert project.templates_dir == tmp_path / "templates"
        assert project.memory_dir == tmp_path / "memory"

    def test_get_feature_path(self, tmp_path):
        """Test getting feature path."""
        project = Project(root=tmp_path)

        # Test with existing feature
        feature_dir = tmp_path / "specs" / "001-test-feature"
        feature_dir.mkdir(parents=True)

        path = project.get_feature_path("001-test-feature")
        assert path == feature_dir

        # Test with non-existing feature
        path = project.get_feature_path("non-existing")
        assert path is None


class TestSDDModels:
    """Test Pydantic models."""

    def test_feature_spec_creation(self):
        """Test FeatureSpec model."""
        spec = FeatureSpec(
            name="Test Feature",
            description="A test feature",
            scenarios=[],
            context={"key": "value"},
        )

        assert spec.name == "Test Feature"
        assert spec.description == "A test feature"

    def test_user_scenario_creation(self):
        """Test UserScenario model."""
        scenario = UserScenario(
            description="User does something to get result",
            priority="medium",
            acceptance_criteria=["done when..."],
            tags=["test"],
        )

        assert scenario.description == "User does something to get result"
        assert scenario.priority == "medium"

    def test_plan_creation(self):
        """Test Plan model."""
        plan = Plan(
            feature_id="test-feature",
            tasks=[],
            rationale="Technical approach",
            assumptions=["assumption1"],
            risks=["risk1"],
        )

        assert plan.feature_id == "test-feature"
        assert plan.rationale == "Technical approach"

    def test_task_creation(self):
        """Test Task model."""
        task = Task(
            id="task-1",
            description="Task description",
            status="pending",
            dependencies=["task-0"],
            assignee="developer",
            priority="medium",
        )

        assert task.id == "task-1"
        assert task.status == "pending"

    def test_constitution_check_creation(self):
        """Test ConstitutionResult model."""
        check = ConstitutionResult(
            feature_id="test-feature",
            checks=[],
            overall_score=0.85,
            passed=True,
            recommendations=["Add tests", "Improve docs"],
        )

        assert check.overall_score == 0.85
        assert check.passed is True


class TestConstitutionManager:
    """Test ConstitutionManager."""

    @pytest.fixture
    def project(self, tmp_path):
        """Create a test project."""
        return Project(root=tmp_path)

    @pytest.fixture
    def manager(self, project):
        """Create a ConstitutionManager."""
        return ConstitutionManager(project)

    @patch("scripts.common.invoke_ai_generation")
    @patch("scripts.common.load_template")
    def test_create_constitution(self, mock_load_template, mock_ai, manager):
        """Test constitution creation."""
        mock_load_template.return_value = (
            "# Constitution Template\n\n## Principles\n\n{{ principles }}"
        )
        mock_ai.return_value = "# Constitution\n\n## Principles\n\n1. Test principle"

        result_path = manager.create_constitution(
            principles="Test principle", force=True
        )

        assert result_path.exists()
        assert result_path.name == "constitution.md"
        content = result_path.read_text()
        assert "Test principle" in content

    @patch("scripts.common.invoke_ai_generation")
    def test_validate_constitution(self, mock_ai, manager):
        """Test constitution validation."""
        mock_ai.return_value = (
            '{"overall_score": 0.9, "passed": true, "recommendations": []}'
        )

        result = manager.validate_constitution("test context")

        assert result.overall_score == 0.9
        assert result.passed is True


class TestSpecManager:
    """Test SpecManager."""

    @pytest.fixture
    def project(self, tmp_path):
        """Create a test project."""
        return Project(root=tmp_path)

    @pytest.fixture
    def manager(self, project):
        """Create a SpecManager."""
        return SpecManager(project)

    @patch("scripts.common.invoke_ai_generation")
    def test_create_specification(self, mock_ai, manager):
        """Test specification creation."""
        mock_ai.return_value = """# Feature Specification

## Overview
Test feature spec

## Requirements
- Req 1
- Req 2
"""

        result_path = manager.create_specification(
            feature_name="test-feature",
            requirements="req1\nreq2",
            context='{"priority": "high"}',
        )

        assert result_path.exists()
        assert "test-feature" in str(result_path)
        content = result_path.read_text()
        assert "Test feature spec" in content


class TestPlanManager:
    """Test PlanManager."""

    @pytest.fixture
    def project(self, tmp_path):
        """Create a test project."""
        return Project(root=tmp_path)

    @pytest.fixture
    def manager(self, project):
        """Create a PlanManager."""
        return PlanManager(project)

    @patch("scripts.common.invoke_ai_generation")
    def test_create_plan(self, mock_ai, manager):
        """Test plan creation."""
        mock_ai.return_value = """# Implementation Plan

## Overview
Test plan

## Components
- Component 1
- Component 2
"""

        # Create a feature directory first
        feature_dir = manager.project.specs_dir / "001-test-feature"
        feature_dir.mkdir(parents=True)

        result_path = manager.create_plan("test-feature")

        assert result_path.exists()
        content = result_path.read_text()
        assert "Test plan" in content


class TestTasksManager:
    """Test TasksManager."""

    @pytest.fixture
    def project(self, tmp_path):
        """Create a test project."""
        return Project(root=tmp_path)

    @pytest.fixture
    def manager(self, project):
        """Create a TasksManager."""
        return TasksManager(project)

    @patch("scripts.common.invoke_ai_generation")
    def test_create_tasks(self, mock_ai, manager):
        """Test tasks creation."""
        mock_ai.return_value = """# Tasks

## Task 1
- Description: First task
- Status: pending

## Task 2
- Description: Second task
- Status: pending
"""

        # Create a feature directory first
        feature_dir = manager.project.specs_dir / "001-test-feature"
        feature_dir.mkdir(parents=True)

        result_path = manager.create_tasks("test-feature")

        assert result_path.exists()
        content = result_path.read_text()
        assert "First task" in content

    def test_update_task_status(self, manager, tmp_path):
        """Test task status update."""
        # Create a tasks file first
        tasks_dir = tmp_path / "specs" / "001-test-feature"
        tasks_dir.mkdir(parents=True)
        tasks_file = tasks_dir / "tasks.md"
        tasks_file.write_text(
            """# Tasks

## Task 1
- Description: First task
- Status: pending

## Task 2
- Description: Second task
- Status: pending
"""
        )

        # Update task status
        manager.update_task_status("001-test-feature", "task-1", "completed")

        content = tasks_file.read_text()
        assert "Status: completed" in content


class TestRequestModels:
    """Test request models for MCP server."""

    def test_constitution_request(self):
        """Test ConstitutionRequest model."""
        req = ConstitutionRequest(principles="principle1\nprinciple2", force=True)

        assert req.principles == "principle1\nprinciple2"
        assert req.force is True

    def test_specification_request(self):
        """Test SpecificationRequest model."""
        req = SpecificationRequest(
            feature_name="test-feature",
            requirements="req1\nreq2",
            context='{"key": "value"}',
        )

        assert req.feature_name == "test-feature"
        assert req.requirements == "req1\nreq2"

    def test_plan_request(self):
        """Test PlanRequest model."""
        req = PlanRequest(feature_name="test-feature")

        assert req.feature_name == "test-feature"

    def test_tasks_request(self):
        """Test TasksRequest model."""
        req = TasksRequest(feature_name="test-feature")

        assert req.feature_name == "test-feature"

    def test_validation_request(self):
        """Test ValidationRequest model."""
        req = ValidationRequest(feature_name="test-feature")

        assert req.feature_name == "test-feature"

    def test_task_update_request(self):
        """Test TaskUpdateRequest model."""
        req = TaskUpdateRequest(
            feature_name="test-feature", task_id="task-1", status="completed"
        )

        assert req.feature_name == "test-feature"
        assert req.task_id == "task-1"
        assert req.status == "completed"
