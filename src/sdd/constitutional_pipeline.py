"""Constitutional SDD Pipeline.

Implements the Specification-Driven Development workflow with integrated
constitutional validation at each stage: specify, plan, and tasks.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from ..constitutional import ConstitutionalScorer
from .models import (
    ConstitutionalValidation,
    PlanRequest,
    PlanResponse,
    SpecifyRequest,
    SpecifyResponse,
    TaskBreakdown,
    TasksRequest,
    TasksResponse,
)


class ConstitutionalSDDPipeline:
    """SDD pipeline with constitutional validation at each gate."""

    def __init__(self, workspace_root: Path | None = None):
        """Initialize the constitutional SDD pipeline."""
        self.workspace_root = workspace_root or Path.cwd()
        self.constitutional_scorer = ConstitutionalScorer()
        self.specs_dir = self.workspace_root / "specs"
        self.specs_dir.mkdir(exist_ok=True)

        # Constitutional compliance threshold
        self.compliance_threshold = 0.75

    async def specify(self, request: SpecifyRequest) -> SpecifyResponse:
        """Execute the /specify phase with constitutional validation."""
        # Generate feature metadata
        feature_id = self._generate_feature_id(request.user_input)
        feature_name = self._slugify(request.user_input)
        feature_dir = self.specs_dir / f"{feature_id}-{feature_name}"
        feature_dir.mkdir(parents=True, exist_ok=True)

        # Generate specification content
        specification = self._generate_specification(
            request.user_input, request.context
        )

        # Write specification file
        spec_file = feature_dir / "spec.md"
        spec_file.write_text(specification, encoding="utf-8")

        # Constitutional validation if requested
        constitutional_compliance = {}
        overall_score = 1.0
        threshold_met = True

        if request.constitutional_gates:
            constitutional_compliance = self._validate_specification(specification)
            overall_score = self._calculate_overall_score(constitutional_compliance)
            threshold_met = overall_score >= self.compliance_threshold

        return SpecifyResponse(
            success=True,
            specification=specification,
            feature_id=feature_id,
            feature_path=str(spec_file),
            constitutional_compliance=constitutional_compliance,
            overall_compliance_score=overall_score,
            compliance_threshold_met=threshold_met,
            next_steps=self._get_specify_next_steps(threshold_met),
            timestamp=datetime.now(),
        )

    async def plan(self, request: PlanRequest) -> PlanResponse:
        """Execute the /plan phase with constitutional validation."""
        # Read specification
        if request.specification_path is None:
            # Should be materialized by the enhanced framework; guard just in case
            raise ValueError("specification_path is required for planning")
        spec_path = Path(request.specification_path)
        if not spec_path.exists():
            raise FileNotFoundError(f"Specification not found: {spec_path}")

        specification = spec_path.read_text(encoding="utf-8")

        # Generate implementation plan
        implementation_plan = self._generate_implementation_plan(
            specification, request.technology_stack, request.constraints
        )

        # Write plan file
        plan_file = spec_path.parent / "implementation-plan.md"
        plan_file.write_text(implementation_plan, encoding="utf-8")

        # Generate supporting documents
        supporting_docs = self._generate_supporting_documents(
            spec_path.parent, specification, request.technology_stack
        )

        # Constitutional validation if requested
        constitutional_compliance = {}
        overall_score = 1.0
        threshold_met = True

        if request.constitutional_gates:
            constitutional_compliance = self._validate_implementation_plan(
                implementation_plan
            )
            overall_score = self._calculate_overall_score(constitutional_compliance)
            threshold_met = overall_score >= self.compliance_threshold

        return PlanResponse(
            success=True,
            implementation_plan=implementation_plan,
            plan=implementation_plan,
            plan_path=str(plan_file),
            supporting_documents=supporting_docs,
            constitutional_compliance=constitutional_compliance,
            overall_compliance_score=overall_score,
            compliance_threshold_met=threshold_met,
            technology_recommendations=self._get_tech_recommendations(
                request.technology_stack
            ),
            architecture_decisions=self._extract_architecture_decisions(
                implementation_plan
            ),
            next_steps=self._get_plan_next_steps(threshold_met),
            timestamp=datetime.now(),
        )

    async def tasks(self, request: TasksRequest) -> TasksResponse:
        """Execute the /tasks phase with constitutional validation."""
        # Read or materialize implementation plan
        plan_path = None
        if getattr(request, "plan_path", None):
            plan_path = Path(request.plan_path)  # type: ignore[arg-type]
        elif getattr(request, "plan", None):
            # Materialize raw plan text
            feature_id = getattr(request, "feature_id", None) or "inline"
            feature_dir = self.specs_dir / f"{feature_id}-inline-plan"
            feature_dir.mkdir(parents=True, exist_ok=True)
            plan_file_tmp = feature_dir / "implementation-plan.md"
            plan_file_tmp.write_text(request.plan or "", encoding="utf-8")
            plan_path = plan_file_tmp
        else:
            raise FileNotFoundError(
                "No plan provided: supply plan_path or raw plan content in 'plan'"
            )

        assert plan_path is not None
        if not plan_path.exists():
            raise FileNotFoundError(f"Implementation plan not found: {plan_path}")

        implementation_plan = plan_path.read_text(encoding="utf-8")

        # Generate task breakdown
        tasks_breakdown = self._generate_task_breakdown(
            implementation_plan, request.priority_focus, request.team_size
        )

        # Write tasks file
        tasks_file = plan_path.parent / "tasks.md"
        tasks_file.write_text(tasks_breakdown, encoding="utf-8")

        # Parse structured tasks
        structured_tasks = self._parse_structured_tasks(tasks_breakdown)

        # Constitutional validation if requested
        constitutional_compliance = {}
        overall_score = 1.0
        threshold_met = True

        if request.constitutional_gates:
            constitutional_compliance = self._validate_task_breakdown(tasks_breakdown)
            overall_score = self._calculate_overall_score(constitutional_compliance)
            threshold_met = overall_score >= self.compliance_threshold

        # Calculate estimates and critical path
        total_hours = sum(task.estimated_hours for task in structured_tasks)
        critical_path = self._calculate_critical_path(structured_tasks)

        return TasksResponse(
            success=True,
            tasks_breakdown=tasks_breakdown,
            tasks_path=str(tasks_file),
            tasks=structured_tasks,
            constitutional_compliance=constitutional_compliance,
            overall_compliance_score=overall_score,
            compliance_threshold_met=threshold_met,
            estimated_total_hours=total_hours,
            critical_path=critical_path,
            next_steps=self._get_tasks_next_steps(threshold_met),
            timestamp=datetime.now(),
        )

    def _generate_feature_id(self, _user_input: str) -> str:
        """Generate a unique feature ID."""
        # Get next sequential number
        existing_features = [
            d for d in self.specs_dir.iterdir() if d.is_dir() and d.name[:3].isdigit()
        ]
        next_num = len(existing_features) + 1
        return f"{next_num:03d}"

    def _slugify(self, text: str) -> str:
        """Convert text to URL-safe slug."""
        import re

        slug = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[-\s]+", "-", slug)
        return slug.strip("-")[:50]

    def _generate_specification(self, user_input: str, context: dict[str, Any]) -> str:
        """Generate specification content."""
        return f"""# Feature Specification

## Overview
{user_input}

## Context
{context.get('description', 'Additional context not provided')}

## Functional Requirements
- [ ] Requirement 1: Core functionality
- [ ] Requirement 2: User interface
- [ ] Requirement 3: Data persistence

## Non-Functional Requirements
- [ ] Performance: Response time < 200ms
- [ ] Security: Authentication and authorization
- [ ] Scalability: Support 1000+ concurrent users

## Acceptance Criteria
- [ ] All functional requirements implemented
- [ ] All tests passing (minimum 80% coverage)
- [ ] Documentation complete

## Constitutional Compliance
- [ ] Library-First: Reuses existing libraries where possible
- [ ] Test-First: Tests written before implementation
- [ ] Simplicity: Maintains low complexity
- [ ] Integration-First: Real environment testing
- [ ] Clarity: Clear and unambiguous requirements
- [ ] Counterfactual: Decision rationale documented

## API Contracts
TBD - Define API endpoints and data models

## Review Checklist
- [ ] Specification reviewed for clarity
- [ ] Requirements validated with stakeholders
- [ ] Constitutional compliance verified
- [ ] Ready for planning phase

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def _generate_implementation_plan(
        self, _specification: str, tech_stack: str, _constraints: dict[str, Any]
    ) -> str:
        """Generate implementation plan."""
        return f"""# Implementation Plan

## Architecture Overview
Based on the specification, this implementation follows constitutional principles.

## Technology Stack
{tech_stack or 'To be determined based on requirements'}

## Project Structure
Following the Simplicity Gate (≤3 projects):
1. Core Library
2. API/CLI Interface
3. Tests and Documentation

## Implementation Phases

### Phase 1: Foundation (Test-First)
- [ ] Set up test infrastructure
- [ ] Implement core data models
- [ ] Create basic API structure

### Phase 2: Core Features
- [ ] Implement main functionality
- [ ] Add CLI interface (Constitutional requirement)
- [ ] Integration testing with real environments

### Phase 3: Polish & Documentation
- [ ] Performance optimization
- [ ] Complete documentation
- [ ] Final constitutional review

## Constitutional Compliance Strategy
- **Library-First**: Research existing solutions before implementation
- **Test-First**: 80% minimum test coverage, TDD workflow
- **Simplicity**: Keep cyclomatic complexity < 10
- **Integration-First**: Test with real data and environments
- **Clarity**: Comprehensive documentation and clear naming
- **Counterfactual**: Document architectural decisions

## Risk Mitigation
- Technical risks identified and mitigated
- Dependencies evaluated for constitutional compliance
- Complexity kept minimal per Simplicity Gate

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def _generate_task_breakdown(
        self, _implementation_plan: str, priority_focus: str, team_size: int
    ) -> str:
        """Generate task breakdown."""
        return f"""# Task Breakdown

## Priority Focus: {priority_focus.title()}

## Epic 1: Foundation & Infrastructure
### Task 1.1: Test Infrastructure Setup
- **Priority**: Critical
- **Estimated Hours**: 8
- **Dependencies**: None
- **Description**: Set up pytest, coverage, CI/CD pipeline
- **Acceptance Criteria**:
  - [ ] Test framework configured
  - [ ] Coverage reporting enabled
  - [ ] CI/CD pipeline functional

### Task 1.2: Core Data Models
- **Priority**: Critical
- **Estimated Hours**: 12
- **Dependencies**: 1.1
- **Description**: Implement core data structures with tests first
- **Acceptance Criteria**:
  - [ ] All models tested (TDD)
  - [ ] Validation logic complete
  - [ ] Documentation generated

## Epic 2: Core Implementation
### Task 2.1: Business Logic
- **Priority**: High
- **Estimated Hours**: 20
- **Dependencies**: 1.2
- **Description**: Implement main feature functionality
- **Acceptance Criteria**:
  - [ ] All requirements implemented
  - [ ] Test coverage ≥ 80%
  - [ ] Performance benchmarks met

### Task 2.2: CLI Interface
- **Priority**: High
- **Estimated Hours**: 10
- **Dependencies**: 2.1
- **Description**: Constitutional requirement for CLI interface
- **Acceptance Criteria**:
  - [ ] Text-in, text-out interface
  - [ ] Help documentation complete
  - [ ] Error handling robust

## Epic 3: Integration & Quality
### Task 3.1: Integration Testing
- **Priority**: Medium
- **Estimated Hours**: 16
- **Dependencies**: 2.2
- **Description**: Real environment testing (not mocks)
- **Acceptance Criteria**:
  - [ ] End-to-end workflows tested
  - [ ] Real data integration verified
  - [ ] Performance under load validated

### Task 3.2: Documentation & Review
- **Priority**: Medium
- **Estimated Hours**: 12
- **Dependencies**: 3.1
- **Description**: Complete documentation and constitutional review
- **Acceptance Criteria**:
  - [ ] All documentation complete
  - [ ] Constitutional compliance verified
  - [ ] Ready for deployment

## Team Size Optimization
Team Size: {team_size} developer(s)
Estimated Timeline: {78 // team_size} days (assuming 6 hours/day)

## Constitutional Requirements per Task
- All tasks must maintain constitutional compliance
- Test-first development required
- Library-first research before custom implementation
- Integration testing with real environments
- Clear documentation and decision rationale

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    def _validate_specification(
        self, specification: str
    ) -> dict[str, ConstitutionalValidation]:
        """Validate specification against constitutional articles."""
        # Use the constitutional scorer
        result = self.constitutional_scorer.score_specification(specification)

        validations = {}
        for violation in result.violations:
            article = violation.article
            if article not in validations:
                validations[article] = ConstitutionalValidation(
                    article=article,
                    compliant=False,
                    score=0.5,
                    violations=[],
                    suggestions=[],
                )
            validations[article].violations.append(violation.message)
            if violation.suggestion:
                validations[article].suggestions.append(violation.suggestion)

        # Add passing articles
        all_articles = [
            "Article I",
            "Article II",
            "Article III",
            "Article IV",
            "Article V",
            "Article VI",
        ]
        for article in all_articles:
            if article not in validations:
                validations[article] = ConstitutionalValidation(
                    article=article,
                    compliant=True,
                    score=result.overall_score,
                    violations=[],
                    suggestions=[],
                )

        return validations

    def _validate_implementation_plan(
        self, implementation_plan: str
    ) -> dict[str, ConstitutionalValidation]:
        """Validate implementation plan against constitutional articles."""
        return self._validate_specification(implementation_plan)

    def _validate_task_breakdown(
        self, tasks_breakdown: str
    ) -> dict[str, ConstitutionalValidation]:
        """Validate task breakdown against constitutional articles."""
        return self._validate_specification(tasks_breakdown)

    def _calculate_overall_score(
        self, validations: dict[str, ConstitutionalValidation]
    ) -> float:
        """Calculate overall constitutional compliance score."""
        if not validations:
            return 1.0

        total_score = sum(v.score for v in validations.values())
        return total_score / len(validations)

    def _get_specify_next_steps(self, threshold_met: bool) -> list[str]:
        """Get next steps for specify phase."""
        if threshold_met:
            return [
                "Review and refine the specification",
                "Validate requirements with stakeholders",
                "Run /plan to create implementation plan",
            ]
        else:
            return [
                "Address constitutional violations",
                "Improve specification clarity",
                "Re-run /specify after corrections",
            ]

    def _get_plan_next_steps(self, threshold_met: bool) -> list[str]:
        """Get next steps for plan phase."""
        if threshold_met:
            return [
                "Review implementation plan",
                "Validate technology choices",
                "Run /tasks to generate task breakdown",
            ]
        else:
            return [
                "Address constitutional violations in plan",
                "Simplify architecture if needed",
                "Re-run /plan after corrections",
            ]

    def _get_tasks_next_steps(self, threshold_met: bool) -> list[str]:
        """Get next steps for tasks phase."""
        if threshold_met:
            return [
                "Begin implementation with test-first approach",
                "Start with highest priority tasks",
                "Maintain constitutional compliance throughout",
            ]
        else:
            return [
                "Address constitutional violations in tasks",
                "Ensure test-first approach in all tasks",
                "Re-run /tasks after corrections",
            ]

    def _generate_supporting_documents(
        self, feature_dir: Path, _specification: str, _tech_stack: str
    ) -> list[str]:
        """Generate supporting documents."""
        docs = []

        # API contract
        api_file = feature_dir / "api-contract.md"
        api_content = f"""# API Contract

## Endpoints
TBD based on requirements

## Data Models
TBD based on specification

---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        api_file.write_text(api_content, encoding="utf-8")
        docs.append(str(api_file))

        return docs

    def _get_tech_recommendations(self, _current_stack: str) -> list[str]:
        """Get technology recommendations."""
        return [
            "Use established libraries (Library-First principle)",
            "Prefer simple, well-documented tools",
            "Ensure CLI interface capability",
            "Choose tools with good testing support",
        ]

    def _extract_architecture_decisions(self, _plan: str) -> list[str]:
        """Extract architecture decisions from plan."""
        return [
            "Modular architecture for simplicity",
            "Test-first development approach",
            "CLI interface for constitutional compliance",
            "Integration testing with real environments",
        ]

    def _parse_structured_tasks(self, _tasks_content: str) -> list[TaskBreakdown]:
        """Parse tasks content into structured format."""
        # Simple parsing - in production this would be more sophisticated
        tasks = [
            TaskBreakdown(
                id="1.1",
                title="Test Infrastructure Setup",
                description="Set up pytest, coverage, CI/CD pipeline",
                priority="critical",
                estimated_hours=8,
                dependencies=[],
                acceptance_criteria=[
                    "Test framework configured",
                    "Coverage reporting enabled",
                    "CI/CD pipeline functional",
                ],
                constitutional_requirements=["Test-First development"],
            ),
            TaskBreakdown(
                id="1.2",
                title="Core Data Models",
                description="Implement core data structures with tests first",
                priority="critical",
                estimated_hours=12,
                dependencies=["1.1"],
                acceptance_criteria=[
                    "All models tested (TDD)",
                    "Validation logic complete",
                    "Documentation generated",
                ],
                constitutional_requirements=["Test-First", "Clarity"],
            ),
        ]
        return tasks

    def _calculate_critical_path(self, tasks: list[TaskBreakdown]) -> list[str]:
        """Calculate critical path through tasks."""
        # Simple implementation - in production would use proper scheduling
        return [task.id for task in tasks if task.priority == "critical"]
