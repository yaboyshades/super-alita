"""Constitutional SDD Pipeline.

Implements the Specification-Driven Development workflow with integrated
constitutional validation at each stage: specify, plan, and tasks.
"""

from dataclasses import dataclass
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


@dataclass
class AgentTaskDefinition:
    """Static metadata for a delegated agent task."""

    id: str
    agent: str
    title: str
    priority: str
    estimated_hours: int
    dependencies: list[str]
    context: list[str]
    responsibilities: list[str]
    deliverables: list[str]
    acceptance_criteria: list[str]
    constitutional_requirements: list[str]


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

        # Generate task definitions and markdown breakdown aligned to agents
        task_definitions = self._get_agent_task_definitions(
            request.priority_focus, request.team_size
        )
        tasks_breakdown = self._generate_task_breakdown(
            request.priority_focus,
            request.team_size,
            task_definitions,
        )

        # Write tasks file
        tasks_file = plan_path.parent / "tasks.md"
        tasks_file.write_text(tasks_breakdown, encoding="utf-8")

        # Parse structured tasks
        structured_tasks = self._build_structured_tasks(task_definitions)

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

    def _priority_focus_details(self, priority_focus: str) -> dict[str, Any]:
        """Return narrative guidance for the requested priority focus."""

        focus_map: dict[str, dict[str, Any]] = {
            "test-first": {
                "summary": (
                    "Lead with failing tests, enforce ≥70% coverage, and gate merges "
                    "on CI reliability."
                ),
                "guardrails": [
                    "Codify property-based tests for deterministic components.",
                    "Extend regression suites before modifying runtime atoms.",
                    "Instrument CI to fail fast when coverage regresses.",
                ],
            },
            "library-first": {
                "summary": (
                    "Prefer existing battle-tested libraries before authoring new "
                    "code paths."
                ),
                "guardrails": [
                    "Evaluate ecosystem options and capture reuse decisions.",
                    "Document licensing and maintenance posture for dependencies.",
                    "Avoid custom forks unless a mitigation plan exists.",
                ],
            },
            "integration-first": {
                "summary": (
                    "Optimize for seamless runtime integration, streaming fidelity, "
                    "and telemetry completeness."
                ),
                "guardrails": [
                    "Validate streaming contracts end-to-end.",
                    "Test load/backpressure scenarios on the EventBus.",
                    "Ensure telemetry atoms bond into the knowledge graph.",
                ],
            },
        }

        return focus_map.get(priority_focus.lower(), focus_map["test-first"])

    def _get_agent_task_definitions(
        self, priority_focus: str, team_size: int
    ) -> list[AgentTaskDefinition]:
        """Create canonical tasks for each specialized agent."""

        focus_lower = priority_focus.lower()
        focus_emphasis = {
            "test-first": (
                "Embed failing tests and instrumentation before feature code."
            ),
            "library-first": (
                "Inventory reusable libraries and only author code when no "
                "hardened option exists."
            ),
            "integration-first": (
                "Prove integration points and streaming flows before expanding "
                "feature scope."
            ),
        }.get(focus_lower, "Embed failing tests and instrumentation before feature code.")

        team_context = (
            f"Coordinate across {team_size} engineer(s) while preserving the closed-loop "
            "cognitive model (Event → Atom/Bond → Energy → TODO → Bandit → Reward)."
        )

        return [
            AgentTaskDefinition(
                id="ARCH-001",
                agent="Architecture Agent",
                title="SDD specification & decomposition refresh",
                priority="critical",
                estimated_hours=6,
                dependencies=[],
                context=[
                    team_context,
                    "Operate in code mode with production constraints and streaming telemetry maintained.",
                    "Leverage the existing implementation plan and PATCHMAP.md guidance to prevent drift.",
                ],
                responsibilities=[
                    "Regenerate or update the SDD specification via /specify to reflect current feature goals.",
                    "Break work into atomic tasks mapped to agent specializations with dependency graphing.",
                    "Document risk and fallback strategies for orchestration changes.",
                ],
                deliverables=[
                    "Updated spec.md and architecture decision records.",
                    "JSON task graph hand-off for downstream agents.",
                    "Risk register highlighting integration and reliability concerns.",
                ],
                acceptance_criteria=[
                    "Specification validated by constitutional scorer ≥ 0.75.",
                    "Task graph references streaming contract checkpoints.",
                    "Risks include mitigation owners and timelines.",
                ],
                constitutional_requirements=[
                    "Article I – Library-First Planning",
                    "Article III – Simplicity Gate",
                    "Article V – Clarity & Unambiguity",
                ],
            ),
            AgentTaskDefinition(
                id="SEC-001",
                agent="Security Agent",
                title="Runtime threat modeling & sandbox policy",
                priority="critical",
                estimated_hours=5,
                dependencies=["ARCH-001"],
                context=[
                    team_context,
                    "Assess sandboxed execution paths, especially reug_runtime.router tooling.",
                    "Ensure env-var based credential handling; forbid shell=True subprocess usage.",
                ],
                responsibilities=[
                    "Perform threat modeling on new orchestration and tool delegation flows.",
                    "Define sandbox guardrails, resource limits, and monitoring hooks.",
                    "Review dependency list for CVEs and license posture.",
                ],
                deliverables=[
                    "Security assessment report with residual risk scoring.",
                    "Sandbox policy updates or confirmations in src/sandbox/ exec tooling.",
                    "Dependency vulnerability scan summary with action items.",
                ],
                acceptance_criteria=[
                    "All dynamic execution paths mapped with mitigations.",
                    "Resource and retry limits documented via REUG_* toggles.",
                    "No high/critical unresolved CVEs remain.",
                ],
                constitutional_requirements=[
                    "Article IV – Integration Reliability",
                    "Article VI – Counterfactual Safety",
                ],
            ),
            AgentTaskDefinition(
                id="IMPL-001",
                agent="Implementation Agent",
                title="Production-grade feature implementation",
                priority="critical",
                estimated_hours=14,
                dependencies=["ARCH-001", "SEC-001"],
                context=[
                    team_context,
                    focus_emphasis,
                    "Maintain streaming response contract (<tool_call>, <tool_result>, <final_answer>).",
                ],
                responsibilities=[
                    "Author modular, testable code that wires into reug_runtime.router and telemetry emissions.",
                    "Respect sandbox policies and leverage src/core/proc.py for subprocess orchestration.",
                    "Instrument feature code with atoms/bonds to propagate energy and TODO scoring.",
                ],
                deliverables=[
                    "Production-ready Python modules under src/ with absolute imports.",
                    "Configuration updates or feature toggles scoped to new behavior.",
                    "Inline docs and type hints supporting mypy --strict.",
                ],
                acceptance_criteria=[
                    "Feature passes pytest focus suites with new tests.",
                    "Mypy --strict and ruff check show no regressions.",
                    "Energy propagation and telemetry hooks verified via local runs.",
                ],
                constitutional_requirements=[
                    "Article II – Test-First Development",
                    "Article IV – Integration Reliability",
                ],
            ),
            AgentTaskDefinition(
                id="TEST-001",
                agent="Testing Agent",
                title="Coverage, regression, and stress harness",
                priority="critical",
                estimated_hours=10,
                dependencies=["IMPL-001"],
                context=[
                    team_context,
                    "Use tests/runtime/ fakes for deterministic coverage; gate integration tests appropriately.",
                    "Target ≥70% coverage and include property-based scenarios where applicable.",
                ],
                responsibilities=[
                    "Design unit, regression, and stress tests covering new runtime paths.",
                    "Validate streaming transcripts and telemetry atoms for new events.",
                    "Ensure CI commands (pytest, pre-commit) remain green.",
                ],
                deliverables=[
                    "New or updated pytest modules under tests/ mirroring src structure.",
                    "Stress harness scripts or marks for load-sensitive flows.",
                    "Coverage report diff demonstrating thresholds maintained.",
                ],
                acceptance_criteria=[
                    "Pytest suites pass locally and in CI tasks.",
                    "Coverage ≥ 70% for touched modules.",
                    "Stress harness shows no dropped events or backpressure failures.",
                ],
                constitutional_requirements=[
                    "Article II – Test-First Development",
                    "Article IV – Integration Reliability",
                ],
            ),
            AgentTaskDefinition(
                id="DOC-001",
                agent="Documentation Agent",
                title="Operational & API documentation updates",
                priority="high",
                estimated_hours=6,
                dependencies=["IMPL-001", "TEST-001"],
                context=[
                    team_context,
                    "Document runtime toggles, telemetry schemas, and agent delegation rationale.",
                    "Ensure docs reflect updated closed-loop cognitive operations.",
                ],
                responsibilities=[
                    "Update docs/, PATCHMAP.md, and AGENTS.md where scope changed.",
                    "Capture API and CLI changes introduced by implementation.",
                    "Record testing strategy and rollback playbooks.",
                ],
                deliverables=[
                    "Doc updates with change logs and operator guidance.",
                    "Revised onboarding instructions for agent delegation.",
                    "Rollback and verification checklist entries.",
                ],
                acceptance_criteria=[
                    "Docs reviewed for clarity and completeness.",
                    "Operational steps align with integration agent runbooks.",
                    "Rollback procedure validated against implementation reality.",
                ],
                constitutional_requirements=[
                    "Article V – Clarity & Unambiguity",
                    "Article VI – Counterfactual Safety",
                ],
            ),
            AgentTaskDefinition(
                id="INT-001",
                agent="Integration Agent",
                title="CI/CD & environment orchestration",
                priority="high",
                estimated_hours=8,
                dependencies=["IMPL-001", "TEST-001"],
                context=[
                    team_context,
                    "Maintain VS Code tasks, CI pipelines, and deployment scripts.",
                    "Verify toggles: REUG_MAX_TOOL_CALLS, REUG_EXEC_TIMEOUT_S, retry configuration.",
                ],
                responsibilities=[
                    "Update CI workflows to run new lint/test gates and publish telemetry artifacts.",
                    "Validate environment parity across local, staging, and production.",
                    "Instrument runtime metrics for latency, retries, and schema enforcement.",
                ],
                deliverables=[
                    "CI/CD configuration diffs with verification steps.",
                    "Environment validation scripts and task definitions.",
                    "Metrics dashboards or log aggregation queries for new events.",
                ],
                acceptance_criteria=[
                    "Pipelines execute successfully with new stages.",
                    "Environment validation scripts pass with documented outputs.",
                    "Runtime telemetry confirms schema adherence and acceptable latency.",
                ],
                constitutional_requirements=[
                    "Article IV – Integration Reliability",
                    "Article VI – Counterfactual Safety",
                ],
            ),
            AgentTaskDefinition(
                id="VAL-001",
                agent="Validation Agent",
                title="Constitutional & quality gate enforcement",
                priority="medium",
                estimated_hours=6,
                dependencies=["IMPL-001", "TEST-001", "INT-001"],
                context=[
                    team_context,
                    "Aggregate compliance metrics (mypy --strict, ruff, pytest, coverage, benchmarks).",
                    "Ensure Mangle-based constitutional checks and knowledge graph updates succeed.",
                ],
                responsibilities=[
                    "Run constitutional scorers and enhanced validation via Mangle integration.",
                    "Certify quality gates: lint, type-check, tests, performance, documentation completeness.",
                    "Coordinate rollback readiness and release notes approvals.",
                ],
                deliverables=[
                    "Validation report with compliance scoring and outstanding actions.",
                    "Performance benchmark results with pass/fail annotations.",
                    "Final release go/no-go recommendation with rollback trigger conditions.",
                ],
                acceptance_criteria=[
                    "Constitutional compliance ≥ 0.75 with no critical violations.",
                    "All quality gates documented with evidence artifacts.",
                    "Rollback procedure rehearsed or simulated.",
                ],
                constitutional_requirements=[
                    "Article II – Test-First Development",
                    "Article IV – Integration Reliability",
                    "Article VI – Counterfactual Safety",
                ],
            ),
        ]

    def _generate_task_breakdown(
        self,
        priority_focus: str,
        team_size: int,
        task_definitions: list[AgentTaskDefinition],
    ) -> str:
        """Generate an agent-aligned task breakdown document."""

        focus_details = self._priority_focus_details(priority_focus)

        lines: list[str] = [
            "# Agent Task Assignments",
            "",
            f"**Priority Focus:** {priority_focus.title()}",
            f"**Team Size:** {team_size} engineer(s)",
            "",
            "## Priority Guidance",
            "",
            f"- {focus_details['summary']}",
            "",
            "### Guardrails",
        ]

        for guardrail in focus_details["guardrails"]:
            lines.append(f"- {guardrail}")

        lines.extend([
            "",
            "## Agent Task Matrix",
            "",
        ])

        for definition in task_definitions:
            dependency_label = (
                ", ".join(definition.dependencies)
                if definition.dependencies
                else "None"
            )
            lines.extend(
                [
                    f"### {definition.agent} — {definition.title} ({definition.id})",
                    (
                        f"- **Priority:** {definition.priority.title()} | **Estimated Effort:** "
                        f"{definition.estimated_hours}h | **Dependencies:** {dependency_label}"
                    ),
                    "",
                    "**Context**",
                ]
            )

            for item in definition.context:
                lines.append(f"- {item}")

            lines.extend(["", "**Responsibilities**"])

            for item in definition.responsibilities:
                lines.append(f"- {item}")

            lines.extend(["", "**Expected Deliverables**"])

            for item in definition.deliverables:
                lines.append(f"- {item}")

            lines.extend(["", "**Acceptance Criteria**"])

            for item in definition.acceptance_criteria:
                lines.append(f"- [ ] {item}")

            lines.extend(["", "**Constitutional Alignment**"])

            for item in definition.constitutional_requirements:
                lines.append(f"- {item}")

            lines.append("")

        lines.extend(
            [
                "---",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ]
        )

        return "\n".join(lines)

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

    def _build_structured_tasks(
        self, task_definitions: list[AgentTaskDefinition]
    ) -> list[TaskBreakdown]:
        """Convert agent definitions into structured TaskBreakdown objects."""

        structured_tasks: list[TaskBreakdown] = []
        for definition in task_definitions:
            description_parts = [
                "Context: " + "; ".join(definition.context),
                "Responsibilities: " + "; ".join(definition.responsibilities),
                "Deliverables: " + "; ".join(definition.deliverables),
            ]

            structured_tasks.append(
                TaskBreakdown(
                    id=definition.id,
                    title=definition.title,
                    description="\n".join(description_parts),
                    priority=definition.priority,
                    estimated_hours=definition.estimated_hours,
                    dependencies=definition.dependencies,
                    acceptance_criteria=definition.acceptance_criteria,
                    constitutional_requirements=definition.constitutional_requirements,
                )
            )

        return structured_tasks

    def _calculate_critical_path(self, tasks: list[TaskBreakdown]) -> list[str]:
        """Calculate critical path through tasks."""
        # Simple implementation - in production would use proper scheduling
        return [task.id for task in tasks if task.priority == "critical"]
