"""
Spec-Driven Development (SDD) Configuration
Integrates Spec Kit methodology into Super-Alita Agent Mode
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SDDConfig:
    """Configuration for Spec-Driven Development workflow."""

    # Core SDD workflow phases
    phases: list[str] = field(
        default_factory=lambda: [
            "specify",
            "plan",
            "tasks",
            "implement",
            "validate",
        ]
    )

    # Template locations
    templates_dir: Path = field(default_factory=lambda: Path("templates/sdd"))
    specs_dir: Path = field(default_factory=lambda: Path("specs"))
    memory_dir: Path = field(default_factory=lambda: Path("memory"))

    # Constitutional framework integration
    constitutional_validation: bool = True
    constitutional_threshold: float = 0.75

    # Feature branching
    auto_branch_creation: bool = True
    branch_prefix: str = "sdd"

    # Validation settings
    require_acceptance_checklist: bool = True
    min_user_stories: int = 3
    max_spec_iterations: int = 5

    # Note: Defaults are provided via default_factory; no post-init needed.


@dataclass
class SDDCommand:
    """Definition for SDD workflow commands."""

    name: str
    description: str
    phase: str
    template: str | None = None
    validation_rules: list[str] = field(default_factory=list)
    constitutional_gates: list[str] = field(default_factory=list)


# Default SDD Commands Configuration
DEFAULT_SDD_COMMANDS = {
    "specify": SDDCommand(
        name="specify",
        description="Create functional specification focusing on WHAT and WHY, not tech stack",
        phase="specify",
        template="spec-template.md",
        validation_rules=[
            "must_define_user_stories",
            "must_define_acceptance_criteria",
            "no_technical_implementation_details",
            "minimum_3_user_stories",
        ],
        constitutional_gates=[
            "Library_First_Compliance",
            "Clarity_Unambiguity",
            "Simplicity_Gate",
        ],
    ),
    "plan": SDDCommand(
        name="plan",
        description="Generate technical implementation plan with tech stack and architecture choices",
        phase="plan",
        template="plan-template.md",
        validation_rules=[
            "must_define_tech_stack",
            "must_define_architecture",
            "must_reference_spec",
            "constitutional_compliance_check",
        ],
        constitutional_gates=[
            "Test_First_Requirements",
            "Integration_First_Testing",
            "Constitutional_Alignment",
        ],
    ),
    "tasks": SDDCommand(
        name="tasks",
        description="Break down implementation into actionable tasks with clear dependencies",
        phase="tasks",
        template="tasks-template.md",
        validation_rules=[
            "must_define_task_dependencies",
            "must_estimate_effort",
            "must_define_acceptance_criteria",
            "constitutional_task_validation",
        ],
        constitutional_gates=[
            "Counterfactual_Justification",
            "Integration_First_Testing",
            "Constitutional_Compliance",
        ],
    ),
}


# SDD Workflow Validation Rules
SDD_VALIDATION_RULES = {
    "must_define_user_stories": {
        "description": "Specification must contain at least 3 user stories",
        "pattern": r"(?i)(?:user story|as a|i want|so that)",
        "min_matches": 3,
    },
    "must_define_acceptance_criteria": {
        "description": "Each user story must have acceptance criteria",
        "pattern": r"(?i)(?:acceptance criteria|given|when|then)",
        "min_matches": 1,
    },
    "no_technical_implementation_details": {
        "description": "Specification should focus on WHAT, not HOW",
        "forbidden_patterns": [
            r"(?i)(?:react|vue|angular|django|flask)",
            r"(?i)(?:database|sql|nosql|mongodb)",
            r"(?i)(?:api|rest|graphql|microservice)",
        ],
    },
    "must_define_tech_stack": {
        "description": "Plan must clearly define technology choices",
        "required_sections": ["Technology Stack", "Architecture", "Dependencies"],
    },
    "constitutional_compliance_check": {
        "description": "All artifacts must pass constitutional compliance threshold",
        "min_score": 0.75,
        "articles": [
            "Library_First_Development",
            "Test_First_Development",
            "Simplicity_Gate",
            "Integration_First_Testing",
            "Clarity_Unambiguity",
            "Counterfactual_Justification",
        ],
    },
}


# SDD Phase Dependencies
SDD_PHASE_DEPENDENCIES = {
    "specify": [],
    "plan": ["specify"],
    "tasks": ["specify", "plan"],
    "implement": ["specify", "plan", "tasks"],
    "validate": ["specify", "plan", "tasks", "implement"],
}


# Constitutional Integration Points
CONSTITUTIONAL_SDD_INTEGRATION = {
    "specify_phase": {
        "pre_hooks": ["validate_input_clarity"],
        "post_hooks": ["score_specification_quality", "check_library_first_mindset"],
        "scoring_criteria": [
            "problem_definition_clarity",
            "user_story_completeness",
            "acceptance_criteria_precision",
            "scope_boundary_definition",
        ],
    },
    "plan_phase": {
        "pre_hooks": ["validate_spec_exists", "check_constitutional_compliance"],
        "post_hooks": ["validate_architecture_choices", "verify_test_first_approach"],
        "scoring_criteria": [
            "library_first_evidence",
            "test_strategy_definition",
            "integration_approach_clarity",
            "simplicity_adherence",
        ],
    },
    "tasks_phase": {
        "pre_hooks": ["validate_plan_exists", "check_dependency_resolution"],
        "post_hooks": ["validate_task_breakdown", "verify_estimation_quality"],
        "scoring_criteria": [
            "task_granularity_appropriateness",
            "dependency_identification",
            "effort_estimation_realism",
            "acceptance_criteria_testability",
        ],
    },
}
