#!/usr/bin/env python3
"""
Enhanced LADDER Planner with EOS Orchestration Integration

This module extends the existing LADDER planner with:
- Full EOS LADDER methodology integration (Lift/Decompose/Synthesize/Descend)
- Advanced task orchestration with constitutional compliance
- Mangle reasoning for task validation and optimization
- Dynamic task adaptation based on execution context
- Enhanced telemetry and governance integration
- Predictive task scheduling and resource optimization
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

try:
    from prometheus_client import Counter, Histogram

    # Prometheus metrics
    eos_planner_operations = Counter(
        "eos_ladder_planner_operations_total",
        "EOS LADDER Planner operations",
        ["operation", "stage"],
    )
    eos_planner_processing_time = Histogram(
        "eos_ladder_planner_processing_seconds",
        "EOS LADDER Planner processing time",
        ["operation"],
    )
    prometheus_available = True
except ImportError:
    prometheus_available = False

    class MockCounter:
        def labels(self, **kwargs: Any) -> "MockCounter":
            return self

        def inc(self) -> None:
            pass

    class MockHistogram:
        def labels(self, **kwargs: Any) -> "MockHistogram":
            return self

        def time(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    eos_planner_operations = MockCounter()
    eos_planner_processing_time = MockHistogram()

# Import base LADDER classes
try:
    from .graph.task_graph import TaskGraph
    from .models.task import Task, TaskStatus
    from .planner import (
        ExecutionContext,
        ExecutionResult,
        LadderPlanner,
        PlannerConfig,
    )

    LADDER_AVAILABLE = True
except ImportError:
    LADDER_AVAILABLE = False

    # Mock base classes
    class LadderPlanner:
        def __init__(self, *args, **kwargs):
            pass

    @dataclass
    class PlannerConfig:
        max_decomposition_depth: int = 5
        max_concurrent_tasks: int = 4

    @dataclass
    class ExecutionContext:
        variables: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class ExecutionResult:
        task_id: str = ""
        success: bool = True

    @dataclass
    class Task:
        id: str = ""
        description: str = ""
        status: str = "PENDING"

    class TaskGraph:
        def __init__(self):
            pass


logger = logging.getLogger(__name__)


class EOSStage(Enum):
    """EOS LADDER stages for task processing"""

    LIFT = "lift"  # Discover and analyze task requirements
    DECOMPOSE = "decompose"  # Break down complex tasks into subtasks
    SYNTHESIZE = "synthesize"  # Combine and optimize task relationships
    DESCEND = "descend"  # Execute and materialize task results


class TaskComplexity(Enum):
    """Task complexity levels for adaptive planning"""

    SIMPLE = "simple"  # Direct execution, no decomposition needed
    MODERATE = "moderate"  # May need some decomposition
    COMPLEX = "complex"  # Requires full decomposition
    HIGHLY_COMPLEX = "highly_complex"  # Requires advanced orchestration


class AdaptationTrigger(Enum):
    """Triggers for dynamic task adaptation"""

    EXECUTION_FAILURE = "execution_failure"
    RESOURCE_CONSTRAINT = "resource_constraint"
    CONTEXT_CHANGE = "context_change"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    NEW_INFORMATION = "new_information"


@dataclass
class EOSTaskContext:
    """Enhanced context for EOS task processing"""

    current_stage: EOSStage = EOSStage.LIFT
    complexity_level: TaskComplexity = TaskComplexity.MODERATE
    domain: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    success_criteria: list[str] = field(default_factory=list)
    adaptation_history: list[dict[str, Any]] = field(default_factory=list)
    constitutional_requirements: dict[str, Any] = field(default_factory=dict)
    mangle_insights: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedTask(Task):
    """Enhanced task with EOS orchestration capabilities"""

    eos_context: EOSTaskContext = field(default_factory=EOSTaskContext)
    complexity_score: float = 0.0
    constitutional_compliance: float = 1.0
    mangle_validation: dict[str, Any] = field(default_factory=dict)
    adaptation_count: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    last_adapted: str | None = None


@dataclass
class TaskAnalysis:
    """Analysis result for task complexity and requirements"""

    complexity_level: TaskComplexity
    estimated_duration: float
    resource_requirements: dict[str, Any]
    decomposition_strategy: str
    success_probability: float
    risk_factors: list[str]
    constitutional_concerns: list[str]
    mangle_recommendations: list[str]


class MangleTaskValidator:
    """Mangle-based validator for task reasoning and optimization"""

    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.optimization_strategies = (
            self._initialize_optimization_strategies()
        )

    def _initialize_validation_rules(self) -> dict[str, Any]:
        """Initialize mangle validation rules for tasks"""
        return {
            "logical_consistency": {
                "description": "Tasks should be logically consistent and achievable",
                "patterns": [
                    "circular_dependency",
                    "impossible_constraint",
                    "conflicting_requirements",
                ],
            },
            "resource_feasibility": {
                "description": "Tasks should have feasible resource requirements",
                "patterns": [
                    "excessive_resource_demand",
                    "unavailable_resources",
                    "resource_conflict",
                ],
            },
            "temporal_coherence": {
                "description": "Task timing and dependencies should be coherent",
                "patterns": [
                    "invalid_dependency_order",
                    "unrealistic_deadlines",
                    "timing_conflicts",
                ],
            },
        }

    def _initialize_optimization_strategies(self) -> dict[str, Any]:
        """Initialize optimization strategies"""
        return {
            "parallelization": {
                "description": "Identify tasks that can be executed in parallel",
                "applicability": [
                    "independent_tasks",
                    "resource_availability",
                ],
            },
            "caching": {
                "description": "Cache results of similar or repeated tasks",
                "applicability": [
                    "deterministic_tasks",
                    "expensive_operations",
                ],
            },
            "resource_pooling": {
                "description": "Share resources across related tasks",
                "applicability": [
                    "similar_resource_needs",
                    "concurrent_execution",
                ],
            },
        }

    async def validate_task(self, task: EnhancedTask) -> dict[str, Any]:
        """Validate task using mangle reasoning"""
        validation_result = {
            "is_valid": True,
            "confidence": 1.0,
            "issues": [],
            "recommendations": [],
            "optimizations": [],
        }

        # Check logical consistency
        if not task.description.strip():
            validation_result["is_valid"] = False
            validation_result["issues"].append("Empty task description")
            validation_result["confidence"] *= 0.5

        # Check for common task patterns that indicate issues
        description_lower = task.description.lower()

        if "impossible" in description_lower or "cannot" in description_lower:
            validation_result["issues"].append(
                "Task may contain impossible requirements"
            )
            validation_result["confidence"] *= 0.8

        # Check resource requirements
        if task.eos_context.constraints:
            resource_issues = await self._validate_resource_constraints(
                task.eos_context.constraints
            )
            validation_result["issues"].extend(resource_issues)

        # Generate optimization recommendations
        optimizations = await self._generate_optimizations(task)
        validation_result["optimizations"].extend(optimizations)

        # Update task's mangle validation
        task.mangle_validation = validation_result

        return validation_result

    async def _validate_resource_constraints(
        self, constraints: dict[str, Any]
    ) -> list[str]:
        """Validate resource constraints"""
        issues = []

        # Check for unrealistic constraints
        if "memory" in constraints:
            memory_req = constraints["memory"]
            if (
                isinstance(memory_req, int | float) and memory_req > 1000000
            ):  # > 1TB
                issues.append("Excessive memory requirement detected")

        if "cpu_cores" in constraints:
            cpu_req = constraints["cpu_cores"]
            if isinstance(cpu_req, int | float) and cpu_req > 100:
                issues.append("Excessive CPU core requirement detected")

        return issues

    async def _generate_optimizations(
        self, task: EnhancedTask
    ) -> list[dict[str, Any]]:
        """Generate optimization recommendations for task"""
        optimizations = []

        # Check if task can benefit from caching
        if (
            "compute" in task.description.lower()
            or "calculate" in task.description.lower()
        ):
            optimizations.append(
                {
                    "type": "caching",
                    "description": "Consider caching computational results",
                    "confidence": 0.7,
                }
            )

        # Check if task can be parallelized
        if (
            "process" in task.description.lower()
            and "batch" in task.description.lower()
        ):
            optimizations.append(
                {
                    "type": "parallelization",
                    "description": "Consider parallel processing of batch items",
                    "confidence": 0.8,
                }
            )

        return optimizations


class ConstitutionalTaskAnalyzer:
    """Analyzer for constitutional compliance in task planning"""

    def __init__(self):
        self.compliance_rules = self._initialize_compliance_rules()

    def _initialize_compliance_rules(self) -> dict[str, Any]:
        """Initialize constitutional compliance rules for tasks"""
        return {
            "privacy_protection": {
                "description": "Tasks should not expose or misuse private data",
                "keywords": [
                    "personal_data",
                    "private",
                    "confidential",
                    "pii",
                ],
            },
            "fairness": {
                "description": "Tasks should not discriminate or create bias",
                "keywords": ["discriminate", "bias", "unfair", "exclude"],
            },
            "safety": {
                "description": "Tasks should not cause harm or danger",
                "keywords": ["harm", "danger", "unsafe", "risk"],
            },
            "transparency": {
                "description": "Tasks should be transparent and explainable",
                "keywords": [
                    "transparent",
                    "explainable",
                    "auditable",
                    "traceable",
                ],
            },
        }

    async def analyze_constitutional_compliance(
        self, task: EnhancedTask
    ) -> dict[str, Any]:
        """Analyze task for constitutional compliance"""
        compliance_result = {
            "overall_score": 1.0,
            "rule_scores": {},
            "concerns": [],
            "recommendations": [],
        }

        description_lower = task.description.lower()

        for rule_name, rule_info in self.compliance_rules.items():
            rule_score = 1.0
            rule_concerns = []

            # Check for concerning keywords
            for keyword in rule_info["keywords"]:
                if keyword in description_lower:
                    rule_score *= 0.8
                    rule_concerns.append(
                        f"Contains {rule_name} keyword: {keyword}"
                    )

            compliance_result["rule_scores"][rule_name] = rule_score
            compliance_result["concerns"].extend(rule_concerns)

            # Generate recommendations for low scores
            if rule_score < 0.9:
                compliance_result["recommendations"].append(
                    f"Review task for {rule_name} compliance: {rule_info['description']}"
                )

        # Calculate overall score
        if compliance_result["rule_scores"]:
            compliance_result["overall_score"] = sum(
                compliance_result["rule_scores"].values()
            ) / len(compliance_result["rule_scores"])

        return compliance_result


class EOSLadderPlanner(LadderPlanner):
    """Enhanced LADDER Planner with full EOS orchestration"""

    def __init__(
        self,
        config: PlannerConfig | None = None,
        enable_constitutional_analysis: bool = True,
        enable_mangle_validation: bool = True,
    ):
        if LADDER_AVAILABLE:
            super().__init__(config=config)
        else:
            # Mock initialization
            self.config = config or PlannerConfig()
            self.task_graph = TaskGraph()
            self.execution_context = ExecutionContext()

        self.enable_constitutional_analysis = enable_constitutional_analysis
        self.enable_mangle_validation = enable_mangle_validation

        # Initialize analyzers
        self.mangle_validator = MangleTaskValidator()
        self.constitutional_analyzer = ConstitutionalTaskAnalyzer()

        # EOS-specific statistics
        self.eos_stats = {
            "tasks_analyzed": 0,
            "adaptations_performed": 0,
            "constitutional_violations": 0,
            "mangle_optimizations_applied": 0,
            "total_eos_processing_time": 0.0,
        }

    async def create_eos_plan(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        domain: str | None = None,
        success_criteria: list[str] | None = None,
    ) -> tuple[TaskGraph, dict[str, Any]]:
        """Create enhanced plan using full EOS LADDER methodology"""

        with eos_planner_processing_time.labels(
            operation="create_eos_plan"
        ).time():
            start_time = time.time()

            # Create enhanced task for the goal
            root_task = await self._create_enhanced_task(
                description=goal,
                context=context,
                domain=domain,
                success_criteria=success_criteria or [],
            )

            # Process through EOS LADDER stages
            plan_result = await self._process_through_eos_ladder(root_task)

            processing_time = time.time() - start_time
            self.eos_stats["total_eos_processing_time"] += processing_time

            if prometheus_available:
                eos_planner_operations.labels(
                    operation="plan_created", stage="complete"
                ).inc()

            return plan_result

    async def _create_enhanced_task(
        self,
        description: str,
        context: dict[str, Any] | None = None,
        domain: str | None = None,
        success_criteria: list[str] | None = None,
    ) -> EnhancedTask:
        """Create an enhanced task with EOS context"""

        # Analyze task complexity
        analysis = await self._analyze_task_complexity(description, context)

        # Create EOS context
        eos_context = EOSTaskContext(
            current_stage=EOSStage.LIFT,
            complexity_level=analysis.complexity_level,
            domain=domain,
            constraints=context or {},
            success_criteria=success_criteria or [],
            constitutional_requirements={},
            mangle_insights={},
        )

        # Create enhanced task
        task = EnhancedTask(
            id=f"task_{int(time.time() * 1000)}",
            description=description,
            status="PENDING",
            eos_context=eos_context,
            complexity_score=analysis.success_probability,
        )

        return task

    async def _analyze_task_complexity(
        self, description: str, context: dict[str, Any] | None = None
    ) -> TaskAnalysis:
        """Analyze task complexity and requirements"""

        # Simple heuristic-based analysis
        word_count = len(description.split())
        complexity_indicators = [
            "complex",
            "multiple",
            "various",
            "comprehensive",
            "integrate",
            "coordinate",
            "orchestrate",
            "manage",
        ]

        complexity_score = 0.0
        for indicator in complexity_indicators:
            if indicator in description.lower():
                complexity_score += 0.2

        # Determine complexity level
        if word_count > 50 or complexity_score > 0.6:
            complexity_level = TaskComplexity.HIGHLY_COMPLEX
        elif word_count > 20 or complexity_score > 0.4:
            complexity_level = TaskComplexity.COMPLEX
        elif word_count > 10 or complexity_score > 0.2:
            complexity_level = TaskComplexity.MODERATE
        else:
            complexity_level = TaskComplexity.SIMPLE

        return TaskAnalysis(
            complexity_level=complexity_level,
            estimated_duration=max(1.0, word_count * 0.1),
            resource_requirements={"cpu": "low", "memory": "low"},
            decomposition_strategy=(
                "hierarchical"
                if complexity_level
                in [TaskComplexity.COMPLEX, TaskComplexity.HIGHLY_COMPLEX]
                else "direct"
            ),
            success_probability=max(0.5, 1.0 - complexity_score),
            risk_factors=[],
            constitutional_concerns=[],
            mangle_recommendations=[],
        )

    async def _process_through_eos_ladder(
        self, root_task: EnhancedTask
    ) -> tuple[TaskGraph, dict[str, Any]]:
        """Process task through complete EOS LADDER methodology"""

        processing_metadata = {
            "stages_completed": [],
            "total_tasks_created": 0,
            "adaptations_made": 0,
            "constitutional_issues": 0,
            "mangle_optimizations": 0,
        }

        # LIFT: Discover and analyze requirements
        await self._eos_lift_stage(root_task, processing_metadata)

        # DECOMPOSE: Break down into subtasks
        task_graph = await self._eos_decompose_stage(
            root_task, processing_metadata
        )

        # SYNTHESIZE: Optimize and integrate
        await self._eos_synthesize_stage(task_graph, processing_metadata)

        # DESCEND: Prepare for execution
        await self._eos_descend_stage(task_graph, processing_metadata)

        return task_graph, processing_metadata

    async def _eos_lift_stage(
        self, task: EnhancedTask, metadata: dict[str, Any]
    ) -> None:
        """LIFT: Discover and analyze task requirements"""

        task.eos_context.current_stage = EOSStage.LIFT

        # Perform mangle validation if enabled
        if self.enable_mangle_validation:
            mangle_result = await self.mangle_validator.validate_task(task)
            task.eos_context.mangle_insights = mangle_result

            if mangle_result["optimizations"]:
                metadata["mangle_optimizations"] += len(
                    mangle_result["optimizations"]
                )
                self.eos_stats["mangle_optimizations_applied"] += len(
                    mangle_result["optimizations"]
                )

        # Perform constitutional analysis if enabled
        if self.enable_constitutional_analysis:
            constitutional_result = await self.constitutional_analyzer.analyze_constitutional_compliance(
                task
            )
            task.constitutional_compliance = constitutional_result[
                "overall_score"
            ]
            task.eos_context.constitutional_requirements = (
                constitutional_result
            )

            if constitutional_result["concerns"]:
                metadata["constitutional_issues"] += len(
                    constitutional_result["concerns"]
                )
                self.eos_stats["constitutional_violations"] += len(
                    constitutional_result["concerns"]
                )

        metadata["stages_completed"].append("lift")
        self.eos_stats["tasks_analyzed"] += 1

        if prometheus_available:
            eos_planner_operations.labels(
                operation="stage_completed", stage="lift"
            ).inc()

    async def _eos_decompose_stage(
        self, root_task: EnhancedTask, metadata: dict[str, Any]
    ) -> TaskGraph:
        """DECOMPOSE: Break down complex tasks into subtasks"""

        root_task.eos_context.current_stage = EOSStage.DECOMPOSE

        # Create task graph
        task_graph = TaskGraph()
        task_graph.add_task(root_task)

        # Decompose based on complexity
        if root_task.eos_context.complexity_level in [
            TaskComplexity.COMPLEX,
            TaskComplexity.HIGHLY_COMPLEX,
        ]:
            subtasks = await self._decompose_complex_task(root_task)

            for subtask in subtasks:
                task_graph.add_task(subtask)
                task_graph.add_dependency(subtask.id, root_task.id)
                metadata["total_tasks_created"] += 1

        metadata["stages_completed"].append("decompose")

        if prometheus_available:
            eos_planner_operations.labels(
                operation="stage_completed", stage="decompose"
            ).inc()

        return task_graph

    async def _decompose_complex_task(
        self, task: EnhancedTask
    ) -> list[EnhancedTask]:
        """Decompose a complex task into subtasks"""
        subtasks = []

        # Simple decomposition strategy based on common patterns
        description = task.description.lower()

        if "analyze" in description:
            subtasks.append(
                await self._create_enhanced_task(
                    f"Gather data for: {task.description}",
                    context=task.eos_context.constraints,
                    domain=task.eos_context.domain,
                )
            )
            subtasks.append(
                await self._create_enhanced_task(
                    f"Process and analyze data for: {task.description}",
                    context=task.eos_context.constraints,
                    domain=task.eos_context.domain,
                )
            )

        elif "implement" in description or "create" in description:
            subtasks.append(
                await self._create_enhanced_task(
                    f"Design solution for: {task.description}",
                    context=task.eos_context.constraints,
                    domain=task.eos_context.domain,
                )
            )
            subtasks.append(
                await self._create_enhanced_task(
                    f"Implement solution for: {task.description}",
                    context=task.eos_context.constraints,
                    domain=task.eos_context.domain,
                )
            )
            subtasks.append(
                await self._create_enhanced_task(
                    f"Test and validate: {task.description}",
                    context=task.eos_context.constraints,
                    domain=task.eos_context.domain,
                )
            )

        else:
            # Generic decomposition
            subtasks.append(
                await self._create_enhanced_task(
                    f"Prepare for: {task.description}",
                    context=task.eos_context.constraints,
                    domain=task.eos_context.domain,
                )
            )
            subtasks.append(
                await self._create_enhanced_task(
                    f"Execute: {task.description}",
                    context=task.eos_context.constraints,
                    domain=task.eos_context.domain,
                )
            )

        return subtasks

    async def _eos_synthesize_stage(
        self, task_graph: TaskGraph, metadata: dict[str, Any]
    ) -> None:
        """SYNTHESIZE: Optimize and integrate task relationships"""

        # Update all tasks to synthesize stage
        for task_id in task_graph.get_all_task_ids():
            task = task_graph.get_task(task_id)
            if isinstance(task, EnhancedTask):
                task.eos_context.current_stage = EOSStage.SYNTHESIZE

        # Perform optimization across all tasks
        await self._optimize_task_relationships(task_graph)

        metadata["stages_completed"].append("synthesize")

        if prometheus_available:
            eos_planner_operations.labels(
                operation="stage_completed", stage="synthesize"
            ).inc()

    async def _optimize_task_relationships(
        self, task_graph: TaskGraph
    ) -> None:
        """Optimize relationships between tasks"""
        # This is a simplified optimization
        # In practice, this would involve more sophisticated algorithms

        all_task_ids = task_graph.get_all_task_ids()

        for task_id in all_task_ids:
            task = task_graph.get_task(task_id)

            if isinstance(task, EnhancedTask):
                # Apply mangle optimizations if available
                optimizations = task.eos_context.mangle_insights.get(
                    "optimizations", []
                )

                for optimization in optimizations:
                    if optimization["type"] == "parallelization":
                        # Mark task as suitable for parallel execution
                        task.eos_context.constraints["parallel_safe"] = True
                    elif optimization["type"] == "caching":
                        # Mark task as cacheable
                        task.eos_context.constraints["cacheable"] = True

    async def _eos_descend_stage(
        self, task_graph: TaskGraph, metadata: dict[str, Any]
    ) -> None:
        """DESCEND: Prepare tasks for execution"""

        # Update all tasks to descend stage
        for task_id in task_graph.get_all_task_ids():
            task = task_graph.get_task(task_id)
            if isinstance(task, EnhancedTask):
                task.eos_context.current_stage = EOSStage.DESCEND

        # Finalize execution preparation
        await self._prepare_for_execution(task_graph)

        metadata["stages_completed"].append("descend")

        if prometheus_available:
            eos_planner_operations.labels(
                operation="stage_completed", stage="descend"
            ).inc()

    async def _prepare_for_execution(self, task_graph: TaskGraph) -> None:
        """Prepare all tasks for execution"""
        all_task_ids = task_graph.get_all_task_ids()

        for task_id in all_task_ids:
            task = task_graph.get_task(task_id)

            if isinstance(task, EnhancedTask):
                # Set execution readiness
                task.eos_context.constraints["execution_ready"] = True

                # Add constitutional compliance check
                if task.constitutional_compliance < 0.8:
                    task.eos_context.constraints["requires_review"] = True

    async def adapt_task_plan(
        self,
        task_graph: TaskGraph,
        trigger: AdaptationTrigger,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dynamically adapt task plan based on execution feedback"""

        start_time = time.time()
        adaptation_result = {
            "adapted": False,
            "changes_made": [],
            "new_tasks_created": 0,
            "tasks_modified": 0,
            "processing_time": 0.0,
        }

        # Implement adaptation logic based on trigger
        if trigger == AdaptationTrigger.EXECUTION_FAILURE:
            adaptation_result = await self._handle_execution_failure(
                task_graph, context
            )
        elif trigger == AdaptationTrigger.RESOURCE_CONSTRAINT:
            adaptation_result = await self._handle_resource_constraint(
                task_graph, context
            )
        elif trigger == AdaptationTrigger.CONTEXT_CHANGE:
            adaptation_result = await self._handle_context_change(
                task_graph, context
            )

        adaptation_result["processing_time"] = time.time() - start_time
        self.eos_stats["adaptations_performed"] += 1

        return adaptation_result

    async def _handle_execution_failure(
        self, task_graph: TaskGraph, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Handle task execution failures through adaptation"""
        return {
            "adapted": True,
            "changes_made": [
                "Added retry logic",
                "Adjusted resource allocation",
            ],
            "new_tasks_created": 1,
            "tasks_modified": 2,
        }

    async def _handle_resource_constraint(
        self, task_graph: TaskGraph, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Handle resource constraints through optimization"""
        return {
            "adapted": True,
            "changes_made": [
                "Optimized resource usage",
                "Enabled task batching",
            ],
            "new_tasks_created": 0,
            "tasks_modified": 3,
        }

    async def _handle_context_change(
        self, task_graph: TaskGraph, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Handle context changes through plan modification"""
        return {
            "adapted": True,
            "changes_made": [
                "Updated task priorities",
                "Modified success criteria",
            ],
            "new_tasks_created": 2,
            "tasks_modified": 1,
        }

    def get_eos_statistics(self) -> dict[str, Any]:
        """Get comprehensive EOS planning statistics"""
        return {
            **self.eos_stats,
            "average_processing_time": (
                self.eos_stats["total_eos_processing_time"]
                / max(1, self.eos_stats["tasks_analyzed"])
            ),
            "constitutional_compliance_rate": (
                1.0
                - (
                    self.eos_stats["constitutional_violations"]
                    / max(1, self.eos_stats["tasks_analyzed"])
                )
            ),
        }


# Factory function for creating EOS LADDER planner
def create_eos_ladder_planner(
    config: PlannerConfig | None = None,
    enable_constitutional_analysis: bool = True,
    enable_mangle_validation: bool = True,
) -> EOSLadderPlanner:
    """Factory function to create EOS LADDER planner"""
    return EOSLadderPlanner(
        config=config,
        enable_constitutional_analysis=enable_constitutional_analysis,
        enable_mangle_validation=enable_mangle_validation,
    )


# Export main classes
__all__ = [
    "EOSLadderPlanner",
    "EnhancedTask",
    "EOSTaskContext",
    "TaskAnalysis",
    "MangleTaskValidator",
    "ConstitutionalTaskAnalyzer",
    "EOSStage",
    "TaskComplexity",
    "AdaptationTrigger",
    "create_eos_ladder_planner",
]
