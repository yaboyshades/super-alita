"""
Agent Handoff Protocol for Solo Developer Multi-Agent Orchestration

Manages structured handoffs between agents with context preservation
and task completion verification.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from src.core.events import create_event

logger = logging.getLogger(__name__)


class HandoffStatus(Enum):
    """Status of agent handoff"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TaskStatus(Enum):
    """Status of individual tasks"""

    CREATED = "created"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class TaskContext:
    """Context information for a task"""

    task_id: str
    description: str
    files_modified: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    outputs_generated: list[str] = field(default_factory=list)
    code_changes: dict[str, str] = field(
        default_factory=dict
    )  # file_path -> changes_description
    test_results: dict[str, Any] = field(default_factory=dict)
    documentation_updates: list[str] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    security_considerations: list[str] = field(default_factory=list)
    custom_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentHandoff:
    """Represents a handoff between two agents"""

    handoff_id: str
    source_agent: str
    target_agent: str
    task_id: str
    context: TaskContext
    handoff_message: str
    status: HandoffStatus = HandoffStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    timeout_at: datetime | None = None
    verification_checklist: list[str] = field(default_factory=list)
    completed_verifications: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStep:
    """Single step in a multi-agent workflow"""

    step_id: str
    agent_id: str
    task_description: str
    depends_on: list[str] = field(default_factory=list)
    outputs_required: list[str] = field(default_factory=list)
    verification_criteria: list[str] = field(default_factory=list)
    estimated_duration: float = 10.0
    status: TaskStatus = TaskStatus.CREATED


@dataclass
class MultiAgentWorkflow:
    """Multi-step workflow involving multiple agents"""

    workflow_id: str
    description: str
    steps: list[WorkflowStep]
    current_step: int = 0
    status: TaskStatus = TaskStatus.CREATED
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    shared_context: TaskContext = field(
        default_factory=lambda: TaskContext(task_id="", description="")
    )
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentHandoffProtocol:
    """Manages agent handoffs and multi-agent workflows"""

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.active_handoffs: dict[str, AgentHandoff] = {}
        self.active_workflows: dict[str, MultiAgentWorkflow] = {}
        self.handoff_history: list[AgentHandoff] = []
        self.context_store: dict[str, TaskContext] = {}
        self.default_timeout_minutes = 30

        # Verification templates for different handoff types
        self.verification_templates = {
            "architecture_to_implementation": [
                "Architecture design is documented",
                "Component interfaces are defined",
                "Dependencies are identified",
                "Design patterns are specified",
            ],
            "implementation_to_testing": [
                "Code implementation is complete",
                "Unit tests can be written",
                "API contracts are defined",
                "Error handling is implemented",
            ],
            "testing_to_deployment": [
                "All tests pass",
                "Coverage requirements met",
                "Performance benchmarks satisfied",
                "Security scans completed",
            ],
            "security_to_implementation": [
                "Security requirements identified",
                "Vulnerability assessment complete",
                "Mitigation strategies defined",
                "Compliance requirements documented",
            ],
        }

    def _generate_handoff_id(self) -> str:
        """Generate unique handoff ID"""
        import uuid

        return f"handoff_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    def _generate_workflow_id(self) -> str:
        """Generate unique workflow ID"""
        import uuid

        return f"workflow_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    def _get_verification_template(
        self, source_agent: str, target_agent: str
    ) -> list[str]:
        """Get verification checklist template for handoff type"""
        # Simple mapping based on agent names - can be enhanced
        handoff_type = f"{source_agent}_to_{target_agent}"

        # Try exact match first
        if handoff_type in self.verification_templates:
            return self.verification_templates[handoff_type].copy()

        # Try partial matches
        for template_key, checklist in self.verification_templates.items():
            if source_agent in template_key or target_agent in template_key:
                return checklist.copy()

        # Default verification checklist
        return [
            "Task requirements are clear",
            "All necessary context is provided",
            "Dependencies are documented",
            "Expected outputs are defined",
        ]

    async def initiate_handoff(
        self,
        source_agent: str,
        target_agent: str,
        task_id: str,
        context: TaskContext,
        handoff_message: str,
        custom_verifications: list[str] | None = None,
        timeout_minutes: int | None = None,
    ) -> str:
        """Initiate a handoff between two agents"""
        handoff_id = self._generate_handoff_id()

        # Get verification checklist
        verification_checklist = (
            custom_verifications
            or self._get_verification_template(source_agent, target_agent)
        )

        # Set timeout
        timeout_minutes = timeout_minutes or self.default_timeout_minutes
        timeout_at = datetime.now(UTC) + timedelta(minutes=timeout_minutes)

        handoff = AgentHandoff(
            handoff_id=handoff_id,
            source_agent=source_agent,
            target_agent=target_agent,
            task_id=task_id,
            context=context,
            handoff_message=handoff_message,
            status=HandoffStatus.PENDING,
            timeout_at=timeout_at,
            verification_checklist=verification_checklist,
        )

        self.active_handoffs[handoff_id] = handoff
        self.context_store[task_id] = context

        # Emit handoff initiated event
        if self.event_bus:
            event = create_event(
                "agent_handoff_initiated",
                handoff_id=handoff_id,
                source_agent=source_agent,
                target_agent=target_agent,
                task_id=task_id,
                handoff_message=handoff_message,
                verification_checklist=verification_checklist,
                timeout_at=timeout_at.isoformat(),
                source_plugin="agent_handoff_protocol",
            )
            await self.event_bus.publish(event)

        logger.info(
            f"Initiated handoff {handoff_id} from {source_agent} to {target_agent} for task {task_id}"
        )
        return handoff_id

    async def accept_handoff(self, handoff_id: str, accepting_agent: str) -> bool:
        """Agent accepts a handoff"""
        if handoff_id not in self.active_handoffs:
            logger.error(f"Handoff {handoff_id} not found")
            return False

        handoff = self.active_handoffs[handoff_id]

        if handoff.target_agent != accepting_agent:
            logger.error(
                f"Agent {accepting_agent} not authorized to accept handoff {handoff_id}"
            )
            return False

        if handoff.status != HandoffStatus.PENDING:
            logger.error(
                f"Handoff {handoff_id} is not in pending status: {handoff.status}"
            )
            return False

        handoff.status = HandoffStatus.IN_PROGRESS

        # Emit handoff accepted event
        if self.event_bus:
            event = create_event(
                "agent_handoff_accepted",
                handoff_id=handoff_id,
                accepting_agent=accepting_agent,
                task_id=handoff.task_id,
                context=json.dumps(handoff.context.__dict__, default=str),
                source_plugin="agent_handoff_protocol",
            )
            await self.event_bus.publish(event)

        logger.info(f"Agent {accepting_agent} accepted handoff {handoff_id}")
        return True

    async def complete_verification(
        self, handoff_id: str, verification_item: str, completed_by: str
    ) -> bool:
        """Mark a verification item as completed"""
        if handoff_id not in self.active_handoffs:
            logger.error(f"Handoff {handoff_id} not found")
            return False

        handoff = self.active_handoffs[handoff_id]

        if verification_item not in handoff.verification_checklist:
            logger.error(
                f"Verification item '{verification_item}' not in checklist for handoff {handoff_id}"
            )
            return False

        handoff.completed_verifications.add(verification_item)

        # Check if all verifications are complete
        if len(handoff.completed_verifications) == len(handoff.verification_checklist):
            await self._complete_handoff(handoff_id)

        # Emit verification completed event
        if self.event_bus:
            event = create_event(
                "handoff_verification_completed",
                handoff_id=handoff_id,
                verification_item=verification_item,
                completed_by=completed_by,
                remaining_verifications=len(handoff.verification_checklist)
                - len(handoff.completed_verifications),
                source_plugin="agent_handoff_protocol",
            )
            await self.event_bus.publish(event)

        logger.info(
            f"Verification '{verification_item}' completed for handoff {handoff_id} by {completed_by}"
        )
        return True

    async def _complete_handoff(self, handoff_id: str):
        """Complete a handoff"""
        handoff = self.active_handoffs[handoff_id]
        handoff.status = HandoffStatus.COMPLETED
        handoff.completed_at = datetime.now(UTC)

        # Move to history
        self.handoff_history.append(handoff)
        del self.active_handoffs[handoff_id]

        # Emit handoff completed event
        if self.event_bus:
            event = create_event(
                "agent_handoff_completed",
                handoff_id=handoff_id,
                source_agent=handoff.source_agent,
                target_agent=handoff.target_agent,
                task_id=handoff.task_id,
                duration_minutes=(
                    handoff.completed_at - handoff.created_at
                ).total_seconds()
                / 60,
                source_plugin="agent_handoff_protocol",
            )
            await self.event_bus.publish(event)

        logger.info(f"Handoff {handoff_id} completed successfully")

    def get_task_context(self, task_id: str) -> TaskContext | None:
        """Retrieve task context"""
        return self.context_store.get(task_id)

    def update_task_context(self, task_id: str, context_updates: dict[str, Any]):
        """Update task context with new information"""
        if task_id not in self.context_store:
            logger.warning(
                f"Task context for {task_id} not found, creating new context"
            )
            self.context_store[task_id] = TaskContext(task_id=task_id, description="")

        context = self.context_store[task_id]

        # Update context fields
        for key, value in context_updates.items():
            if hasattr(context, key):
                if isinstance(getattr(context, key), list):
                    if isinstance(value, list):
                        getattr(context, key).extend(value)
                    else:
                        getattr(context, key).append(value)
                elif isinstance(getattr(context, key), dict):
                    if isinstance(value, dict):
                        getattr(context, key).update(value)
                    else:
                        logger.warning(
                            f"Cannot update dict field {key} with non-dict value"
                        )
                else:
                    setattr(context, key, value)
            else:
                context.custom_metadata[key] = value

        logger.debug(f"Updated context for task {task_id}")

    async def create_workflow(
        self,
        description: str,
        steps: list[WorkflowStep],
        shared_context: TaskContext | None = None,
    ) -> str:
        """Create a multi-agent workflow"""
        workflow_id = self._generate_workflow_id()

        workflow = MultiAgentWorkflow(
            workflow_id=workflow_id,
            description=description,
            steps=steps,
            shared_context=shared_context
            or TaskContext(task_id=workflow_id, description=description),
        )

        self.active_workflows[workflow_id] = workflow
        self.context_store[workflow_id] = workflow.shared_context

        # Emit workflow created event
        if self.event_bus:
            event = create_event(
                "multi_agent_workflow_created",
                workflow_id=workflow_id,
                description=description,
                total_steps=len(steps),
                first_agent=steps[0].agent_id if steps else None,
                source_plugin="agent_handoff_protocol",
            )
            await self.event_bus.publish(event)

        logger.info(f"Created workflow {workflow_id} with {len(steps)} steps")
        return workflow_id

    async def advance_workflow(self, workflow_id: str) -> bool:
        """Advance workflow to next step"""
        if workflow_id not in self.active_workflows:
            logger.error(f"Workflow {workflow_id} not found")
            return False

        workflow = self.active_workflows[workflow_id]

        if workflow.current_step >= len(workflow.steps):
            logger.info(f"Workflow {workflow_id} already completed")
            return False

        current_step = workflow.steps[workflow.current_step]

        # Check dependencies
        for dep_step_id in current_step.depends_on:
            dep_step = next(
                (s for s in workflow.steps if s.step_id == dep_step_id), None
            )
            if not dep_step or dep_step.status != TaskStatus.COMPLETED:
                logger.warning(
                    f"Dependency {dep_step_id} not completed for step {current_step.step_id}"
                )
                return False

        # Start current step
        current_step.status = TaskStatus.IN_PROGRESS

        # If this is not the first step, create handoff from previous step
        if workflow.current_step > 0:
            prev_step = workflow.steps[workflow.current_step - 1]
            handoff_message = f"Workflow step handoff: {prev_step.task_description} -> {current_step.task_description}"

            await self.initiate_handoff(
                source_agent=prev_step.agent_id,
                target_agent=current_step.agent_id,
                task_id=workflow_id,
                context=workflow.shared_context,
                handoff_message=handoff_message,
                custom_verifications=current_step.verification_criteria,
            )

        workflow.current_step += 1

        # Emit workflow advanced event
        if self.event_bus:
            event = create_event(
                "workflow_step_started",
                workflow_id=workflow_id,
                step_id=current_step.step_id,
                agent_id=current_step.agent_id,
                step_description=current_step.task_description,
                step_number=workflow.current_step,
                total_steps=len(workflow.steps),
                source_plugin="agent_handoff_protocol",
            )
            await self.event_bus.publish(event)

        logger.info(
            f"Advanced workflow {workflow_id} to step {workflow.current_step}/{len(workflow.steps)}"
        )
        return True

    def get_handoff_status(self, handoff_id: str) -> dict[str, Any] | None:
        """Get status of a handoff"""
        handoff = self.active_handoffs.get(handoff_id)
        if not handoff:
            # Check history
            handoff = next(
                (h for h in self.handoff_history if h.handoff_id == handoff_id), None
            )

        if not handoff:
            return None

        return {
            "handoff_id": handoff.handoff_id,
            "source_agent": handoff.source_agent,
            "target_agent": handoff.target_agent,
            "task_id": handoff.task_id,
            "status": handoff.status.value,
            "created_at": handoff.created_at.isoformat(),
            "completed_at": (
                handoff.completed_at.isoformat() if handoff.completed_at else None
            ),
            "verification_progress": f"{len(handoff.completed_verifications)}/{len(handoff.verification_checklist)}",
            "remaining_verifications": [
                v
                for v in handoff.verification_checklist
                if v not in handoff.completed_verifications
            ],
        }

    def get_workflow_status(self, workflow_id: str) -> dict[str, Any] | None:
        """Get status of a workflow"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return None

        return {
            "workflow_id": workflow.workflow_id,
            "description": workflow.description,
            "status": workflow.status.value,
            "current_step": workflow.current_step,
            "total_steps": len(workflow.steps),
            "steps": [
                {
                    "step_id": step.step_id,
                    "agent_id": step.agent_id,
                    "status": step.status.value,
                    "description": step.task_description,
                }
                for step in workflow.steps
            ],
            "created_at": workflow.created_at.isoformat(),
            "completed_at": (
                workflow.completed_at.isoformat() if workflow.completed_at else None
            ),
        }

    async def cleanup_expired_handoffs(self):
        """Clean up expired handoffs"""
        now = datetime.now(UTC)
        expired_handoffs = []

        for handoff_id, handoff in self.active_handoffs.items():
            if handoff.timeout_at and now > handoff.timeout_at:
                expired_handoffs.append(handoff_id)

        for handoff_id in expired_handoffs:
            handoff = self.active_handoffs[handoff_id]
            handoff.status = HandoffStatus.TIMEOUT
            handoff.completed_at = now

            # Move to history
            self.handoff_history.append(handoff)
            del self.active_handoffs[handoff_id]

            # Emit timeout event
            if self.event_bus:
                event = create_event(
                    "agent_handoff_timeout",
                    handoff_id=handoff_id,
                    source_agent=handoff.source_agent,
                    target_agent=handoff.target_agent,
                    task_id=handoff.task_id,
                    source_plugin="agent_handoff_protocol",
                )
                await self.event_bus.publish(event)

            logger.warning(f"Handoff {handoff_id} timed out")

        return len(expired_handoffs)
