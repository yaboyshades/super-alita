"""
Solo Developer Multi-Agent Orchestration System

Central integration point for agent task routing, handoff protocols,
performance analytics, cost management, and capability mapping.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.core.events import create_event
from src.orchestration.agent_task_router import (
    AgentTaskRouter,
    TaskRequest,
    AgentSpecialization,
    TaskType,
)
from src.orchestration.agent_handoff_protocol import (
    AgentHandoffProtocol,
    TaskContext,
    WorkflowStep,
)
from src.orchestration.agent_performance_analytics import AgentPerformanceAnalytics
from src.orchestration.cost_management_dashboard import (
    CostManagementDashboard,
    CostCategory,
)
from src.orchestration.agent_capability_mapping import (
    AgentCapabilityMapping,
    ProficiencyLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestration system"""

    enable_cost_tracking: bool = True
    enable_performance_analytics: bool = True
    enable_capability_mapping: bool = True
    enable_handoff_protocol: bool = True
    default_task_timeout_minutes: int = 30
    max_concurrent_tasks_per_agent: int = 3
    cost_alert_threshold: float = 50.0  # USD
    performance_alert_threshold: float = 0.7  # Success rate


class SoloDevMultiAgentOrchestrator:
    """
    Central orchestrator for solo developer multi-agent workflows.

    Integrates all agent coordination systems:
    - Agent Task Router
    - Agent Handoff Protocol
    - Agent Performance Analytics
    - Cost Management Dashboard
    - Agent Capability Mapping
    """

    def __init__(self, event_bus=None, config: Optional[OrchestrationConfig] = None):
        self.event_bus = event_bus
        self.config = config or OrchestrationConfig()

        # Initialize subsystems
        self.task_router = AgentTaskRouter(event_bus)
        self.handoff_protocol = AgentHandoffProtocol(event_bus)
        self.performance_analytics = AgentPerformanceAnalytics(event_bus)
        self.cost_dashboard = CostManagementDashboard(event_bus)
        self.capability_mapping = AgentCapabilityMapping(event_bus)

        # Active tasks tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_executions: Dict[str, str] = {}  # task_id -> execution_id

        # System state
        self.is_running = False
        self.stats = {
            "total_tasks_processed": 0,
            "total_handoffs_completed": 0,
            "total_cost": 0.0,
            "system_start_time": None,
        }

        logger.info("Solo Developer Multi-Agent Orchestrator initialized")

    async def start(self):
        """Start the orchestration system"""
        if self.is_running:
            logger.warning("Orchestrator already running")
            return

        self.is_running = True
        self.stats["system_start_time"] = datetime.now(UTC)

        # Initialize default agent capabilities
        await self._initialize_default_agents()

        # Set up default budget limits
        await self._setup_default_budgets()

        # Start background tasks
        asyncio.create_task(self._cleanup_task())
        asyncio.create_task(self._performance_monitoring_task())

        logger.info("Solo Developer Multi-Agent Orchestrator started")

    async def stop(self):
        """Stop the orchestration system"""
        self.is_running = False
        logger.info("Solo Developer Multi-Agent Orchestrator stopped")

    async def _initialize_default_agents(self):
        """Initialize default agent capabilities"""
        default_agents = [
            {
                "agent_id": "security_agent",
                "specialization": AgentSpecialization.SECURITY,
                "capabilities": {
                    "security_analysis": ProficiencyLevel.EXPERT,
                    "secure_coding": ProficiencyLevel.ADVANCED,
                    "python_programming": ProficiencyLevel.INTERMEDIATE,
                },
            },
            {
                "agent_id": "architecture_agent",
                "specialization": AgentSpecialization.ARCHITECTURE,
                "capabilities": {
                    "system_architecture": ProficiencyLevel.EXPERT,
                    "api_design": ProficiencyLevel.ADVANCED,
                    "python_programming": ProficiencyLevel.ADVANCED,
                },
            },
            {
                "agent_id": "implementation_agent",
                "specialization": AgentSpecialization.IMPLEMENTATION,
                "capabilities": {
                    "python_programming": ProficiencyLevel.EXPERT,
                    "javascript_programming": ProficiencyLevel.ADVANCED,
                    "git_version_control": ProficiencyLevel.ADVANCED,
                },
            },
            {
                "agent_id": "testing_agent",
                "specialization": AgentSpecialization.TESTING,
                "capabilities": {
                    "unit_testing": ProficiencyLevel.EXPERT,
                    "integration_testing": ProficiencyLevel.ADVANCED,
                    "python_programming": ProficiencyLevel.INTERMEDIATE,
                },
            },
            {
                "agent_id": "documentation_agent",
                "specialization": AgentSpecialization.DOCUMENTATION,
                "capabilities": {
                    "technical_documentation": ProficiencyLevel.EXPERT,
                    "api_design": ProficiencyLevel.INTERMEDIATE,
                },
            },
        ]

        for agent_config in default_agents:
            # Register with capability mapping
            for capability_id, proficiency in agent_config["capabilities"].items():
                self.capability_mapping.register_agent_capability(
                    agent_config["agent_id"], capability_id, proficiency
                )

        logger.info(
            f"Initialized {len(default_agents)} default agents with capabilities"
        )

    async def _setup_default_budgets(self):
        """Set up default budget limits"""
        if self.config.enable_cost_tracking:
            # Daily budget limit
            self.cost_dashboard.create_budget_limit(
                name="Daily Development Budget",
                amount=self.config.cost_alert_threshold,
                period="daily",
                alert_thresholds=[0.5, 0.75, 0.9, 1.0],
                auto_disable=False,
            )

            # Monthly budget limit
            self.cost_dashboard.create_budget_limit(
                name="Monthly Development Budget",
                amount=self.config.cost_alert_threshold * 20,  # 20x daily
                period="monthly",
                alert_thresholds=[0.75, 0.9, 1.0],
                auto_disable=False,
            )

            logger.info("Set up default budget limits")

    async def submit_task(
        self,
        task_description: str,
        task_type: TaskType,
        priority: int = 5,
        complexity: int = 3,
        session_id: str = "",
        conversation_id: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a task for processing by the multi-agent system"""
        import uuid

        task_id = f"task_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # Auto-detect required capabilities
        if self.config.enable_capability_mapping:
            required_capabilities = (
                self.capability_mapping.auto_detect_task_requirements(
                    task_description, task_type.value
                )
            )

            # Define task requirements
            self.capability_mapping.define_task_requirements(
                task_id=task_id,
                task_type=task_type.value,
                required_capabilities=required_capabilities,
                complexity_score=complexity,
                description=task_description,
            )

        # Create task request
        task_request = TaskRequest(
            task_id=task_id,
            task_type=task_type,
            description=task_description,
            priority=priority,
            context=context or {},
            estimated_complexity=complexity,
            session_id=session_id,
            conversation_id=conversation_id,
        )

        # Route task to appropriate agent
        routing_decision = await self.task_router.route_task(task_request)

        # Start performance tracking
        if self.config.enable_performance_analytics:
            execution_id = self.performance_analytics.start_task_execution(
                agent_id=routing_decision.selected_agent,
                task_id=task_id,
                task_type=task_type.value,
            )
            self.task_executions[task_id] = execution_id

        # Create task context
        task_context = TaskContext(
            task_id=task_id, description=task_description, custom_metadata=context or {}
        )

        # Track active task
        self.active_tasks[task_id] = {
            "task_request": task_request,
            "routing_decision": routing_decision,
            "task_context": task_context,
            "start_time": datetime.now(UTC),
            "status": "active",
            "assigned_agent": routing_decision.selected_agent,
        }

        self.stats["total_tasks_processed"] += 1

        # Emit task submitted event
        if self.event_bus:
            event = create_event(
                "task_submitted",
                task_id=task_id,
                task_type=task_type.value,
                assigned_agent=routing_decision.selected_agent,
                priority=priority,
                complexity=complexity,
                estimated_cost=routing_decision.estimated_cost,
                estimated_duration=routing_decision.estimated_duration,
                source_plugin="solo_dev_orchestrator",
                session_id=session_id,
                conversation_id=conversation_id,
            )
            await self.event_bus.publish(event)

        logger.info(
            f"Submitted task {task_id} to agent {routing_decision.selected_agent}"
        )
        return task_id

    async def complete_task(
        self,
        task_id: str,
        success: bool,
        cost: float = 0.0,
        quality_score: Optional[float] = None,
        user_satisfaction: Optional[float] = None,
        outputs_generated: Optional[List[str]] = None,
        files_modified: Optional[List[str]] = None,
        tests_passed: int = 0,
        tests_failed: int = 0,
        code_lines_changed: int = 0,
        documentation_updated: bool = False,
        security_issues_found: int = 0,
        performance_improvement: Optional[float] = None,
        error_message: Optional[str] = None,
        provider: str = "",
        tokens_used: int = 0,
        api_requests: int = 0,
    ):
        """Complete a task with results and update all tracking systems"""
        if task_id not in self.active_tasks:
            logger.error(f"Task {task_id} not found in active tasks")
            return

        task_info = self.active_tasks[task_id]
        agent_id = task_info["assigned_agent"]
        task_type = task_info["task_request"].task_type.value

        # Record cost if provided
        if self.config.enable_cost_tracking and (cost > 0 or tokens_used > 0):
            if tokens_used > 0 and provider:
                await self.cost_dashboard.record_api_usage(
                    agent_id=agent_id,
                    task_id=task_id,
                    provider=provider,
                    model="default",  # Could be parameterized
                    tokens_used=tokens_used,
                    requests_count=api_requests or 1,
                    task_type=task_type,
                )
            elif cost > 0:
                await self.cost_dashboard.record_cost(
                    agent_id=agent_id,
                    task_id=task_id,
                    task_type=task_type,
                    category=CostCategory.API_CALLS,
                    amount=cost,
                    description=f"Task completion: {task_type}",
                )

        # Complete performance tracking
        if self.config.enable_performance_analytics and task_id in self.task_executions:
            execution_id = self.task_executions[task_id]
            await self.performance_analytics.complete_task_execution(
                execution_id=execution_id,
                success=success,
                cost=cost,
                quality_score=quality_score,
                user_satisfaction=user_satisfaction,
                outputs_generated=outputs_generated,
                files_modified=files_modified,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                code_lines_changed=code_lines_changed,
                documentation_updated=documentation_updated,
                security_issues_found=security_issues_found,
                performance_improvement=performance_improvement,
                error_message=error_message,
            )
            del self.task_executions[task_id]

        # Update capability scores
        if self.config.enable_capability_mapping:
            required_capabilities = []
            if task_id in self.capability_mapping.task_requirements:
                required_capabilities = self.capability_mapping.task_requirements[
                    task_id
                ].required_capabilities

            for capability_id in required_capabilities:
                performance_score = quality_score or (0.8 if success else 0.3)
                self.capability_mapping.update_agent_capability_score(
                    agent_id=agent_id,
                    capability_id=capability_id,
                    performance_score=performance_score,
                    task_success=success,
                )

        # Update task context with results
        self.handoff_protocol.update_task_context(
            task_id,
            {
                "outputs_generated": outputs_generated or [],
                "files_modified": files_modified or [],
                "test_results": {
                    "tests_passed": tests_passed,
                    "tests_failed": tests_failed,
                },
                "documentation_updated": (
                    [f"Task {task_id} documentation"] if documentation_updated else []
                ),
                "security_considerations": (
                    [f"Found {security_issues_found} security issues"]
                    if security_issues_found > 0
                    else []
                ),
                "performance_metrics": (
                    {
                        "lines_changed": code_lines_changed,
                        "performance_improvement": performance_improvement,
                    }
                    if performance_improvement
                    else {}
                ),
            },
        )

        # Update task status
        task_info["status"] = "completed" if success else "failed"
        task_info["end_time"] = datetime.now(UTC)
        task_info["success"] = success

        # Update router with performance feedback
        duration = (
            task_info["end_time"] - task_info["start_time"]
        ).total_seconds() / 60
        self.task_router.update_agent_performance(agent_id, success, duration)

        # Update stats
        self.stats["total_cost"] += cost

        # Remove from active tasks
        del self.active_tasks[task_id]

        # Emit task completed event
        if self.event_bus:
            event = create_event(
                "task_completed",
                task_id=task_id,
                agent_id=agent_id,
                task_type=task_type,
                success=success,
                duration_minutes=duration,
                cost=cost,
                quality_score=quality_score,
                user_satisfaction=user_satisfaction,
                source_plugin="solo_dev_orchestrator",
            )
            await self.event_bus.publish(event)

        logger.info(f"Completed task {task_id}: success={success}, cost=${cost:.2f}")

    async def create_multi_agent_workflow(
        self,
        workflow_description: str,
        steps: List[Dict[str, Any]],
        session_id: str = "",
        conversation_id: str = "",
    ) -> str:
        """Create a multi-agent workflow"""
        # Convert step dictionaries to WorkflowStep objects
        workflow_steps = []
        for i, step_data in enumerate(steps):
            step = WorkflowStep(
                step_id=f"step_{i+1}",
                agent_id=step_data["agent_id"],
                task_description=step_data["description"],
                depends_on=step_data.get("depends_on", []),
                outputs_required=step_data.get("outputs_required", []),
                verification_criteria=step_data.get("verification_criteria", []),
                estimated_duration=step_data.get("estimated_duration", 10.0),
            )
            workflow_steps.append(step)

        # Create shared context
        shared_context = TaskContext(
            task_id="", description=workflow_description  # Will be set by workflow
        )

        # Create workflow
        workflow_id = await self.handoff_protocol.create_workflow(
            description=workflow_description,
            steps=workflow_steps,
            shared_context=shared_context,
        )

        self.stats["total_handoffs_completed"] += 1

        logger.info(
            f"Created multi-agent workflow {workflow_id} with {len(steps)} steps"
        )
        return workflow_id

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "orchestrator": {
                "is_running": self.is_running,
                "uptime_minutes": 0,
                "active_tasks": len(self.active_tasks),
                "stats": self.stats.copy(),
            }
        }

        if self.stats["system_start_time"]:
            uptime = datetime.now(UTC) - self.stats["system_start_time"]
            status["orchestrator"]["uptime_minutes"] = uptime.total_seconds() / 60

        # Agent status
        status["agents"] = self.task_router.get_agent_status()

        # Performance analytics
        if self.config.enable_performance_analytics:
            comparative_analysis = self.performance_analytics.get_comparative_analysis()
            status["performance"] = {
                "total_agents": comparative_analysis["total_agents"],
                "summary": comparative_analysis["summary"],
            }

        # Cost management
        if self.config.enable_cost_tracking:
            cost_summary = self.cost_dashboard.get_cost_summary(7)  # Last 7 days
            budget_status = self.cost_dashboard.get_budget_status()
            status["costs"] = {
                "last_7_days": cost_summary["total_cost"],
                "projected_monthly": cost_summary["projected_monthly"],
                "budget_status": len(
                    [b for b in budget_status.values() if b["status"] != "healthy"]
                ),
            }

        # Capability mapping
        if self.config.enable_capability_mapping:
            capability_analytics = self.capability_mapping.get_capability_analytics()
            status["capabilities"] = {
                "total_capabilities": capability_analytics["total_capabilities"],
                "capability_gaps": len(capability_analytics["capability_gaps"]),
            }

        # Active handoffs
        status["handoffs"] = {
            "active_handoffs": len(self.handoff_protocol.active_handoffs),
            "active_workflows": len(self.handoff_protocol.active_workflows),
        }

        return status

    async def get_recommendations(self) -> Dict[str, List[str]]:
        """Get system-wide recommendations for optimization"""
        recommendations = {
            "performance": [],
            "cost": [],
            "capabilities": [],
            "system": [],
        }

        # Performance recommendations
        if self.config.enable_performance_analytics:
            perf_report = await self.performance_analytics.generate_performance_report()
            if "recommendations" in perf_report:
                recommendations["performance"] = perf_report["recommendations"]

        # Cost recommendations
        if self.config.enable_cost_tracking:
            cost_recs = self.cost_dashboard.get_cost_optimization_recommendations()
            recommendations["cost"] = cost_recs

        # Capability recommendations
        if self.config.enable_capability_mapping:
            capability_analytics = self.capability_mapping.get_capability_analytics()
            gaps = capability_analytics["capability_gaps"]

            for gap in gaps[:5]:  # Top 5 gaps
                if gap["severity"] == "critical":
                    recommendations["capabilities"].append(
                        f"Critical: No agents have {gap['capability_name']} capability"
                    )
                elif gap["severity"] == "high":
                    recommendations["capabilities"].append(
                        f"High: Only one agent has {gap['capability_name']} capability"
                    )

        # System recommendations
        if len(self.active_tasks) > 10:
            recommendations["system"].append("High task load - consider scaling agents")

        if self.stats["total_cost"] > self.config.cost_alert_threshold * 5:
            recommendations["system"].append(
                "High cumulative costs - review budget limits"
            )

        return recommendations

    async def _cleanup_task(self):
        """Background task for system cleanup"""
        while self.is_running:
            try:
                # Clean up expired handoffs
                if self.config.enable_handoff_protocol:
                    expired_count = (
                        await self.handoff_protocol.cleanup_expired_handoffs()
                    )
                    if expired_count > 0:
                        logger.info(f"Cleaned up {expired_count} expired handoffs")

                # Clean up old task data (keep last 1000 entries)
                if len(self.performance_analytics.performance_history) > 1000:
                    self.performance_analytics.performance_history = (
                        self.performance_analytics.performance_history[-1000:]
                    )

                if len(self.cost_dashboard.cost_entries) > 1000:
                    self.cost_dashboard.cost_entries = self.cost_dashboard.cost_entries[
                        -1000:
                    ]

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)

    async def _performance_monitoring_task(self):
        """Background task for performance monitoring"""
        while self.is_running:
            try:
                # Check for performance issues
                agent_status = self.task_router.get_agent_status()

                for agent_id, status in agent_status.items():
                    success_rate = status.get("success_rate", 1.0)
                    if success_rate < self.config.performance_alert_threshold:
                        logger.warning(
                            f"Agent {agent_id} performance below threshold: {success_rate:.2f}"
                        )

                        # Emit performance alert
                        if self.event_bus:
                            event = create_event(
                                "agent_performance_degraded",
                                agent_id=agent_id,
                                success_rate=success_rate,
                                threshold=self.config.performance_alert_threshold,
                                source_plugin="solo_dev_orchestrator",
                            )
                            await self.event_bus.publish(event)

                await asyncio.sleep(600)  # Run every 10 minutes

            except Exception as e:
                logger.error(f"Error in performance monitoring task: {e}")
                await asyncio.sleep(60)
