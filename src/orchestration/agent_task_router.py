"""
Agent Task Router for Solo Developer Multi-Agent Orchestration

Routes different types of issues to specialized agents based on task type,
capability requirements, and agent performance history.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.core.events import create_event

logger = logging.getLogger(__name__)


class AgentSpecialization(Enum):
    """Types of specialized agents"""

    SECURITY = "security"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    PERFORMANCE = "performance"


class TaskType(Enum):
    """Types of development tasks"""

    BUG_FIX = "bug_fix"
    FEATURE_DEVELOPMENT = "feature_development"
    SECURITY_SCAN = "security_scan"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    ARCHITECTURE_DESIGN = "architecture_design"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DEPLOYMENT = "deployment"


@dataclass
class AgentCapability:
    """Agent capability definition"""

    agent_id: str
    specialization: AgentSpecialization
    supported_tasks: list[TaskType]
    performance_score: float = 0.8
    cost_per_task: float = 0.0
    availability: bool = True
    current_load: int = 0
    max_load: int = 5


@dataclass
class TaskRequest:
    """Task routing request"""

    task_id: str
    task_type: TaskType
    description: str
    priority: int = 5  # 1-10, 10 = highest
    context: dict[str, Any] = field(default_factory=dict)
    required_capabilities: list[str] = field(default_factory=list)
    estimated_complexity: int = 3  # 1-5, 5 = most complex
    deadline: datetime | None = None
    session_id: str = ""
    conversation_id: str = ""


@dataclass
class RoutingDecision:
    """Agent routing decision"""

    task_id: str
    selected_agent: str
    agent_specialization: AgentSpecialization
    confidence: float
    reasoning: str
    estimated_duration: float
    estimated_cost: float
    fallback_agents: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentTaskRouter:
    """Routes tasks to specialized agents based on capabilities and performance"""

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.agents: dict[str, AgentCapability] = {}
        self.task_history: list[dict[str, Any]] = []
        self.performance_metrics: dict[str, dict[str, float]] = {}
        self.routing_strategy = (
            "capability_match"  # capability_match, load_balance, performance_optimized
        )

        # Initialize default specialized agents
        self._initialize_default_agents()

    def _initialize_default_agents(self):
        """Initialize default specialized agents"""
        default_agents = [
            AgentCapability(
                agent_id="security_agent",
                specialization=AgentSpecialization.SECURITY,
                supported_tasks=[TaskType.SECURITY_SCAN, TaskType.CODE_REVIEW],
                performance_score=0.9,
                cost_per_task=0.5,
            ),
            AgentCapability(
                agent_id="architecture_agent",
                specialization=AgentSpecialization.ARCHITECTURE,
                supported_tasks=[TaskType.ARCHITECTURE_DESIGN, TaskType.REFACTORING],
                performance_score=0.85,
                cost_per_task=1.0,
            ),
            AgentCapability(
                agent_id="implementation_agent",
                specialization=AgentSpecialization.IMPLEMENTATION,
                supported_tasks=[TaskType.FEATURE_DEVELOPMENT, TaskType.BUG_FIX],
                performance_score=0.8,
                cost_per_task=0.7,
            ),
            AgentCapability(
                agent_id="testing_agent",
                specialization=AgentSpecialization.TESTING,
                supported_tasks=[TaskType.TESTING],
                performance_score=0.88,
                cost_per_task=0.4,
            ),
            AgentCapability(
                agent_id="documentation_agent",
                specialization=AgentSpecialization.DOCUMENTATION,
                supported_tasks=[TaskType.DOCUMENTATION],
                performance_score=0.75,
                cost_per_task=0.3,
            ),
            AgentCapability(
                agent_id="debugging_agent",
                specialization=AgentSpecialization.DEBUGGING,
                supported_tasks=[TaskType.BUG_FIX, TaskType.PERFORMANCE_OPTIMIZATION],
                performance_score=0.85,
                cost_per_task=0.6,
            ),
        ]

        for agent in default_agents:
            self.agents[agent.agent_id] = agent
            self.performance_metrics[agent.agent_id] = {
                "success_rate": agent.performance_score,
                "avg_duration": 10.0,
                "total_tasks": 0,
                "failed_tasks": 0,
            }

    def register_agent(self, agent: AgentCapability):
        """Register a new agent with the router"""
        self.agents[agent.agent_id] = agent
        if agent.agent_id not in self.performance_metrics:
            self.performance_metrics[agent.agent_id] = {
                "success_rate": agent.performance_score,
                "avg_duration": 10.0,
                "total_tasks": 0,
                "failed_tasks": 0,
            }
        logger.info(
            f"Registered agent: {agent.agent_id} with specialization: {agent.specialization}"
        )

    def update_agent_performance(
        self, agent_id: str, task_success: bool, duration: float
    ):
        """Update agent performance metrics"""
        if agent_id not in self.performance_metrics:
            return

        metrics = self.performance_metrics[agent_id]
        metrics["total_tasks"] += 1

        if not task_success:
            metrics["failed_tasks"] += 1

        # Update success rate (exponential moving average)
        current_success_rate = metrics["success_rate"]
        new_success_rate = 1 if task_success else 0
        metrics["success_rate"] = 0.9 * current_success_rate + 0.1 * new_success_rate

        # Update average duration (exponential moving average)
        current_avg = metrics["avg_duration"]
        metrics["avg_duration"] = 0.9 * current_avg + 0.1 * duration

        # Update agent capability performance score
        if agent_id in self.agents:
            self.agents[agent_id].performance_score = metrics["success_rate"]

        logger.debug(
            f"Updated performance for {agent_id}: success_rate={metrics['success_rate']:.2f}"
        )

    def _calculate_agent_score(
        self, agent: AgentCapability, task: TaskRequest
    ) -> float:
        """Calculate score for agent-task match"""
        base_score = 0.0

        # Task type compatibility
        if task.task_type in agent.supported_tasks:
            base_score += 0.4

        # Performance score
        base_score += 0.3 * agent.performance_score

        # Load balancing
        load_factor = 1.0 - (agent.current_load / max(agent.max_load, 1))
        base_score += 0.2 * load_factor

        # Cost efficiency (lower cost = higher score)
        cost_factor = 1.0 - min(agent.cost_per_task / 2.0, 1.0)
        base_score += 0.1 * cost_factor

        # Availability
        if not agent.availability:
            base_score *= 0.1

        return min(base_score, 1.0)

    def _select_best_agent(self, task: TaskRequest) -> tuple[str | None, list[str]]:
        """Select best agent for task and return fallback options"""
        agent_scores = []

        for agent_id, agent in self.agents.items():
            score = self._calculate_agent_score(agent, task)
            agent_scores.append((agent_id, score, agent))

        # Sort by score descending
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        if not agent_scores or agent_scores[0][1] < 0.1:
            return None, []

        best_agent = agent_scores[0][0]
        fallback_agents = [
            agent_id for agent_id, score, _ in agent_scores[1:3] if score > 0.2
        ]

        return best_agent, fallback_agents

    async def route_task(self, task: TaskRequest) -> RoutingDecision:
        """Route task to best available agent"""
        logger.info(f"Routing task {task.task_id} of type {task.task_type}")

        best_agent, fallback_agents = self._select_best_agent(task)

        if not best_agent:
            # No suitable agent found - create a fallback decision
            logger.warning(f"No suitable agent found for task {task.task_id}")
            return RoutingDecision(
                task_id=task.task_id,
                selected_agent="general_agent",
                agent_specialization=AgentSpecialization.IMPLEMENTATION,
                confidence=0.1,
                reasoning="No specialized agent available, routing to general agent",
                estimated_duration=20.0,
                estimated_cost=1.0,
                fallback_agents=[],
            )

        agent = self.agents[best_agent]
        confidence = self._calculate_agent_score(agent, task)

        # Estimate duration and cost
        base_duration = 10.0 * task.estimated_complexity
        agent_duration_factor = (
            self.performance_metrics[best_agent]["avg_duration"] / 10.0
        )
        estimated_duration = base_duration * agent_duration_factor
        estimated_cost = agent.cost_per_task * task.estimated_complexity

        decision = RoutingDecision(
            task_id=task.task_id,
            selected_agent=best_agent,
            agent_specialization=agent.specialization,
            confidence=confidence,
            reasoning=f"Selected {best_agent} based on specialization {agent.specialization} and performance score {agent.performance_score:.2f}",
            estimated_duration=estimated_duration,
            estimated_cost=estimated_cost,
            fallback_agents=fallback_agents,
            metadata={
                "task_type": task.task_type.value,
                "priority": task.priority,
                "complexity": task.estimated_complexity,
            },
        )

        # Update agent load
        agent.current_load += 1

        # Record routing decision
        self.task_history.append(
            {
                "timestamp": datetime.now(UTC),
                "task_id": task.task_id,
                "agent_id": best_agent,
                "task_type": task.task_type.value,
                "confidence": confidence,
            }
        )

        # Emit routing event
        if self.event_bus:
            event = create_event(
                "agent_task_routed",
                task_id=task.task_id,
                selected_agent=best_agent,
                agent_specialization=agent.specialization.value,
                confidence=confidence,
                estimated_duration=estimated_duration,
                estimated_cost=estimated_cost,
                source_plugin="agent_task_router",
                session_id=task.session_id,
                conversation_id=task.conversation_id,
            )
            await self.event_bus.publish(event)

        logger.info(
            f"Routed task {task.task_id} to {best_agent} with confidence {confidence:.2f}"
        )
        return decision

    def get_agent_status(self) -> dict[str, Any]:
        """Get status of all agents"""
        status = {}
        for agent_id, agent in self.agents.items():
            metrics = self.performance_metrics.get(agent_id, {})
            status[agent_id] = {
                "specialization": agent.specialization.value,
                "availability": agent.availability,
                "current_load": agent.current_load,
                "max_load": agent.max_load,
                "performance_score": agent.performance_score,
                "total_tasks": metrics.get("total_tasks", 0),
                "success_rate": metrics.get("success_rate", 0.0),
                "avg_duration": metrics.get("avg_duration", 0.0),
            }
        return status

    def get_routing_analytics(self) -> dict[str, Any]:
        """Get routing analytics and performance data"""
        total_tasks = len(self.task_history)

        if total_tasks == 0:
            return {"total_tasks": 0, "agents": {}}

        # Analyze task distribution
        agent_task_counts = {}
        task_type_counts = {}

        for task in self.task_history:
            agent_id = task["agent_id"]
            task_type = task["task_type"]

            agent_task_counts[agent_id] = agent_task_counts.get(agent_id, 0) + 1
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

        return {
            "total_tasks": total_tasks,
            "agent_distribution": agent_task_counts,
            "task_type_distribution": task_type_counts,
            "agent_performance": self.performance_metrics.copy(),
            "recent_tasks": self.task_history[-10:],  # Last 10 tasks
        }
