"""
Agent Performance Analytics for Solo Developer Multi-Agent Orchestration

Tracks agent performance metrics, success rates, and provides
analytics for optimizing agent task routing.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

from src.core.events import create_event

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Types of performance metrics"""

    SUCCESS_RATE = "success_rate"
    AVERAGE_DURATION = "average_duration"
    COST_EFFICIENCY = "cost_efficiency"
    QUALITY_SCORE = "quality_score"
    USER_SATISFACTION = "user_satisfaction"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class TaskExecution:
    """Record of a task execution by an agent"""

    execution_id: str
    agent_id: str
    task_id: str
    task_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    success: Optional[bool] = None
    cost: float = 0.0
    quality_score: Optional[float] = None  # 0-1 scale
    user_satisfaction: Optional[float] = None  # 0-1 scale
    error_message: Optional[str] = None
    outputs_generated: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    code_lines_changed: int = 0
    documentation_updated: bool = False
    security_issues_found: int = 0
    performance_improvement: Optional[float] = None  # Percentage improvement
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformanceProfile:
    """Performance profile for an agent"""

    agent_id: str
    specialization: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_duration: float = 0.0
    total_cost: float = 0.0
    average_quality_score: float = 0.0
    average_user_satisfaction: float = 0.0
    task_type_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recent_executions: List[TaskExecution] = field(default_factory=list)
    performance_trend: List[Tuple[datetime, float]] = field(
        default_factory=list
    )  # (timestamp, success_rate)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


class AgentPerformanceAnalytics:
    """Analytics system for tracking and analyzing agent performance"""

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.executions: Dict[str, TaskExecution] = {}
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}
        self.performance_history: List[TaskExecution] = []
        self.analytics_cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, datetime] = {}

        # Performance thresholds for alerts
        self.performance_thresholds = {
            "min_success_rate": 0.7,
            "max_average_duration": 30.0,  # minutes
            "min_quality_score": 0.6,
            "max_error_rate": 0.3,
        }

    def start_task_execution(
        self,
        agent_id: str,
        task_id: str,
        task_type: str,
        execution_id: Optional[str] = None,
    ) -> str:
        """Start tracking a task execution"""
        if not execution_id:
            import uuid

            execution_id = f"exec_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        execution = TaskExecution(
            execution_id=execution_id,
            agent_id=agent_id,
            task_id=task_id,
            task_type=task_type,
            start_time=datetime.now(UTC),
        )

        self.executions[execution_id] = execution

        # Initialize agent profile if not exists
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = AgentPerformanceProfile(
                agent_id=agent_id, specialization="unknown"
            )

        logger.info(f"Started tracking execution {execution_id} for agent {agent_id}")
        return execution_id

    async def complete_task_execution(
        self,
        execution_id: str,
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
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Complete task execution with results"""
        if execution_id not in self.executions:
            logger.error(f"Execution {execution_id} not found")
            return

        execution = self.executions[execution_id]
        execution.end_time = datetime.now(UTC)
        execution.duration_minutes = (
            execution.end_time - execution.start_time
        ).total_seconds() / 60
        execution.success = success
        execution.cost = cost
        execution.quality_score = quality_score
        execution.user_satisfaction = user_satisfaction
        execution.outputs_generated = outputs_generated or []
        execution.files_modified = files_modified or []
        execution.tests_passed = tests_passed
        execution.tests_failed = tests_failed
        execution.code_lines_changed = code_lines_changed
        execution.documentation_updated = documentation_updated
        execution.security_issues_found = security_issues_found
        execution.performance_improvement = performance_improvement
        execution.error_message = error_message
        execution.metadata = metadata or {}

        # Move to history
        self.performance_history.append(execution)
        del self.executions[execution_id]

        # Update agent profile
        await self._update_agent_profile(execution)

        # Emit performance event
        if self.event_bus:
            event = create_event(
                "agent_task_completed",
                execution_id=execution_id,
                agent_id=execution.agent_id,
                task_id=execution.task_id,
                task_type=execution.task_type,
                success=success,
                duration_minutes=execution.duration_minutes,
                cost=cost,
                quality_score=quality_score,
                source_plugin="agent_performance_analytics",
            )
            await self.event_bus.publish(event)

        # Check for performance alerts
        await self._check_performance_alerts(execution.agent_id)

        logger.info(
            f"Completed execution {execution_id} for agent {execution.agent_id}: success={success}"
        )

    async def _update_agent_profile(self, execution: TaskExecution):
        """Update agent performance profile with new execution data"""
        agent_id = execution.agent_id
        profile = self.agent_profiles[agent_id]

        # Update basic counts
        profile.total_tasks += 1
        if execution.success:
            profile.successful_tasks += 1
        else:
            profile.failed_tasks += 1

        # Update averages (exponential moving average)
        alpha = 0.1  # Learning rate

        if execution.duration_minutes:
            if profile.average_duration == 0:
                profile.average_duration = execution.duration_minutes
            else:
                profile.average_duration = (
                    1 - alpha
                ) * profile.average_duration + alpha * execution.duration_minutes

        profile.total_cost += execution.cost

        if execution.quality_score is not None:
            if profile.average_quality_score == 0:
                profile.average_quality_score = execution.quality_score
            else:
                profile.average_quality_score = (
                    1 - alpha
                ) * profile.average_quality_score + alpha * execution.quality_score

        if execution.user_satisfaction is not None:
            if profile.average_user_satisfaction == 0:
                profile.average_user_satisfaction = execution.user_satisfaction
            else:
                profile.average_user_satisfaction = (
                    (1 - alpha) * profile.average_user_satisfaction
                    + alpha * execution.user_satisfaction
                )

        # Update task type performance
        task_type = execution.task_type
        if task_type not in profile.task_type_performance:
            profile.task_type_performance[task_type] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "average_duration": 0.0,
                "average_cost": 0.0,
            }

        task_perf = profile.task_type_performance[task_type]
        task_perf["total_tasks"] += 1
        if execution.success:
            task_perf["successful_tasks"] += 1

        if execution.duration_minutes:
            if task_perf["average_duration"] == 0:
                task_perf["average_duration"] = execution.duration_minutes
            else:
                task_perf["average_duration"] = (1 - alpha) * task_perf[
                    "average_duration"
                ] + alpha * execution.duration_minutes

        if task_perf["average_cost"] == 0:
            task_perf["average_cost"] = execution.cost
        else:
            task_perf["average_cost"] = (1 - alpha) * task_perf[
                "average_cost"
            ] + alpha * execution.cost

        # Keep recent executions (last 20)
        profile.recent_executions.append(execution)
        if len(profile.recent_executions) > 20:
            profile.recent_executions = profile.recent_executions[-20:]

        # Update performance trend
        success_rate = (
            profile.successful_tasks / profile.total_tasks
            if profile.total_tasks > 0
            else 0
        )
        profile.performance_trend.append((datetime.now(UTC), success_rate))
        if len(profile.performance_trend) > 100:  # Keep last 100 data points
            profile.performance_trend = profile.performance_trend[-100:]

        # Analyze strengths and weaknesses
        await self._analyze_agent_capabilities(agent_id)

        profile.last_updated = datetime.now(UTC)

        # Clear analytics cache for this agent
        self._clear_agent_cache(agent_id)

    async def _analyze_agent_capabilities(self, agent_id: str):
        """Analyze agent capabilities to identify strengths and weaknesses"""
        profile = self.agent_profiles[agent_id]

        strengths = []
        weaknesses = []

        # Analyze overall performance
        success_rate = (
            profile.successful_tasks / profile.total_tasks
            if profile.total_tasks > 0
            else 0
        )
        if success_rate > 0.9:
            strengths.append("High success rate")
        elif success_rate < 0.7:
            weaknesses.append("Low success rate")

        # Analyze duration performance
        if profile.average_duration < 10:
            strengths.append("Fast execution")
        elif profile.average_duration > 30:
            weaknesses.append("Slow execution")

        # Analyze quality
        if profile.average_quality_score > 0.8:
            strengths.append("High quality output")
        elif profile.average_quality_score < 0.6:
            weaknesses.append("Quality concerns")

        # Analyze task type performance
        best_task_types = []
        worst_task_types = []

        for task_type, perf in profile.task_type_performance.items():
            if perf["total_tasks"] >= 3:  # Only consider if enough data
                task_success_rate = perf["successful_tasks"] / perf["total_tasks"]
                if task_success_rate > 0.85:
                    best_task_types.append(task_type)
                elif task_success_rate < 0.6:
                    worst_task_types.append(task_type)

        if best_task_types:
            strengths.append(f"Excellent at: {', '.join(best_task_types)}")
        if worst_task_types:
            weaknesses.append(f"Struggles with: {', '.join(worst_task_types)}")

        # Analyze recent trend
        if len(profile.performance_trend) >= 10:
            recent_trend = profile.performance_trend[-10:]
            recent_success_rates = [rate for _, rate in recent_trend]
            if len(recent_success_rates) > 1:
                trend_slope = (
                    recent_success_rates[-1] - recent_success_rates[0]
                ) / len(recent_success_rates)
                if trend_slope > 0.05:
                    strengths.append("Improving performance")
                elif trend_slope < -0.05:
                    weaknesses.append("Declining performance")

        profile.strengths = strengths
        profile.weaknesses = weaknesses

    async def _check_performance_alerts(self, agent_id: str):
        """Check if agent performance requires alerts"""
        profile = self.agent_profiles[agent_id]

        alerts = []

        # Check success rate
        success_rate = (
            profile.successful_tasks / profile.total_tasks
            if profile.total_tasks > 0
            else 1
        )
        if success_rate < self.performance_thresholds["min_success_rate"]:
            alerts.append(f"Low success rate: {success_rate:.2f}")

        # Check average duration
        if (
            profile.average_duration
            > self.performance_thresholds["max_average_duration"]
        ):
            alerts.append(
                f"High average duration: {profile.average_duration:.1f} minutes"
            )

        # Check quality score
        if (
            profile.average_quality_score > 0
            and profile.average_quality_score
            < self.performance_thresholds["min_quality_score"]
        ):
            alerts.append(f"Low quality score: {profile.average_quality_score:.2f}")

        # Check error rate
        error_rate = (
            profile.failed_tasks / profile.total_tasks if profile.total_tasks > 0 else 0
        )
        if error_rate > self.performance_thresholds["max_error_rate"]:
            alerts.append(f"High error rate: {error_rate:.2f}")

        # Emit alerts if any
        if alerts and self.event_bus:
            event = create_event(
                "agent_performance_alert",
                agent_id=agent_id,
                alerts=alerts,
                success_rate=success_rate,
                average_duration=profile.average_duration,
                quality_score=profile.average_quality_score,
                error_rate=error_rate,
                source_plugin="agent_performance_analytics",
            )
            await self.event_bus.publish(event)

            logger.warning(
                f"Performance alerts for agent {agent_id}: {'; '.join(alerts)}"
            )

    def get_agent_performance(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get performance summary for an agent"""
        if agent_id not in self.agent_profiles:
            return None

        # Check cache first
        cache_key = f"agent_perf_{agent_id}"
        if (
            cache_key in self.analytics_cache
            and cache_key in self.cache_expiry
            and datetime.now(UTC) < self.cache_expiry[cache_key]
        ):
            return self.analytics_cache[cache_key]

        profile = self.agent_profiles[agent_id]

        success_rate = (
            profile.successful_tasks / profile.total_tasks
            if profile.total_tasks > 0
            else 0
        )
        error_rate = (
            profile.failed_tasks / profile.total_tasks if profile.total_tasks > 0 else 0
        )

        performance_summary = {
            "agent_id": agent_id,
            "specialization": profile.specialization,
            "total_tasks": profile.total_tasks,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "average_duration_minutes": profile.average_duration,
            "total_cost": profile.total_cost,
            "average_cost_per_task": profile.total_cost / max(profile.total_tasks, 1),
            "average_quality_score": profile.average_quality_score,
            "average_user_satisfaction": profile.average_user_satisfaction,
            "task_type_performance": profile.task_type_performance,
            "strengths": profile.strengths,
            "weaknesses": profile.weaknesses,
            "recent_performance_trend": (
                profile.performance_trend[-10:] if profile.performance_trend else []
            ),
            "last_updated": profile.last_updated.isoformat(),
        }

        # Cache result for 5 minutes
        self.analytics_cache[cache_key] = performance_summary
        self.cache_expiry[cache_key] = datetime.now(UTC) + timedelta(minutes=5)

        return performance_summary

    def get_comparative_analysis(self) -> Dict[str, Any]:
        """Get comparative analysis of all agents"""
        cache_key = "comparative_analysis"
        if (
            cache_key in self.analytics_cache
            and cache_key in self.cache_expiry
            and datetime.now(UTC) < self.cache_expiry[cache_key]
        ):
            return self.analytics_cache[cache_key]

        if not self.agent_profiles:
            return {"agents": [], "summary": {}}

        agent_summaries = []
        success_rates = []
        durations = []
        costs = []
        quality_scores = []

        for agent_id, profile in self.agent_profiles.items():
            if profile.total_tasks == 0:
                continue

            success_rate = profile.successful_tasks / profile.total_tasks
            success_rates.append(success_rate)
            durations.append(profile.average_duration)
            costs.append(profile.total_cost / profile.total_tasks)
            if profile.average_quality_score > 0:
                quality_scores.append(profile.average_quality_score)

            agent_summaries.append(
                {
                    "agent_id": agent_id,
                    "specialization": profile.specialization,
                    "success_rate": success_rate,
                    "average_duration": profile.average_duration,
                    "cost_per_task": profile.total_cost / profile.total_tasks,
                    "quality_score": profile.average_quality_score,
                    "total_tasks": profile.total_tasks,
                }
            )

        # Calculate summary statistics
        summary = {}
        if success_rates:
            summary["average_success_rate"] = statistics.mean(success_rates)
            summary["best_success_rate"] = max(success_rates)
            summary["worst_success_rate"] = min(success_rates)

        if durations:
            summary["average_duration"] = statistics.mean(durations)
            summary["fastest_average"] = min(durations)
            summary["slowest_average"] = max(durations)

        if costs:
            summary["average_cost_per_task"] = statistics.mean(costs)
            summary["most_cost_effective"] = min(costs)
            summary["most_expensive"] = max(costs)

        if quality_scores:
            summary["average_quality"] = statistics.mean(quality_scores)
            summary["highest_quality"] = max(quality_scores)
            summary["lowest_quality"] = min(quality_scores)

        # Find top performers
        if agent_summaries:
            top_by_success = max(agent_summaries, key=lambda x: x["success_rate"])
            top_by_speed = min(agent_summaries, key=lambda x: x["average_duration"])
            top_by_cost = min(agent_summaries, key=lambda x: x["cost_per_task"])

            summary["top_performers"] = {
                "highest_success_rate": top_by_success["agent_id"],
                "fastest_execution": top_by_speed["agent_id"],
                "most_cost_effective": top_by_cost["agent_id"],
            }

        result = {
            "agents": sorted(
                agent_summaries, key=lambda x: x["success_rate"], reverse=True
            ),
            "summary": summary,
            "total_agents": len(agent_summaries),
            "generated_at": datetime.now(UTC).isoformat(),
        }

        # Cache for 10 minutes
        self.analytics_cache[cache_key] = result
        self.cache_expiry[cache_key] = datetime.now(UTC) + timedelta(minutes=10)

        return result

    def get_task_type_analysis(self) -> Dict[str, Any]:
        """Analyze performance by task type across all agents"""
        cache_key = "task_type_analysis"
        if (
            cache_key in self.analytics_cache
            and cache_key in self.cache_expiry
            and datetime.now(UTC) < self.cache_expiry[cache_key]
        ):
            return self.analytics_cache[cache_key]

        task_type_stats = {}

        for profile in self.agent_profiles.values():
            for task_type, perf in profile.task_type_performance.items():
                if task_type not in task_type_stats:
                    task_type_stats[task_type] = {
                        "total_tasks": 0,
                        "successful_tasks": 0,
                        "agents": [],
                        "durations": [],
                        "costs": [],
                    }

                stats = task_type_stats[task_type]
                stats["total_tasks"] += perf["total_tasks"]
                stats["successful_tasks"] += perf["successful_tasks"]
                stats["durations"].append(perf["average_duration"])
                stats["costs"].append(perf["average_cost"])
                stats["agents"].append(
                    {
                        "agent_id": profile.agent_id,
                        "success_rate": perf["successful_tasks"]
                        / max(perf["total_tasks"], 1),
                        "average_duration": perf["average_duration"],
                        "average_cost": perf["average_cost"],
                    }
                )

        # Calculate aggregated statistics
        result = {}
        for task_type, stats in task_type_stats.items():
            success_rate = stats["successful_tasks"] / max(stats["total_tasks"], 1)
            avg_duration = (
                statistics.mean(stats["durations"]) if stats["durations"] else 0
            )
            avg_cost = statistics.mean(stats["costs"]) if stats["costs"] else 0

            # Find best agent for this task type
            best_agent = (
                max(stats["agents"], key=lambda x: x["success_rate"])
                if stats["agents"]
                else None
            )

            result[task_type] = {
                "total_tasks": stats["total_tasks"],
                "success_rate": success_rate,
                "average_duration": avg_duration,
                "average_cost": avg_cost,
                "best_agent": best_agent["agent_id"] if best_agent else None,
                "best_agent_success_rate": (
                    best_agent["success_rate"] if best_agent else 0
                ),
                "agent_count": len(stats["agents"]),
            }

        # Cache for 15 minutes
        self.analytics_cache[cache_key] = result
        self.cache_expiry[cache_key] = datetime.now(UTC) + timedelta(minutes=15)

        return result

    def _clear_agent_cache(self, agent_id: str):
        """Clear cached data for an agent"""
        keys_to_clear = [
            f"agent_perf_{agent_id}",
            "comparative_analysis",
            "task_type_analysis",
        ]
        for key in keys_to_clear:
            if key in self.analytics_cache:
                del self.analytics_cache[key]
            if key in self.cache_expiry:
                del self.cache_expiry[key]

    def get_cost_analysis(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Get cost analysis for specified time period"""
        cutoff_date = datetime.now(UTC) - timedelta(days=time_period_days)

        recent_executions = [
            exec for exec in self.performance_history if exec.start_time >= cutoff_date
        ]

        if not recent_executions:
            return {"total_cost": 0, "executions": 0, "agents": {}}

        total_cost = sum(exec.cost for exec in recent_executions)
        agent_costs = {}
        task_type_costs = {}

        for exec in recent_executions:
            # Agent costs
            if exec.agent_id not in agent_costs:
                agent_costs[exec.agent_id] = {"cost": 0, "tasks": 0, "success_rate": 0}
            agent_costs[exec.agent_id]["cost"] += exec.cost
            agent_costs[exec.agent_id]["tasks"] += 1

            # Task type costs
            if exec.task_type not in task_type_costs:
                task_type_costs[exec.task_type] = {"cost": 0, "tasks": 0}
            task_type_costs[exec.task_type]["cost"] += exec.cost
            task_type_costs[exec.task_type]["tasks"] += 1

        # Calculate success rates for cost analysis
        for agent_id in agent_costs:
            agent_executions = [
                exec for exec in recent_executions if exec.agent_id == agent_id
            ]
            successful = sum(1 for exec in agent_executions if exec.success)
            agent_costs[agent_id]["success_rate"] = successful / len(agent_executions)
            agent_costs[agent_id]["cost_per_task"] = (
                agent_costs[agent_id]["cost"] / agent_costs[agent_id]["tasks"]
            )

        return {
            "time_period_days": time_period_days,
            "total_cost": total_cost,
            "total_executions": len(recent_executions),
            "average_cost_per_task": total_cost / len(recent_executions),
            "agent_costs": agent_costs,
            "task_type_costs": task_type_costs,
            "most_expensive_agent": (
                max(agent_costs.items(), key=lambda x: x[1]["cost"])[0]
                if agent_costs
                else None
            ),
            "most_cost_effective_agent": (
                min(agent_costs.items(), key=lambda x: x[1]["cost_per_task"])[0]
                if agent_costs
                else None
            ),
        }

    async def generate_performance_report(
        self, agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if agent_id:
            # Single agent report
            agent_perf = self.get_agent_performance(agent_id)
            if not agent_perf:
                return {"error": f"Agent {agent_id} not found"}

            return {
                "report_type": "single_agent",
                "agent_id": agent_id,
                "performance": agent_perf,
                "recommendations": self._generate_agent_recommendations(agent_id),
                "generated_at": datetime.now(UTC).isoformat(),
            }
        else:
            # System-wide report
            comparative = self.get_comparative_analysis()
            task_analysis = self.get_task_type_analysis()
            cost_analysis = self.get_cost_analysis()

            return {
                "report_type": "system_wide",
                "comparative_analysis": comparative,
                "task_type_analysis": task_analysis,
                "cost_analysis": cost_analysis,
                "recommendations": self._generate_system_recommendations(),
                "generated_at": datetime.now(UTC).isoformat(),
            }

    def _generate_agent_recommendations(self, agent_id: str) -> List[str]:
        """Generate recommendations for improving agent performance"""
        profile = self.agent_profiles.get(agent_id)
        if not profile:
            return []

        recommendations = []
        success_rate = profile.successful_tasks / max(profile.total_tasks, 1)

        if success_rate < 0.7:
            recommendations.append(
                "Consider additional training or capability refinement"
            )

        if profile.average_duration > 30:
            recommendations.append("Optimize for faster execution times")

        if profile.average_quality_score < 0.6:
            recommendations.append("Focus on improving output quality")

        # Task-specific recommendations
        for task_type, perf in profile.task_type_performance.items():
            task_success_rate = perf["successful_tasks"] / max(perf["total_tasks"], 1)
            if task_success_rate < 0.6 and perf["total_tasks"] >= 3:
                recommendations.append(
                    f"Consider limiting {task_type} tasks or providing specialized training"
                )

        if not recommendations:
            recommendations.append("Performance is satisfactory - continue monitoring")

        return recommendations

    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []
        comparative = self.get_comparative_analysis()

        if not comparative["agents"]:
            return ["No performance data available"]

        summary = comparative["summary"]

        if summary.get("average_success_rate", 0) < 0.75:
            recommendations.append(
                "Overall success rate needs improvement - review task routing and agent capabilities"
            )

        if summary.get("average_duration", 0) > 25:
            recommendations.append(
                "Consider optimizing task distribution for faster execution"
            )

        # Cost optimization
        cost_analysis = self.get_cost_analysis()
        if cost_analysis["total_cost"] > 0:
            most_expensive = cost_analysis.get("most_expensive_agent")
            most_effective = cost_analysis.get("most_cost_effective_agent")
            if most_expensive and most_effective and most_expensive != most_effective:
                recommendations.append(
                    f"Consider routing more tasks to cost-effective agent: {most_effective}"
                )

        return recommendations or ["System performance is optimal"]
