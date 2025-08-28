"""
Cost Management Dashboard for Solo Developer Multi-Agent Orchestration

Tracks API usage, costs across agents, and provides cost optimization
recommendations and budget controls.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from src.core.events import create_event

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of costs"""

    API_CALLS = "api_calls"
    COMPUTE_TIME = "compute_time"
    STORAGE = "storage"
    EXTERNAL_SERVICES = "external_services"
    MODEL_INFERENCE = "model_inference"


class AlertLevel(Enum):
    """Cost alert levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class CostEntry:
    """Individual cost entry"""

    entry_id: str
    timestamp: datetime
    agent_id: str
    task_id: str
    task_type: str
    category: CostCategory
    amount: float
    currency: str = "USD"
    description: str = ""
    provider: str = ""  # e.g., "openai", "anthropic", "google"
    usage_metrics: dict[str, Any] = field(
        default_factory=dict
    )  # tokens, requests, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetLimit:
    """Budget limit configuration"""

    limit_id: str
    name: str
    amount: float
    period: str  # "daily", "weekly", "monthly"
    category: CostCategory | None = None
    agent_id: str | None = None  # None for system-wide
    alert_thresholds: list[float] = field(
        default_factory=lambda: [0.75, 0.9, 1.0]
    )  # Percentages
    auto_disable: bool = False  # Auto-disable agent when budget exceeded
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    active: bool = True


@dataclass
class CostAlert:
    """Cost alert notification"""

    alert_id: str
    timestamp: datetime
    level: AlertLevel
    message: str
    budget_limit_id: str | None = None
    agent_id: str | None = None
    current_amount: float = 0.0
    limit_amount: float = 0.0
    percentage_used: float = 0.0
    recommended_actions: list[str] = field(default_factory=list)


@dataclass
class UsageMetrics:
    """Usage metrics for cost analysis"""

    total_api_calls: int = 0
    total_tokens: int = 0
    total_requests: int = 0
    total_compute_minutes: float = 0.0
    peak_usage_hour: str | None = None
    average_cost_per_task: float = 0.0
    most_expensive_task_type: str | None = None


class CostManagementDashboard:
    """Dashboard for tracking and managing costs across agents"""

    def __init__(self, event_bus=None, default_currency: str = "USD"):
        self.event_bus = event_bus
        self.default_currency = default_currency
        self.cost_entries: list[CostEntry] = []
        self.budget_limits: dict[str, BudgetLimit] = {}
        self.cost_alerts: list[CostAlert] = []
        self.provider_rates: dict[str, dict[str, float]] = {}

        # Initialize default provider rates (can be updated)
        self._initialize_default_rates()

        # Cost tracking by different dimensions
        self.agent_costs: dict[str, float] = {}
        self.daily_costs: dict[str, float] = {}
        self.monthly_costs: dict[str, float] = {}

        # Usage tracking
        self.usage_metrics = UsageMetrics()

    def _initialize_default_rates(self):
        """Initialize default rates for common providers"""
        self.provider_rates = {
            "openai": {
                "gpt-4": 0.03,  # per 1K tokens
                "gpt-3.5-turbo": 0.002,
                "text-embedding": 0.0001,
            },
            "anthropic": {"claude-3": 0.015, "claude-instant": 0.008},
            "google": {"gemini-pro": 0.0005, "text-embedding": 0.00025},
            "compute": {"cpu_minute": 0.001, "gpu_minute": 0.01},
        }

    def update_provider_rates(self, provider: str, model: str, rate: float):
        """Update rate for a specific provider and model"""
        if provider not in self.provider_rates:
            self.provider_rates[provider] = {}
        self.provider_rates[provider][model] = rate
        logger.info(f"Updated rate for {provider}/{model}: ${rate}")

    def _generate_entry_id(self) -> str:
        """Generate unique cost entry ID"""
        import uuid

        return f"cost_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        import uuid

        return f"alert_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    async def record_cost(
        self,
        agent_id: str,
        task_id: str,
        task_type: str,
        category: CostCategory,
        amount: float,
        description: str = "",
        provider: str = "",
        usage_metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a cost entry"""
        entry_id = self._generate_entry_id()

        entry = CostEntry(
            entry_id=entry_id,
            timestamp=datetime.now(UTC),
            agent_id=agent_id,
            task_id=task_id,
            task_type=task_type,
            category=category,
            amount=amount,
            currency=self.default_currency,
            description=description,
            provider=provider,
            usage_metrics=usage_metrics or {},
            metadata=metadata or {},
        )

        self.cost_entries.append(entry)

        # Update aggregated costs
        self._update_aggregated_costs(entry)

        # Update usage metrics
        self._update_usage_metrics(entry)

        # Check budget limits
        await self._check_budget_limits(entry)

        # Emit cost event
        if self.event_bus:
            event = create_event(
                "cost_recorded",
                entry_id=entry_id,
                agent_id=agent_id,
                task_id=task_id,
                task_type=task_type,
                category=category.value,
                amount=amount,
                provider=provider,
                source_plugin="cost_management_dashboard",
            )
            await self.event_bus.publish(event)

        logger.debug(
            f"Recorded cost entry {entry_id}: ${amount} for {agent_id}/{task_type}"
        )
        return entry_id

    async def record_api_usage(
        self,
        agent_id: str,
        task_id: str,
        provider: str,
        model: str,
        tokens_used: int,
        requests_count: int = 1,
        task_type: str = "unknown",
    ) -> str:
        """Record API usage and calculate cost automatically"""
        # Calculate cost based on provider rates
        cost = 0.0
        if provider in self.provider_rates and model in self.provider_rates[provider]:
            rate = self.provider_rates[provider][model]
            cost = (tokens_used / 1000) * rate  # Most rates are per 1K tokens

        usage_metrics = {
            "tokens": tokens_used,
            "requests": requests_count,
            "model": model,
            "rate_per_1k_tokens": self.provider_rates.get(provider, {}).get(model, 0),
        }

        description = (
            f"{provider}/{model}: {tokens_used} tokens, {requests_count} requests"
        )

        return await self.record_cost(
            agent_id=agent_id,
            task_id=task_id,
            task_type=task_type,
            category=CostCategory.API_CALLS,
            amount=cost,
            description=description,
            provider=provider,
            usage_metrics=usage_metrics,
        )

    def _update_aggregated_costs(self, entry: CostEntry):
        """Update aggregated cost tracking"""
        # Agent costs
        if entry.agent_id not in self.agent_costs:
            self.agent_costs[entry.agent_id] = 0.0
        self.agent_costs[entry.agent_id] += entry.amount

        # Daily costs
        date_str = entry.timestamp.strftime("%Y-%m-%d")
        if date_str not in self.daily_costs:
            self.daily_costs[date_str] = 0.0
        self.daily_costs[date_str] += entry.amount

        # Monthly costs
        month_str = entry.timestamp.strftime("%Y-%m")
        if month_str not in self.monthly_costs:
            self.monthly_costs[month_str] = 0.0
        self.monthly_costs[month_str] += entry.amount

    def _update_usage_metrics(self, entry: CostEntry):
        """Update usage metrics"""
        self.usage_metrics.total_api_calls += entry.usage_metrics.get("requests", 0)
        self.usage_metrics.total_tokens += entry.usage_metrics.get("tokens", 0)

        if entry.category == CostCategory.COMPUTE_TIME:
            self.usage_metrics.total_compute_minutes += entry.usage_metrics.get(
                "minutes", 0
            )

    def create_budget_limit(
        self,
        name: str,
        amount: float,
        period: str,
        category: CostCategory | None = None,
        agent_id: str | None = None,
        alert_thresholds: list[float] | None = None,
        auto_disable: bool = False,
    ) -> str:
        """Create a new budget limit"""
        limit_id = f"budget_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{len(self.budget_limits)}"

        budget_limit = BudgetLimit(
            limit_id=limit_id,
            name=name,
            amount=amount,
            period=period,
            category=category,
            agent_id=agent_id,
            alert_thresholds=alert_thresholds or [0.75, 0.9, 1.0],
            auto_disable=auto_disable,
        )

        self.budget_limits[limit_id] = budget_limit
        logger.info(f"Created budget limit {limit_id}: {name} - ${amount}/{period}")
        return limit_id

    async def _check_budget_limits(self, entry: CostEntry):
        """Check if any budget limits are exceeded"""
        for _limit_id, budget_limit in self.budget_limits.items():
            if not budget_limit.active:
                continue

            # Check if this entry applies to this budget limit
            if budget_limit.agent_id and budget_limit.agent_id != entry.agent_id:
                continue

            if budget_limit.category and budget_limit.category != entry.category:
                continue

            # Calculate current usage for the budget period
            current_amount = self._calculate_period_cost(budget_limit)
            percentage_used = (
                current_amount / budget_limit.amount if budget_limit.amount > 0 else 0
            )

            # Check alert thresholds
            for threshold in budget_limit.alert_thresholds:
                if percentage_used >= threshold:
                    await self._create_budget_alert(
                        budget_limit, current_amount, percentage_used, threshold
                    )

    def _calculate_period_cost(self, budget_limit: BudgetLimit) -> float:
        """Calculate cost for the budget period"""
        now = datetime.now(UTC)

        if budget_limit.period == "daily":
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif budget_limit.period == "weekly":
            days_since_monday = now.weekday()
            start_date = now - timedelta(days=days_since_monday)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif budget_limit.period == "monthly":
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            # Default to daily
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)

        total_cost = 0.0
        for entry in self.cost_entries:
            if entry.timestamp < start_date:
                continue

            # Check filters
            if budget_limit.agent_id and entry.agent_id != budget_limit.agent_id:
                continue

            if budget_limit.category and entry.category != budget_limit.category:
                continue

            total_cost += entry.amount

        return total_cost

    async def _create_budget_alert(
        self,
        budget_limit: BudgetLimit,
        current_amount: float,
        percentage_used: float,
        threshold: float,
    ):
        """Create a budget alert"""
        alert_id = self._generate_alert_id()

        # Determine alert level
        if percentage_used >= 1.0:
            level = AlertLevel.BUDGET_EXCEEDED
        elif percentage_used >= 0.9:
            level = AlertLevel.CRITICAL
        elif percentage_used >= 0.75:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.INFO

        # Generate message
        scope = (
            f"agent {budget_limit.agent_id}" if budget_limit.agent_id else "system-wide"
        )
        category_str = (
            f" for {budget_limit.category.value}" if budget_limit.category else ""
        )
        message = f"Budget alert: {scope}{category_str} has used {percentage_used:.1%} of {budget_limit.period} budget (${current_amount:.2f}/${budget_limit.amount:.2f})"

        # Generate recommendations
        recommendations = []
        if percentage_used >= 1.0:
            recommendations.append(
                "Budget exceeded - consider increasing limit or reducing usage"
            )
            if budget_limit.auto_disable:
                recommendations.append("Auto-disable feature will be triggered")
        elif percentage_used >= 0.9:
            recommendations.append("Approaching budget limit - monitor usage closely")
        else:
            recommendations.append("Budget usage is high - consider optimizing tasks")

        if budget_limit.agent_id:
            recommendations.append(
                f"Review {budget_limit.agent_id} task efficiency and routing"
            )

        alert = CostAlert(
            alert_id=alert_id,
            timestamp=datetime.now(UTC),
            level=level,
            message=message,
            budget_limit_id=budget_limit.limit_id,
            agent_id=budget_limit.agent_id,
            current_amount=current_amount,
            limit_amount=budget_limit.amount,
            percentage_used=percentage_used,
            recommended_actions=recommendations,
        )

        self.cost_alerts.append(alert)

        # Auto-disable if configured
        if budget_limit.auto_disable and percentage_used >= 1.0:
            # This would need to integrate with agent management system
            logger.warning(f"Auto-disable triggered for budget {budget_limit.limit_id}")

        # Emit alert event
        if self.event_bus:
            event = create_event(
                "budget_alert",
                alert_id=alert_id,
                level=level.value,
                message=message,
                budget_limit_id=budget_limit.limit_id,
                agent_id=budget_limit.agent_id,
                current_amount=current_amount,
                limit_amount=budget_limit.amount,
                percentage_used=percentage_used,
                source_plugin="cost_management_dashboard",
            )
            await self.event_bus.publish(event)

        logger.warning(f"Budget alert {alert_id}: {message}")

    def get_cost_summary(self, time_period_days: int = 30) -> dict[str, Any]:
        """Get cost summary for specified time period"""
        cutoff_date = datetime.now(UTC) - timedelta(days=time_period_days)

        relevant_entries = [
            entry for entry in self.cost_entries if entry.timestamp >= cutoff_date
        ]

        if not relevant_entries:
            return {
                "time_period_days": time_period_days,
                "total_cost": 0,
                "entries_count": 0,
                "agents": {},
                "categories": {},
                "providers": {},
            }

        total_cost = sum(entry.amount for entry in relevant_entries)

        # Analyze by agent
        agent_analysis = {}
        for entry in relevant_entries:
            if entry.agent_id not in agent_analysis:
                agent_analysis[entry.agent_id] = {
                    "total_cost": 0,
                    "entries": 0,
                    "categories": {},
                    "avg_cost_per_task": 0,
                }

            agent_data = agent_analysis[entry.agent_id]
            agent_data["total_cost"] += entry.amount
            agent_data["entries"] += 1

            if entry.category.value not in agent_data["categories"]:
                agent_data["categories"][entry.category.value] = 0
            agent_data["categories"][entry.category.value] += entry.amount

        # Calculate averages
        for agent_data in agent_analysis.values():
            agent_data["avg_cost_per_task"] = (
                agent_data["total_cost"] / agent_data["entries"]
            )

        # Analyze by category
        category_analysis = {}
        for entry in relevant_entries:
            if entry.category.value not in category_analysis:
                category_analysis[entry.category.value] = {
                    "total_cost": 0,
                    "entries": 0,
                }
            category_analysis[entry.category.value]["total_cost"] += entry.amount
            category_analysis[entry.category.value]["entries"] += 1

        # Analyze by provider
        provider_analysis = {}
        for entry in relevant_entries:
            if entry.provider:
                if entry.provider not in provider_analysis:
                    provider_analysis[entry.provider] = {"total_cost": 0, "entries": 0}
                provider_analysis[entry.provider]["total_cost"] += entry.amount
                provider_analysis[entry.provider]["entries"] += 1

        return {
            "time_period_days": time_period_days,
            "total_cost": total_cost,
            "entries_count": len(relevant_entries),
            "average_cost_per_entry": total_cost / len(relevant_entries),
            "agents": agent_analysis,
            "categories": category_analysis,
            "providers": provider_analysis,
            "daily_average": total_cost / time_period_days,
            "projected_monthly": (total_cost / time_period_days) * 30,
        }

    def get_budget_status(self) -> dict[str, Any]:
        """Get status of all budget limits"""
        budget_status = {}

        for limit_id, budget_limit in self.budget_limits.items():
            current_amount = self._calculate_period_cost(budget_limit)
            percentage_used = (
                current_amount / budget_limit.amount if budget_limit.amount > 0 else 0
            )

            status = "healthy"
            if percentage_used >= 1.0:
                status = "exceeded"
            elif percentage_used >= 0.9:
                status = "critical"
            elif percentage_used >= 0.75:
                status = "warning"

            budget_status[limit_id] = {
                "name": budget_limit.name,
                "period": budget_limit.period,
                "amount": budget_limit.amount,
                "current_amount": current_amount,
                "percentage_used": percentage_used,
                "status": status,
                "agent_id": budget_limit.agent_id,
                "category": (
                    budget_limit.category.value if budget_limit.category else None
                ),
                "active": budget_limit.active,
                "auto_disable": budget_limit.auto_disable,
            }

        return budget_status

    def get_cost_optimization_recommendations(self) -> list[str]:
        """Generate cost optimization recommendations"""
        recommendations = []

        # Analyze recent costs
        recent_summary = self.get_cost_summary(7)  # Last 7 days

        if recent_summary["total_cost"] == 0:
            return ["No recent cost data available for analysis"]

        # Find most expensive agent
        if recent_summary["agents"]:
            most_expensive_agent = max(
                recent_summary["agents"].items(), key=lambda x: x[1]["total_cost"]
            )
            recommendations.append(
                f"Most expensive agent: {most_expensive_agent[0]} (${most_expensive_agent[1]['total_cost']:.2f})"
            )

        # Find most expensive category
        if recent_summary["categories"]:
            most_expensive_category = max(
                recent_summary["categories"].items(), key=lambda x: x[1]["total_cost"]
            )
            recommendations.append(
                f"Highest cost category: {most_expensive_category[0]} (${most_expensive_category[1]['total_cost']:.2f})"
            )

        # Provider recommendations
        if recent_summary["providers"]:
            provider_costs = [
                (k, v["total_cost"]) for k, v in recent_summary["providers"].items()
            ]
            if len(provider_costs) > 1:
                provider_costs.sort(key=lambda x: x[1], reverse=True)
                recommendations.append(
                    f"Consider optimizing usage of expensive provider: {provider_costs[0][0]}"
                )

        # Usage pattern recommendations
        projected_monthly = recent_summary["projected_monthly"]
        if projected_monthly > 100:  # Arbitrary threshold
            recommendations.append(
                f"High projected monthly cost: ${projected_monthly:.2f} - consider budget limits"
            )

        # Agent efficiency recommendations
        for agent_id, agent_data in recent_summary["agents"].items():
            if agent_data["avg_cost_per_task"] > 5:  # Arbitrary threshold
                recommendations.append(
                    f"Agent {agent_id} has high cost per task: ${agent_data['avg_cost_per_task']:.2f}"
                )

        return recommendations or ["Costs appear optimized"]

    def get_usage_analytics(self) -> dict[str, Any]:
        """Get detailed usage analytics"""
        # Calculate hourly usage patterns
        hourly_costs = {}
        for entry in self.cost_entries:
            hour = entry.timestamp.hour
            if hour not in hourly_costs:
                hourly_costs[hour] = 0
            hourly_costs[hour] += entry.amount

        peak_hour = (
            max(hourly_costs.items(), key=lambda x: x[1])[0] if hourly_costs else None
        )

        # Task type analysis
        task_type_costs = {}
        task_type_counts = {}
        for entry in self.cost_entries:
            task_type = entry.task_type
            if task_type not in task_type_costs:
                task_type_costs[task_type] = 0
                task_type_counts[task_type] = 0
            task_type_costs[task_type] += entry.amount
            task_type_counts[task_type] += 1

        # Calculate average cost per task type
        task_type_averages = {}
        for task_type in task_type_costs:
            task_type_averages[task_type] = (
                task_type_costs[task_type] / task_type_counts[task_type]
            )

        most_expensive_task_type = (
            max(task_type_averages.items(), key=lambda x: x[1])[0]
            if task_type_averages
            else None
        )

        return {
            "total_entries": len(self.cost_entries),
            "total_cost": sum(entry.amount for entry in self.cost_entries),
            "usage_metrics": {
                "total_api_calls": self.usage_metrics.total_api_calls,
                "total_tokens": self.usage_metrics.total_tokens,
                "total_compute_minutes": self.usage_metrics.total_compute_minutes,
                "average_cost_per_api_call": (
                    sum(
                        entry.amount
                        for entry in self.cost_entries
                        if entry.category == CostCategory.API_CALLS
                    )
                    / max(self.usage_metrics.total_api_calls, 1)
                ),
                "cost_per_1k_tokens": (
                    (
                        sum(
                            entry.amount
                            for entry in self.cost_entries
                            if entry.category == CostCategory.API_CALLS
                        )
                        / max(self.usage_metrics.total_tokens / 1000, 1)
                    )
                    if self.usage_metrics.total_tokens > 0
                    else 0
                ),
            },
            "hourly_distribution": hourly_costs,
            "peak_usage_hour": peak_hour,
            "task_type_costs": task_type_costs,
            "task_type_averages": task_type_averages,
            "most_expensive_task_type": most_expensive_task_type,
            "recent_alerts": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "level": alert.level.value,
                    "message": alert.message,
                }
                for alert in self.cost_alerts[-10:]  # Last 10 alerts
            ],
        }

    def export_cost_data(
        self, start_date: datetime, end_date: datetime
    ) -> list[dict[str, Any]]:
        """Export cost data for specified date range"""
        relevant_entries = [
            entry
            for entry in self.cost_entries
            if start_date <= entry.timestamp <= end_date
        ]

        return [
            {
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp.isoformat(),
                "agent_id": entry.agent_id,
                "task_id": entry.task_id,
                "task_type": entry.task_type,
                "category": entry.category.value,
                "amount": entry.amount,
                "currency": entry.currency,
                "description": entry.description,
                "provider": entry.provider,
                "usage_metrics": entry.usage_metrics,
                "metadata": entry.metadata,
            }
            for entry in relevant_entries
        ]
