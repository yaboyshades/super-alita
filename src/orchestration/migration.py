"""Gradual Migration Framework with A/B testing and canary releases.

Implements safe migration between component versions:
- A/B testing with traffic splitting
- Canary releases (gradual rollout)
- Performance-based ramping
- Automatic rollback on degradation
- Constitutional compliance gates
"""

from __future__ import annotations

import asyncio
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class MigrationStrategy(str, Enum):
    """Migration rollout strategies."""

    AB_TEST = "ab_test"  # Fixed split for testing
    CANARY = "canary"  # Gradual increase
    BLUE_GREEN = "blue_green"  # Instant switch
    SHADOW = "shadow"  # Mirror traffic to new version


class MigrationStatus(str, Enum):
    """Migration lifecycle states."""

    PLANNING = "planning"
    ACTIVE = "active"
    RAMPING = "ramping"
    COMPLETE = "complete"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"


@dataclass
class VersionMetrics:
    """Metrics for a component version."""

    version: str
    requests: int = 0
    successes: int = 0
    failures: int = 0
    avg_latency_ms: float = 0.0
    constitutional_score: float = 0.0
    error_rate: float = 0.0

    # History
    latency_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    success_history: deque[bool] = field(
        default_factory=lambda: deque(maxlen=100)
    )


@dataclass
class MigrationConfig:
    """Configuration for a migration."""

    # Versions
    old_version: str
    new_version: str

    # Strategy
    strategy: MigrationStrategy = MigrationStrategy.CANARY
    initial_traffic_percent: float = 5.0  # Start at 5%
    target_traffic_percent: float = 100.0  # End at 100%
    ramp_step_percent: float = 10.0  # Increase by 10% each step
    ramp_interval_seconds: float = 300.0  # 5 minutes between steps

    # Safety thresholds
    max_error_rate: float = 0.05  # 5% error rate triggers rollback
    max_latency_increase_percent: float = 50.0  # 50% latency increase
    min_constitutional_score: float = 0.75

    # Minimum requests before ramping
    min_requests_per_step: int = 100


@dataclass
class MigrationState:
    """Current state of a migration."""

    config: MigrationConfig
    status: MigrationStatus = MigrationStatus.PLANNING
    current_traffic_percent: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_ramp_at: datetime | None = None

    # Metrics
    old_metrics: VersionMetrics = field(init=False)
    new_metrics: VersionMetrics = field(init=False)

    def __post_init__(self):
        """Initialize metrics."""
        self.old_metrics = VersionMetrics(version=self.config.old_version)
        self.new_metrics = VersionMetrics(version=self.config.new_version)


class MigrationFramework:
    """Gradual migration framework with A/B testing and canary releases.

    Safely migrates traffic between component versions with:
    - Automatic performance monitoring
    - Rollback on degradation
    - Constitutional compliance gates
    - Configurable ramp strategies
    """

    def __init__(
        self,
        constitutional_validator: Callable[[str, str], float] | None = None,
    ):
        """Initialize migration framework.

        Args:
            constitutional_validator: Function to validate compliance
        """
        self.constitutional_validator = constitutional_validator

        # Active migrations
        self.migrations: dict[str, MigrationState] = {}

        # Statistics
        self.total_migrations = 0
        self.successful_migrations = 0
        self.rollbacks = 0

    def start_migration(
        self, component_id: str, config: MigrationConfig
    ) -> MigrationState:
        """Start a new migration.

        Args:
            component_id: Component being migrated
            config: Migration configuration

        Returns:
            MigrationState tracking migration progress
        """
        if component_id in self.migrations:
            existing = self.migrations[component_id]
            if existing.status in {
                MigrationStatus.ACTIVE,
                MigrationStatus.RAMPING,
            }:
                raise ValueError(
                    f"Migration already active for {component_id}"
                )

        state = MigrationState(config=config)
        state.status = MigrationStatus.ACTIVE
        state.started_at = datetime.now(UTC)
        state.current_traffic_percent = config.initial_traffic_percent

        self.migrations[component_id] = state
        self.total_migrations += 1

        return state

    async def route_request(self, component_id: str, request_id: str) -> str:
        """Route request to appropriate version.

        Args:
            component_id: Component identifier
            request_id: Request identifier

        Returns:
            Version to use (old_version or new_version)
        """
        if component_id not in self.migrations:
            # No migration, use default
            return "default"

        state = self.migrations[component_id]

        if state.status not in {
            MigrationStatus.ACTIVE,
            MigrationStatus.RAMPING,
        }:
            # Migration not active
            return state.config.old_version

        # Determine version based on strategy
        if state.config.strategy == MigrationStrategy.SHADOW:
            # Shadow mode: always old version (new version gets copy)
            return state.config.old_version

        elif state.config.strategy == MigrationStrategy.BLUE_GREEN:
            # Blue-green: instant switch
            if state.current_traffic_percent >= 100.0:
                return state.config.new_version
            return state.config.old_version

        else:
            # A/B test or canary: probabilistic routing
            roll = random.random() * 100.0
            if roll < state.current_traffic_percent:
                return state.config.new_version
            return state.config.old_version

    async def record_result(
        self,
        component_id: str,
        version: str,
        success: bool,
        latency_ms: float,
        constitutional_score: float | None = None,
    ) -> None:
        """Record request result for a version.

        Args:
            component_id: Component identifier
            version: Version that handled request
            success: Whether request succeeded
            latency_ms: Request latency
            constitutional_score: Optional compliance score
        """
        if component_id not in self.migrations:
            return

        state = self.migrations[component_id]

        # Determine which metrics to update
        if version == state.config.new_version:
            metrics = state.new_metrics
        else:
            metrics = state.old_metrics

        # Update metrics
        metrics.requests += 1
        if success:
            metrics.successes += 1
        else:
            metrics.failures += 1

        metrics.success_history.append(success)
        metrics.latency_history.append(latency_ms)

        # Update averages (exponential moving average)
        alpha = 0.2
        if metrics.avg_latency_ms == 0.0:
            metrics.avg_latency_ms = latency_ms
        else:
            metrics.avg_latency_ms = (
                alpha * latency_ms + (1 - alpha) * metrics.avg_latency_ms
            )

        metrics.error_rate = (
            metrics.failures / metrics.requests
            if metrics.requests > 0
            else 0.0
        )

        if constitutional_score is not None:
            if metrics.constitutional_score == 0.0:
                metrics.constitutional_score = constitutional_score
            else:
                metrics.constitutional_score = (
                    alpha * constitutional_score
                    + (1 - alpha) * metrics.constitutional_score
                )

    async def check_and_ramp(self, component_id: str) -> bool:
        """Check if migration should ramp up.

        Args:
            component_id: Component identifier

        Returns:
            True if ramp occurred, False otherwise
        """
        if component_id not in self.migrations:
            return False

        state = self.migrations[component_id]

        if state.status != MigrationStatus.ACTIVE:
            return False

        # Check if ready to ramp
        now = datetime.now(UTC)

        if state.last_ramp_at:
            elapsed = (now - state.last_ramp_at).total_seconds()
            if elapsed < state.config.ramp_interval_seconds:
                return False

        # Check minimum requests
        if state.new_metrics.requests < state.config.min_requests_per_step:
            return False

        # Safety checks
        if not await self._is_safe_to_ramp(state):
            # Rollback
            await self.rollback(component_id, "Safety threshold exceeded")
            return False

        # Ramp up
        state.current_traffic_percent = min(
            state.current_traffic_percent + state.config.ramp_step_percent,
            state.config.target_traffic_percent,
        )
        state.last_ramp_at = now
        state.status = MigrationStatus.RAMPING

        # Check if complete
        if (
            state.current_traffic_percent
            >= state.config.target_traffic_percent
        ):
            await self.complete_migration(component_id)

        return True

    async def _is_safe_to_ramp(self, state: MigrationState) -> bool:
        """Check if it's safe to ramp up traffic.

        Args:
            state: Migration state

        Returns:
            True if safe to ramp, False if rollback needed
        """
        new_metrics = state.new_metrics
        old_metrics = state.old_metrics

        # Check error rate
        if new_metrics.error_rate > state.config.max_error_rate:
            return False

        # Check latency increase
        if old_metrics.avg_latency_ms > 0:
            latency_increase = (
                (new_metrics.avg_latency_ms - old_metrics.avg_latency_ms)
                / old_metrics.avg_latency_ms
                * 100.0
            )
            if latency_increase > state.config.max_latency_increase_percent:
                return False

        # Check constitutional score
        return (
            not new_metrics.constitutional_score
            < state.config.min_constitutional_score
        )

    async def rollback(self, component_id: str, reason: str) -> MigrationState:
        """Rollback a migration.

        Args:
            component_id: Component identifier
            reason: Reason for rollback

        Returns:
            Updated MigrationState
        """
        if component_id not in self.migrations:
            raise ValueError(f"No migration found for {component_id}")

        state = self.migrations[component_id]
        state.status = MigrationStatus.ROLLING_BACK
        state.current_traffic_percent = 0.0  # Route all to old version

        self.rollbacks += 1

        # After a delay, mark as failed
        await asyncio.sleep(1.0)
        state.status = MigrationStatus.FAILED
        state.completed_at = datetime.now(UTC)

        return state

    async def complete_migration(self, component_id: str) -> MigrationState:
        """Complete a migration successfully.

        Args:
            component_id: Component identifier

        Returns:
            Updated MigrationState
        """
        if component_id not in self.migrations:
            raise ValueError(f"No migration found for {component_id}")

        state = self.migrations[component_id]
        state.status = MigrationStatus.COMPLETE
        state.current_traffic_percent = 100.0
        state.completed_at = datetime.now(UTC)

        self.successful_migrations += 1

        return state

    def get_migration_status(self, component_id: str) -> dict[str, Any] | None:
        """Get current status of a migration.

        Args:
            component_id: Component identifier

        Returns:
            Status dictionary or None if no migration
        """
        if component_id not in self.migrations:
            return None

        state = self.migrations[component_id]

        return {
            "component_id": component_id,
            "status": state.status,
            "old_version": state.config.old_version,
            "new_version": state.config.new_version,
            "strategy": state.config.strategy,
            "current_traffic_percent": state.current_traffic_percent,
            "started_at": (
                state.started_at.isoformat() if state.started_at else None
            ),
            "old_metrics": {
                "requests": state.old_metrics.requests,
                "success_rate": (
                    state.old_metrics.successes / state.old_metrics.requests
                    if state.old_metrics.requests > 0
                    else 0.0
                ),
                "avg_latency_ms": state.old_metrics.avg_latency_ms,
                "error_rate": state.old_metrics.error_rate,
                "constitutional_score": state.old_metrics.constitutional_score,
            },
            "new_metrics": {
                "requests": state.new_metrics.requests,
                "success_rate": (
                    state.new_metrics.successes / state.new_metrics.requests
                    if state.new_metrics.requests > 0
                    else 0.0
                ),
                "avg_latency_ms": state.new_metrics.avg_latency_ms,
                "error_rate": state.new_metrics.error_rate,
                "constitutional_score": state.new_metrics.constitutional_score,
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Get migration framework statistics."""
        return {
            "total_migrations": self.total_migrations,
            "successful_migrations": self.successful_migrations,
            "rollbacks": self.rollbacks,
            "active_migrations": sum(
                1
                for state in self.migrations.values()
                if state.status
                in {MigrationStatus.ACTIVE, MigrationStatus.RAMPING}
            ),
        }


# Example usage
async def example_migration() -> None:
    """Example demonstrating MigrationFramework."""
    framework = MigrationFramework()

    # Start canary migration
    config = MigrationConfig(
        old_version="v1.0",
        new_version="v2.0",
        strategy=MigrationStrategy.CANARY,
        initial_traffic_percent=10.0,
        ramp_step_percent=20.0,
        ramp_interval_seconds=5.0,  # Fast for demo
        min_requests_per_step=20,
    )

    state = framework.start_migration("my_service", config)
    print(f"Started migration: {state.status}")

    # Simulate requests
    for i in range(200):
        # Route request
        version = await framework.route_request("my_service", f"req_{i}")

        # Simulate processing
        await asyncio.sleep(0.05)

        # Record result (v2 slightly worse for demo)
        success = random.random() > 0.05  # 95% success
        latency = (
            10.0 + random.random() * 5.0
            if version == "v1.0"
            else 12.0 + random.random() * 5.0
        )
        constitutional = 0.85 + random.random() * 0.1

        await framework.record_result(
            "my_service", version, success, latency, constitutional
        )

        # Check for ramp every 25 requests
        if i % 25 == 0:
            ramped = await framework.check_and_ramp("my_service")
            if ramped:
                status = framework.get_migration_status("my_service")
                print(
                    f"Request {i}: Ramped to "
                    f"{status['current_traffic_percent']:.1f}%"  # type: ignore[index]
                )

    # Final status
    print("\nFinal Migration Status:")
    import json

    status = framework.get_migration_status("my_service")
    print(json.dumps(status, indent=2))

    # Framework stats
    print("\nFramework Stats:")
    print(json.dumps(framework.get_stats(), indent=2))


if __name__ == "__main__":
    asyncio.run(example_migration())
