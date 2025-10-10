"""Constitutional Load Balancer with predictive routing and fairness.

Implements intelligent load balancing that considers:
- Constitutional compliance history
- Component health and capacity
- Fairness scoring (prevent starvation)
- Predictive routing based on past performance
- Adaptive load distribution
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class RoutingStrategy(str, Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CONSTITUTIONAL_WEIGHTED = "constitutional_weighted"
    PREDICTIVE = "predictive"
    FAIRNESS_AWARE = "fairness_aware"


@dataclass
class ComponentMetrics:
    """Metrics for a single component."""

    component_id: str
    current_load: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    constitutional_score: float = 1.0
    last_request_time: datetime | None = None

    # Fairness tracking
    requests_in_window: int = 0
    fairness_score: float = 1.0

    # History for predictive routing
    latency_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    success_history: deque[bool] = field(
        default_factory=lambda: deque(maxlen=100)
    )


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    component_id: str
    strategy: RoutingStrategy
    confidence: float
    reasoning: str
    metrics: dict[str, Any] = field(default_factory=dict)


class ConstitutionalLoadBalancer:
    """Load balancer with constitutional compliance awareness.

    Routes requests to components based on:
    - Health and capacity
    - Constitutional compliance history
    - Fairness (prevent overloading or starving components)
    - Predictive performance modeling
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.CONSTITUTIONAL_WEIGHTED,
        fairness_window_seconds: float = 60.0,
        constitutional_weight: float = 0.4,
        performance_weight: float = 0.3,
        fairness_weight: float = 0.3,
    ):
        """Initialize load balancer.

        Args:
            strategy: Default routing strategy
            fairness_window_seconds: Time window for fairness calculation
            constitutional_weight: Weight for constitutional compliance
            performance_weight: Weight for performance metrics
            fairness_weight: Weight for fairness scoring
        """
        self.strategy = strategy
        self.fairness_window_seconds = fairness_window_seconds
        self.constitutional_weight = constitutional_weight
        self.performance_weight = performance_weight
        self.fairness_weight = fairness_weight

        # Component tracking
        self.components: dict[str, ComponentMetrics] = {}
        self.component_order: list[str] = []  # For round-robin
        self.current_index = 0

        # Request history for fairness
        self.request_timestamps: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Statistics
        self.total_routes = 0
        self.strategy_counts: dict[RoutingStrategy, int] = defaultdict(int)

    def register_component(
        self, component_id: str, capacity: int = 100
    ) -> None:
        """Register a component with the load balancer.

        Args:
            component_id: Component identifier
            capacity: Maximum concurrent requests
        """
        if component_id not in self.components:
            self.components[component_id] = ComponentMetrics(
                component_id=component_id
            )
            self.component_order.append(component_id)

    def unregister_component(self, component_id: str) -> None:
        """Remove a component from load balancing.

        Args:
            component_id: Component identifier
        """
        if component_id in self.components:
            del self.components[component_id]
            if component_id in self.component_order:
                self.component_order.remove(component_id)

    async def route(
        self,
        request_id: str | None = None,
        strategy: RoutingStrategy | None = None,
    ) -> RoutingDecision:
        """Route a request to a component.

        Args:
            request_id: Optional request identifier
            strategy: Override default strategy

        Returns:
            RoutingDecision with selected component
        """
        strategy = strategy or self.strategy
        self.total_routes += 1
        self.strategy_counts[strategy] += 1

        if not self.components:
            raise ValueError("No components registered")

        # Update fairness scores
        await self._update_fairness_scores()

        # Select component based on strategy
        if strategy == RoutingStrategy.ROUND_ROBIN:
            decision = self._route_round_robin()
        elif strategy == RoutingStrategy.LEAST_LOADED:
            decision = self._route_least_loaded()
        elif strategy == RoutingStrategy.CONSTITUTIONAL_WEIGHTED:
            decision = self._route_constitutional()
        elif strategy == RoutingStrategy.PREDICTIVE:
            decision = self._route_predictive()
        elif strategy == RoutingStrategy.FAIRNESS_AWARE:
            decision = self._route_fairness_aware()
        else:
            decision = self._route_constitutional()

        # Record request
        component = self.components[decision.component_id]
        component.current_load += 1
        component.total_requests += 1
        component.last_request_time = datetime.now(UTC)

        now = time.time()
        self.request_timestamps[decision.component_id].append(now)

        return decision

    def _route_round_robin(self) -> RoutingDecision:
        """Simple round-robin routing."""
        if not self.component_order:
            raise ValueError("No components available")

        component_id = self.component_order[self.current_index]
        self.current_index = (self.current_index + 1) % len(
            self.component_order
        )

        return RoutingDecision(
            component_id=component_id,
            strategy=RoutingStrategy.ROUND_ROBIN,
            confidence=1.0,
            reasoning="Round-robin selection",
        )

    def _route_least_loaded(self) -> RoutingDecision:
        """Route to component with lowest current load."""
        component = min(self.components.values(), key=lambda c: c.current_load)

        return RoutingDecision(
            component_id=component.component_id,
            strategy=RoutingStrategy.LEAST_LOADED,
            confidence=0.8,
            reasoning=f"Least loaded: {component.current_load} requests",
            metrics={"current_load": component.current_load},
        )

    def _route_constitutional(self) -> RoutingDecision:
        """Route based on constitutional compliance scores."""
        # Score = constitutional_score * (1 - load_ratio)
        scores = {}
        for comp_id, comp in self.components.items():
            load_ratio = comp.current_load / 100.0  # Normalize by capacity
            scores[comp_id] = comp.constitutional_score * (1.0 - load_ratio)

        best_id = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_comp = self.components[best_id]

        return RoutingDecision(
            component_id=best_id,
            strategy=RoutingStrategy.CONSTITUTIONAL_WEIGHTED,
            confidence=0.85,
            reasoning=f"Constitutional score: {best_comp.constitutional_score:.2f}",
            metrics={
                "constitutional_score": best_comp.constitutional_score,
                "composite_score": scores[best_id],
            },
        )

    def _route_predictive(self) -> RoutingDecision:
        """Route based on predicted performance."""
        # Predict: success_rate * (1 / avg_latency)
        scores = {}
        for comp_id, comp in self.components.items():
            if comp.total_requests == 0:
                scores[comp_id] = 1.0  # New component
                continue

            success_rate = (
                comp.successful_requests / comp.total_requests
                if comp.total_requests > 0
                else 1.0
            )
            latency_factor = 1.0 / (
                comp.avg_latency_ms + 1.0
            )  # Avoid div by zero

            scores[comp_id] = success_rate * latency_factor

        best_id = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_comp = self.components[best_id]

        return RoutingDecision(
            component_id=best_id,
            strategy=RoutingStrategy.PREDICTIVE,
            confidence=0.9,
            reasoning=f"Predicted best performance: {scores[best_id]:.3f}",
            metrics={
                "success_rate": (
                    best_comp.successful_requests / best_comp.total_requests
                    if best_comp.total_requests > 0
                    else 1.0
                ),
                "avg_latency_ms": best_comp.avg_latency_ms,
            },
        )

    def _route_fairness_aware(self) -> RoutingDecision:
        """Route with fairness consideration (prevent starvation)."""
        # Composite score: constitutional * performance * fairness
        scores = {}
        for comp_id, comp in self.components.items():
            constitutional = comp.constitutional_score
            performance = (
                comp.successful_requests / comp.total_requests
                if comp.total_requests > 0
                else 1.0
            )
            fairness = comp.fairness_score

            scores[comp_id] = (
                self.constitutional_weight * constitutional
                + self.performance_weight * performance
                + self.fairness_weight * fairness
            )

        best_id = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_comp = self.components[best_id]

        return RoutingDecision(
            component_id=best_id,
            strategy=RoutingStrategy.FAIRNESS_AWARE,
            confidence=0.95,
            reasoning=f"Fairness-aware composite score: {scores[best_id]:.3f}",
            metrics={
                "constitutional_score": best_comp.constitutional_score,
                "fairness_score": best_comp.fairness_score,
                "composite_score": scores[best_id],
            },
        )

    async def _update_fairness_scores(self) -> None:
        """Update fairness scores for all components."""
        now = time.time()
        cutoff = now - self.fairness_window_seconds

        total_recent = 0
        for comp_id, timestamps in self.request_timestamps.items():
            # Count recent requests
            recent = sum(1 for ts in timestamps if ts >= cutoff)
            if comp_id in self.components:
                self.components[comp_id].requests_in_window = recent
            total_recent += recent

        if total_recent == 0:
            # No recent requests, all equal
            for comp in self.components.values():
                comp.fairness_score = 1.0
            return

        # Fairness score = 1 - (deviation from fair share)
        fair_share = total_recent / len(self.components)
        for comp in self.components.values():
            deviation = abs(comp.requests_in_window - fair_share) / (
                fair_share + 1.0
            )
            comp.fairness_score = max(0.0, 1.0 - deviation)

    async def complete_request(
        self,
        component_id: str,
        success: bool,
        latency_ms: float,
        constitutional_score: float | None = None,
    ) -> None:
        """Record completion of a request.

        Args:
            component_id: Component that handled request
            success: Whether request succeeded
            latency_ms: Request latency in milliseconds
            constitutional_score: Optional constitutional compliance score
        """
        if component_id not in self.components:
            return

        comp = self.components[component_id]
        comp.current_load = max(0, comp.current_load - 1)

        if success:
            comp.successful_requests += 1
        else:
            comp.failed_requests += 1

        # Update latency (exponential moving average)
        alpha = 0.2
        if comp.avg_latency_ms == 0.0:
            comp.avg_latency_ms = latency_ms
        else:
            comp.avg_latency_ms = (
                alpha * latency_ms + (1 - alpha) * comp.avg_latency_ms
            )

        # Update history
        comp.latency_history.append(latency_ms)
        comp.success_history.append(success)

        # Update constitutional score if provided
        if constitutional_score is not None:
            comp.constitutional_score = (
                alpha * constitutional_score
                + (1 - alpha) * comp.constitutional_score
            )

    def get_stats(self) -> dict[str, Any]:
        """Get load balancer statistics."""
        return {
            "total_routes": self.total_routes,
            "strategy_counts": dict(self.strategy_counts),
            "components": len(self.components),
            "component_stats": {
                comp_id: {
                    "current_load": comp.current_load,
                    "total_requests": comp.total_requests,
                    "success_rate": (
                        comp.successful_requests / comp.total_requests
                        if comp.total_requests > 0
                        else 0.0
                    ),
                    "avg_latency_ms": comp.avg_latency_ms,
                    "constitutional_score": comp.constitutional_score,
                    "fairness_score": comp.fairness_score,
                }
                for comp_id, comp in self.components.items()
            },
        }

    async def health_check(self) -> dict[str, Any]:
        """Health check for load balancer."""
        await self._update_fairness_scores()

        healthy_components = sum(
            1
            for comp in self.components.values()
            if comp.constitutional_score >= 0.75
        )

        return {
            "status": "healthy" if healthy_components > 0 else "degraded",
            "total_components": len(self.components),
            "healthy_components": healthy_components,
            "total_routes": self.total_routes,
            "active_load": sum(
                comp.current_load for comp in self.components.values()
            ),
        }


# Example usage
async def example_load_balancer() -> None:
    """Example demonstrating ConstitutionalLoadBalancer."""
    lb = ConstitutionalLoadBalancer(
        strategy=RoutingStrategy.FAIRNESS_AWARE,
        constitutional_weight=0.4,
        performance_weight=0.3,
        fairness_weight=0.3,
    )

    # Register components
    lb.register_component("service_a")
    lb.register_component("service_b")
    lb.register_component("service_c")

    # Simulate requests
    for i in range(20):
        decision = await lb.route(f"req_{i}")
        print(
            f"Request {i}: {decision.component_id} "
            f"({decision.strategy}, confidence={decision.confidence:.2f})"
        )

        # Simulate processing
        await asyncio.sleep(0.01)

        # Complete request with varying success/latency
        success = i % 5 != 0  # Every 5th request fails
        latency = 10.0 + (i % 10) * 2.0
        constitutional = 0.85 + (i % 5) * 0.03

        await lb.complete_request(
            decision.component_id, success, latency, constitutional
        )

    # Stats
    print("\nLoad Balancer Stats:")
    import json

    print(json.dumps(lb.get_stats(), indent=2))

    # Health check
    health = await lb.health_check()
    print("\nHealth Check:")
    print(json.dumps(health, indent=2))


if __name__ == "__main__":
    asyncio.run(example_load_balancer())
