"""
Scalability Manager Module

Provides horizontal scaling and load balancing capabilities:
- Service discovery and registration
- Load balancing algorithms (round-robin, weighted, least-connections)
- Auto-scaling based on metrics
- Circuit breaker pattern for fault tolerance
- Health monitoring and failover
- Distributed coordination
"""

import asyncio
import logging
import random
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms"""

    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    CONSISTENT_HASH = "consistent_hash"


@dataclass
class ServiceInstance:
    """Service instance representation"""

    id: str
    host: str
    port: int
    weight: float = 1.0
    status: ServiceStatus = ServiceStatus.UNKNOWN
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_health_check: datetime | None = None
    response_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return 1.0 - (self.failed_requests / self.total_requests)


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration"""

    metric_name: str
    threshold_scale_up: float
    threshold_scale_down: float
    min_instances: int = 1
    max_instances: int = 10
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    evaluation_window: int = 300  # seconds


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance"""

    failure_threshold: int = 5
    timeout_seconds: int = 60
    half_open_max_calls: int = 3

    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: datetime | None = None
    half_open_calls: int = 0

    def should_allow_request(self) -> bool:
        """Check if request should be allowed"""
        now = datetime.now()

        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if (
                self.last_failure_time
                and (now - self.last_failure_time).total_seconds()
                > self.timeout_seconds
            ):
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls

        return False

    def record_success(self):
        """Record successful request"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN


class ServiceRegistry:
    """Service discovery and registration"""

    def __init__(self):
        self.services: dict[str, list[ServiceInstance]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._watchers: dict[str, list[Callable]] = defaultdict(list)

    async def register_service(
        self, service_name: str, instance: ServiceInstance
    ):
        """Register service instance"""
        async with self._lock:
            if instance not in self.services[service_name]:
                self.services[service_name].append(instance)
                logger.info(
                    f"Registered service instance: {service_name} -> {instance.endpoint}"
                )

                # Notify watchers
                for watcher in self._watchers[service_name]:
                    try:
                        await watcher("register", instance)
                    except Exception as e:
                        logger.error(f"Error notifying watcher: {e}")

    async def deregister_service(self, service_name: str, instance_id: str):
        """Deregister service instance"""
        async with self._lock:
            instances = self.services[service_name]
            for i, instance in enumerate(instances):
                if instance.id == instance_id:
                    removed = instances.pop(i)
                    logger.info(
                        f"Deregistered service instance: {service_name} -> {removed.endpoint}"
                    )

                    # Notify watchers
                    for watcher in self._watchers[service_name]:
                        try:
                            await watcher("deregister", removed)
                        except Exception as e:
                            logger.error(f"Error notifying watcher: {e}")
                    break

    async def get_service_instances(
        self, service_name: str, healthy_only: bool = True
    ) -> list[ServiceInstance]:
        """Get service instances"""
        async with self._lock:
            instances = self.services[service_name].copy()

            if healthy_only:
                instances = [
                    i for i in instances if i.status == ServiceStatus.HEALTHY
                ]

            return instances

    async def update_instance_status(
        self, service_name: str, instance_id: str, status: ServiceStatus
    ):
        """Update instance health status"""
        async with self._lock:
            for instance in self.services[service_name]:
                if instance.id == instance_id:
                    old_status = instance.status
                    instance.status = status
                    instance.last_health_check = datetime.now()

                    if old_status != status:
                        logger.info(
                            f"Service {service_name}/{instance_id} status: {old_status} -> {status}"
                        )
                    break

    def watch_service(self, service_name: str, callback: Callable):
        """Watch for service changes"""
        self._watchers[service_name].append(callback)


class LoadBalancer:
    """Load balancer with multiple algorithms"""

    def __init__(
        self,
        algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN,
    ):
        self.algorithm = algorithm
        self._round_robin_counters: dict[str, int] = defaultdict(int)
        self._consistent_hash_rings: dict[str, list[tuple]] = defaultdict(list)

    async def select_instance(
        self,
        service_name: str,
        instances: list[ServiceInstance],
        context: dict[str, Any] | None = None,
    ) -> ServiceInstance | None:
        """Select instance using configured algorithm"""
        if not instances:
            return None

        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin_select(service_name, instances)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(service_name, instances)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections_select(instances)
        elif (
            self.algorithm == LoadBalancingAlgorithm.WEIGHTED_LEAST_CONNECTIONS
        ):
            return self._weighted_least_connections_select(instances)
        elif self.algorithm == LoadBalancingAlgorithm.RANDOM:
            return random.choice(instances)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_RANDOM:
            return self._weighted_random_select(instances)
        elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            return self._consistent_hash_select(
                service_name, instances, context
            )

        return instances[0]  # Fallback

    def _round_robin_select(
        self, service_name: str, instances: list[ServiceInstance]
    ) -> ServiceInstance:
        """Round-robin selection"""
        index = self._round_robin_counters[service_name] % len(instances)
        self._round_robin_counters[service_name] += 1
        return instances[index]

    def _weighted_round_robin_select(
        self, service_name: str, instances: list[ServiceInstance]
    ) -> ServiceInstance:
        """Weighted round-robin selection"""
        # Simple weighted selection based on instance weights
        weights = [max(instance.weight, 0.1) for instance in instances]
        total_weight = sum(weights)

        # Use counter to distribute requests proportionally
        counter = self._round_robin_counters[service_name]
        self._round_robin_counters[service_name] += 1

        # Calculate weighted position
        position = (counter * max(weights)) % total_weight
        cumulative_weight = 0

        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if position < cumulative_weight:
                return instances[i]

        return instances[0]

    def _least_connections_select(
        self, instances: list[ServiceInstance]
    ) -> ServiceInstance:
        """Select instance with least connections"""
        return min(instances, key=lambda x: x.current_connections)

    def _weighted_least_connections_select(
        self, instances: list[ServiceInstance]
    ) -> ServiceInstance:
        """Select instance with least weighted connections"""

        def weighted_connections(instance):
            if instance.weight <= 0:
                return float("inf")
            return instance.current_connections / instance.weight

        return min(instances, key=weighted_connections)

    def _weighted_random_select(
        self, instances: list[ServiceInstance]
    ) -> ServiceInstance:
        """Weighted random selection"""
        weights = [max(instance.weight, 0.1) for instance in instances]
        total_weight = sum(weights)

        rand = random.uniform(0, total_weight)
        cumulative_weight = 0

        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if rand <= cumulative_weight:
                return instances[i]

        return instances[-1]

    def _consistent_hash_select(
        self,
        service_name: str,
        instances: list[ServiceInstance],
        context: dict[str, Any] | None,
    ) -> ServiceInstance:
        """Consistent hash selection"""
        if not context or "hash_key" not in context:
            return self._round_robin_select(service_name, instances)

        hash_key = str(context["hash_key"])
        hash_value = hash(hash_key) % (2**32)

        # Build hash ring if needed
        ring_key = f"{service_name}:{len(instances)}"
        if ring_key not in self._consistent_hash_rings:
            ring = []
            for i, instance in enumerate(instances):
                for replica in range(100):  # Virtual nodes
                    virtual_key = hash(f"{instance.id}:{replica}") % (2**32)
                    ring.append((virtual_key, i))
            ring.sort()
            self._consistent_hash_rings[ring_key] = ring

        ring = self._consistent_hash_rings[ring_key]

        # Find closest instance
        for ring_hash, instance_idx in ring:
            if ring_hash >= hash_value:
                return instances[instance_idx]

        # Wrap around
        return instances[ring[0][1]] if ring else instances[0]


class AutoScaler:
    """Automatic scaling based on metrics"""

    def __init__(self):
        self.scaling_rules: dict[str, list[ScalingRule]] = defaultdict(list)
        self.metrics_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.last_scale_action: dict[str, datetime] = {}
        self._running = False
        self._monitoring_task: asyncio.Task | None = None

    def add_scaling_rule(self, service_name: str, rule: ScalingRule):
        """Add auto-scaling rule"""
        self.scaling_rules[service_name].append(rule)
        logger.info(
            f"Added scaling rule for {service_name}: {rule.metric_name}"
        )

    def record_metric(self, service_name: str, metric_name: str, value: float):
        """Record metric value"""
        timestamp = datetime.now()
        key = f"{service_name}:{metric_name}"
        self.metrics_history[key].append((timestamp, value))

    async def start_monitoring(self, scale_callback: Callable):
        """Start auto-scaling monitoring"""
        self._running = True
        self._scale_callback = scale_callback

        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Auto-scaler monitoring started")

    async def stop_monitoring(self):
        """Stop auto-scaling monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        logger.info("Auto-scaler monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                for service_name, rules in self.scaling_rules.items():
                    for rule in rules:
                        await self._evaluate_scaling_rule(service_name, rule)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-scaling monitoring: {e}")

    async def _evaluate_scaling_rule(
        self, service_name: str, rule: ScalingRule
    ):
        """Evaluate single scaling rule"""
        key = f"{service_name}:{rule.metric_name}"

        if key not in self.metrics_history:
            return

        # Get recent metrics within evaluation window
        now = datetime.now()
        cutoff = now - timedelta(seconds=rule.evaluation_window)

        recent_metrics = [
            value
            for timestamp, value in self.metrics_history[key]
            if timestamp >= cutoff
        ]

        if not recent_metrics:
            return

        avg_value = sum(recent_metrics) / len(recent_metrics)

        # Check cooldown
        last_action_key = f"{service_name}:{rule.metric_name}"
        last_action = self.last_scale_action.get(last_action_key)

        if last_action:
            time_since_action = (now - last_action).total_seconds()

            if (
                avg_value > rule.threshold_scale_up
                and time_since_action < rule.scale_up_cooldown
            ):
                return

            if (
                avg_value < rule.threshold_scale_down
                and time_since_action < rule.scale_down_cooldown
            ):
                return

        # Determine scaling action
        if avg_value > rule.threshold_scale_up:
            await self._scale_service(service_name, rule, "up", avg_value)
        elif avg_value < rule.threshold_scale_down:
            await self._scale_service(service_name, rule, "down", avg_value)

    async def _scale_service(
        self,
        service_name: str,
        rule: ScalingRule,
        direction: str,
        metric_value: float,
    ):
        """Execute scaling action"""
        try:
            if hasattr(self, "_scale_callback"):
                await self._scale_callback(
                    service_name, rule, direction, metric_value
                )

            # Record action time
            action_key = f"{service_name}:{rule.metric_name}"
            self.last_scale_action[action_key] = datetime.now()

            logger.info(
                f"Scaled {service_name} {direction} due to {rule.metric_name}={metric_value}"
            )

        except Exception as e:
            logger.error(f"Error scaling service {service_name}: {e}")


class ScalabilityManager:
    """Main scalability management coordinator"""

    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler()
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Background tasks
        self._health_check_task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        """Start scalability management"""
        self._running = True

        # Start health monitoring
        self._health_check_task = asyncio.create_task(
            self._health_check_loop()
        )

        # Start auto-scaling
        await self.auto_scaler.start_monitoring(self._handle_scaling_event)

        logger.info("Scalability manager started")

    async def stop(self):
        """Stop scalability management"""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()

        await self.auto_scaler.stop_monitoring()

        logger.info("Scalability manager stopped")

    async def register_service(
        self,
        service_name: str,
        host: str,
        port: int,
        weight: float = 1.0,
        metadata: dict[str, Any] = None,
    ):
        """Register service instance"""
        instance = ServiceInstance(
            id=f"{host}:{port}",
            host=host,
            port=port,
            weight=weight,
            metadata=metadata or {},
        )

        await self.service_registry.register_service(service_name, instance)

        # Initialize circuit breaker
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()

    async def get_service_instance(
        self, service_name: str, context: dict[str, Any] | None = None
    ) -> ServiceInstance | None:
        """Get service instance using load balancing"""
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(service_name)
        if circuit_breaker and not circuit_breaker.should_allow_request():
            logger.warning(f"Circuit breaker open for service: {service_name}")
            return None

        # Get healthy instances
        instances = await self.service_registry.get_service_instances(
            service_name, healthy_only=True
        )

        if not instances:
            logger.warning(f"No healthy instances for service: {service_name}")
            return None

        # Select instance using load balancing
        return await self.load_balancer.select_instance(
            service_name, instances, context
        )

    async def record_request_result(
        self,
        service_name: str,
        instance_id: str,
        success: bool,
        response_time_ms: float,
    ):
        """Record request result for monitoring"""
        # Update instance stats
        instances = await self.service_registry.get_service_instances(
            service_name, healthy_only=False
        )
        for instance in instances:
            if instance.id == instance_id:
                instance.total_requests += 1
                if not success:
                    instance.failed_requests += 1
                instance.response_time_ms = response_time_ms
                break

        # Update circuit breaker
        circuit_breaker = self.circuit_breakers.get(service_name)
        if circuit_breaker:
            if success:
                circuit_breaker.record_success()
            else:
                circuit_breaker.record_failure()

        # Record metrics for auto-scaling
        self.auto_scaler.record_metric(
            service_name, "response_time", response_time_ms
        )
        self.auto_scaler.record_metric(
            service_name, "success_rate", 1.0 if success else 0.0
        )

    def add_scaling_rule(
        self,
        service_name: str,
        metric_name: str,
        scale_up_threshold: float,
        scale_down_threshold: float,
        min_instances: int = 1,
        max_instances: int = 10,
    ):
        """Add auto-scaling rule"""
        rule = ScalingRule(
            metric_name=metric_name,
            threshold_scale_up=scale_up_threshold,
            threshold_scale_down=scale_down_threshold,
            min_instances=min_instances,
            max_instances=max_instances,
        )

        self.auto_scaler.add_scaling_rule(service_name, rule)

    async def _handle_scaling_event(
        self,
        service_name: str,
        rule: ScalingRule,
        direction: str,
        metric_value: float,
    ):
        """Handle auto-scaling event"""
        instances = await self.service_registry.get_service_instances(
            service_name, healthy_only=False
        )
        current_count = len(instances)

        if direction == "up" and current_count < rule.max_instances:
            logger.info(
                f"Would scale up {service_name} (metric: {rule.metric_name}={metric_value})"
            )
            # In real implementation, would trigger instance creation

        elif direction == "down" and current_count > rule.min_instances:
            logger.info(
                f"Would scale down {service_name} (metric: {rule.metric_name}={metric_value})"
            )
            # In real implementation, would trigger instance removal

    async def _health_check_loop(self):
        """Background health checking"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                for (
                    service_name,
                    instances,
                ) in self.service_registry.services.items():
                    for instance in instances.copy():
                        await self._check_instance_health(
                            service_name, instance
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def _check_instance_health(
        self, service_name: str, instance: ServiceInstance
    ):
        """Check individual instance health"""
        try:
            # Simulate health check (in real implementation, would make HTTP request)
            start_time = time.time()

            # Mock health check logic
            is_healthy = (
                instance.success_rate > 0.8
                and instance.response_time_ms < 1000
            )

            (time.time() - start_time) * 1000

            status = (
                ServiceStatus.HEALTHY
                if is_healthy
                else ServiceStatus.UNHEALTHY
            )

            await self.service_registry.update_instance_status(
                service_name, instance.id, status
            )

        except Exception as e:
            logger.error(
                f"Health check failed for {service_name}/{instance.id}: {e}"
            )
            await self.service_registry.update_instance_status(
                service_name, instance.id, ServiceStatus.UNHEALTHY
            )

    def get_service_stats(self) -> dict[str, Any]:
        """Get comprehensive service statistics"""
        stats = {
            "services": {},
            "circuit_breakers": {},
            "load_balancer": {"algorithm": self.load_balancer.algorithm.value},
        }

        # Service stats
        for service_name, instances in self.service_registry.services.items():
            service_stats = {
                "instance_count": len(instances),
                "healthy_instances": len(
                    [i for i in instances if i.status == ServiceStatus.HEALTHY]
                ),
                "total_requests": sum(i.total_requests for i in instances),
                "failed_requests": sum(i.failed_requests for i in instances),
                "avg_response_time": (
                    sum(i.response_time_ms for i in instances) / len(instances)
                    if instances
                    else 0
                ),
                "instances": [
                    {
                        "id": i.id,
                        "endpoint": i.endpoint,
                        "status": i.status.value,
                        "current_connections": i.current_connections,
                        "success_rate": i.success_rate,
                        "response_time_ms": i.response_time_ms,
                    }
                    for i in instances
                ],
            }
            stats["services"][service_name] = service_stats

        # Circuit breaker stats
        for service_name, cb in self.circuit_breakers.items():
            stats["circuit_breakers"][service_name] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "last_failure": (
                    cb.last_failure_time.isoformat()
                    if cb.last_failure_time
                    else None
                ),
            }

        return stats
