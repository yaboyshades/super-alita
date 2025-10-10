#!/usr/bin/env python3
"""
Resilience Manager - Circuit Breakers and Bulkhead Isolation for Super Alita

This module implements resilience patterns to handle failures gracefully:
1. Circuit Breaker: Prevents cascading failures by short-circuiting failing services
2. Bulkhead Isolation: Partitions resources to contain failures
3. Request Hedging: Reduces tail latency with speculative retries
4. Graceful Degradation: Fallback strategies when services are unavailable

Design Philosophy:
- Fail fast when systems are unhealthy
- Isolate failures to prevent system-wide degradation
- Provide meaningful fallbacks and user feedback
- Monitor and adapt to changing system conditions
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

import pybreaker

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceCategory(Enum):
    """Categories of services for bulkhead isolation"""

    LLM_API = "llm_api"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    NETWORK_IO = "network_io"
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ServiceMetrics:
    """Metrics for a service"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    min_latency: float = float("inf")
    max_latency: float = 0.0
    last_success: float | None = None
    last_failure: float | None = None
    circuit_breaker_trips: int = 0


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation"""

    max_concurrent: int = 10
    max_queue_size: int = 100
    timeout_seconds: float = 30.0
    category: ServiceCategory = ServiceCategory.EXTERNAL_API


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: Exception | tuple = Exception
    name: str | None = None


class FallbackStrategy(ABC):
    """Abstract fallback strategy"""

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> Any:
        """Execute fallback logic"""
        pass


class CachedResponseFallback(FallbackStrategy):
    """Return cached response if available"""

    def __init__(
        self,
        cache: dict[str, Any],
        cache_key_fn: Callable[[dict[str, Any]], str],
    ):
        self.cache = cache
        self.cache_key_fn = cache_key_fn

    async def execute(self, context: dict[str, Any]) -> Any:
        cache_key = self.cache_key_fn(context)
        cached_value = self.cache.get(cache_key)
        if cached_value:
            logger.info(f"Returning cached response for {cache_key}")
            return cached_value

        return {"error": "Service unavailable and no cached response"}


class DefaultResponseFallback(FallbackStrategy):
    """Return a default response"""

    def __init__(self, default_response: Any):
        self.default_response = default_response

    async def execute(self, context: dict[str, Any]) -> Any:
        logger.info("Returning default fallback response")
        return self.default_response


class CircuitBreakerManager:
    """Manages circuit breakers for different services"""

    def __init__(self):
        self.circuit_breakers: dict[str, pybreaker.CircuitBreaker] = {}
        self.service_metrics: dict[str, ServiceMetrics] = {}
        self.fallback_strategies: dict[str, FallbackStrategy] = {}

    def create_circuit_breaker(
        self,
        service_name: str,
        config: CircuitBreakerConfig,
        fallback_strategy: FallbackStrategy | None = None,
    ) -> pybreaker.CircuitBreaker:
        """Create and register a circuit breaker for a service"""

        def fallback_fn(*args, **kwargs):
            """Fallback function called when circuit is open"""
            logger.warning(
                f"Circuit breaker {service_name} is open, executing fallback"
            )

            if fallback_strategy:
                # For sync compatibility, we'll return a simple fallback
                if hasattr(fallback_strategy, "default_response"):
                    return fallback_strategy.default_response
                else:
                    return {
                        "error": f"Service {service_name} is temporarily unavailable"
                    }

            return {
                "error": f"Service {service_name} is temporarily unavailable"
            }

        def notify_fn(breaker, transition):
            """Notification function for state changes"""
            logger.info(
                f"Circuit breaker {service_name} transitioned to {transition}"
            )
            if service_name in self.service_metrics:
                if transition == "open":
                    self.service_metrics[
                        service_name
                    ].circuit_breaker_trips += 1

        circuit_breaker = pybreaker.CircuitBreaker(
            fail_max=config.failure_threshold,
            reset_timeout=config.recovery_timeout,
            exclude=(
                config.expected_exception
                if config.expected_exception != Exception
                else None
            ),
            name=config.name or service_name,
        )

        # Add state change listener if available
        if hasattr(circuit_breaker, "add_state_change_listener"):
            circuit_breaker.add_state_change_listener(notify_fn)

        self.circuit_breakers[service_name] = circuit_breaker
        self.service_metrics[service_name] = ServiceMetrics()

        if fallback_strategy:
            self.fallback_strategies[service_name] = fallback_strategy

        logger.info(f"Created circuit breaker for {service_name}")
        return circuit_breaker

    def get_circuit_breaker(
        self, service_name: str
    ) -> pybreaker.CircuitBreaker | None:
        """Get circuit breaker for a service"""
        return self.circuit_breakers.get(service_name)

    async def call_with_circuit_breaker(
        self, service_name: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection"""
        circuit_breaker = self.get_circuit_breaker(service_name)
        if not circuit_breaker:
            # No circuit breaker configured, call directly
            return await self._execute_and_measure(
                service_name, func, *args, **kwargs
            )

        start_time = time.time()
        try:
            # PyBreaker doesn't support async directly, so we wrap it
            if asyncio.iscoroutinefunction(func):

                async def async_wrapper():
                    return await func(*args, **kwargs)

                # Execute with circuit breaker protection
                try:
                    result = circuit_breaker(
                        lambda: asyncio.run(async_wrapper())
                    )()
                except Exception as e:
                    if "CircuitBreakerError" in str(type(e)):
                        # Circuit breaker is open, try fallback
                        fallback = self.fallback_strategies.get(service_name)
                        if fallback:
                            context = {
                                "service": service_name,
                                "args": args,
                                "kwargs": kwargs,
                                "error": str(e),
                            }
                            return await fallback.execute(context)
                        else:
                            return {
                                "error": f"Service {service_name} is temporarily unavailable"
                            }
                    else:
                        raise
            else:
                result = circuit_breaker(func)(*args, **kwargs)

            # Record success metrics
            latency = time.time() - start_time
            self._record_success(service_name, latency)

            return result

        except Exception as e:
            # Record failure metrics
            latency = time.time() - start_time
            self._record_failure(service_name, latency)

            # Try fallback if available
            fallback = self.fallback_strategies.get(service_name)
            if fallback and "temporarily unavailable" not in str(e):
                context = {
                    "service": service_name,
                    "args": args,
                    "kwargs": kwargs,
                    "error": str(e),
                }
                return await fallback.execute(context)

            raise

    async def _execute_and_measure(
        self, service_name: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Execute function and measure performance"""
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            latency = time.time() - start_time
            self._record_success(service_name, latency)
            return result

        except Exception:
            latency = time.time() - start_time
            self._record_failure(service_name, latency)
            raise

    def _record_success(self, service_name: str, latency: float):
        """Record successful request metrics"""
        if service_name not in self.service_metrics:
            self.service_metrics[service_name] = ServiceMetrics()

        metrics = self.service_metrics[service_name]
        metrics.total_requests += 1
        metrics.successful_requests += 1
        metrics.total_latency += latency
        metrics.min_latency = min(metrics.min_latency, latency)
        metrics.max_latency = max(metrics.max_latency, latency)
        metrics.last_success = time.time()

    def _record_failure(self, service_name: str, latency: float):
        """Record failed request metrics"""
        if service_name not in self.service_metrics:
            self.service_metrics[service_name] = ServiceMetrics()

        metrics = self.service_metrics[service_name]
        metrics.total_requests += 1
        metrics.failed_requests += 1
        metrics.total_latency += latency
        metrics.last_failure = time.time()

    def get_service_metrics(self, service_name: str) -> ServiceMetrics | None:
        """Get metrics for a service"""
        return self.service_metrics.get(service_name)

    def get_all_metrics(self) -> dict[str, ServiceMetrics]:
        """Get metrics for all services"""
        return self.service_metrics.copy()

    def get_circuit_breaker_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers"""
        status = {}
        for service_name, breaker in self.circuit_breakers.items():
            metrics = self.service_metrics.get(service_name, ServiceMetrics())

            # Get circuit breaker state safely
            current_state = getattr(breaker, "current_state", "unknown")
            fail_counter = getattr(breaker, "fail_counter", 0)

            status[service_name] = {
                "state": current_state,
                "failure_count": fail_counter,
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "success_rate": (
                    metrics.successful_requests
                    / max(1, metrics.total_requests)
                ),
                "avg_latency": (
                    (metrics.total_latency / max(1, metrics.total_requests))
                    if metrics.total_requests > 0
                    else 0.0
                ),
                "circuit_breaker_trips": metrics.circuit_breaker_trips,
                "last_success": metrics.last_success,
                "last_failure": metrics.last_failure,
            }

        return status


class BulkheadManager:
    """Manages resource isolation using bulkhead pattern"""

    def __init__(self):
        self.executors: dict[ServiceCategory, ThreadPoolExecutor] = {}
        self.semaphores: dict[ServiceCategory, asyncio.Semaphore] = {}
        self.configs: dict[ServiceCategory, BulkheadConfig] = {}
        self.metrics: dict[ServiceCategory, dict[str, Any]] = {}

    def create_bulkhead(
        self, category: ServiceCategory, config: BulkheadConfig
    ):
        """Create bulkhead for a service category"""
        # Thread pool for CPU-bound tasks
        self.executors[category] = ThreadPoolExecutor(
            max_workers=config.max_concurrent,
            thread_name_prefix=f"bulkhead-{category.value}",
        )

        # Semaphore for async concurrency control
        self.semaphores[category] = asyncio.Semaphore(config.max_concurrent)

        self.configs[category] = config
        self.metrics[category] = {
            "total_requests": 0,
            "active_requests": 0,
            "queued_requests": 0,
            "rejected_requests": 0,
            "timeout_requests": 0,
        }

        logger.info(
            f"Created bulkhead for {category.value} with {config.max_concurrent} workers"
        )

    async def execute_in_bulkhead(
        self, category: ServiceCategory, func: Callable, *args, **kwargs
    ) -> Any:
        """Execute function within bulkhead limits"""
        if category not in self.semaphores:
            # No bulkhead configured, execute directly
            logger.warning(
                f"No bulkhead configured for {category.value}, executing directly"
            )
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        config = self.configs[category]
        semaphore = self.semaphores[category]
        metrics = self.metrics[category]

        # Check if we can acquire the semaphore
        try:
            # Try to acquire with timeout
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=0.1,  # Very short timeout for queue check
            )
        except TimeoutError:
            # Check queue size
            if (
                semaphore._waiters
                and len(semaphore._waiters) >= config.max_queue_size
            ):
                metrics["rejected_requests"] += 1
                raise Exception(
                    f"Bulkhead {category.value} queue is full, rejecting request"
                )

        metrics["total_requests"] += 1
        metrics["active_requests"] += 1

        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=config.timeout_seconds
                )
            else:
                # Use thread pool for sync functions
                executor = self.executors[category]
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, func, *args, **kwargs
                )

            return result

        except TimeoutError:
            metrics["timeout_requests"] += 1
            raise Exception(f"Request timed out in bulkhead {category.value}")
        finally:
            metrics["active_requests"] -= 1
            semaphore.release()

    def get_bulkhead_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all bulkheads"""
        status = {}
        for category, config in self.configs.items():
            semaphore = self.semaphores[category]
            executor = self.executors[category]
            metrics = self.metrics[category]

            status[category.value] = {
                "max_concurrent": config.max_concurrent,
                "max_queue_size": config.max_queue_size,
                "timeout_seconds": config.timeout_seconds,
                "available_permits": semaphore._value,
                "waiting_count": (
                    len(semaphore._waiters) if semaphore._waiters else 0
                ),
                "thread_pool_size": executor._max_workers,
                "active_threads": executor._threads,
                **metrics,
            }

        return status

    def shutdown(self):
        """Shutdown all thread pools"""
        for executor in self.executors.values():
            executor.shutdown(wait=True)
        logger.info("All bulkhead thread pools shutdown")


class RequestHedger:
    """Implements request hedging (speculative retries) to reduce tail latency"""

    def __init__(self, hedge_after_ms: float = 500):
        self.hedge_after_ms = hedge_after_ms
        self.metrics = {
            "total_hedged_requests": 0,
            "hedge_wins": 0,  # Hedge response came back first
            "original_wins": 0,  # Original response came back first
            "both_failed": 0,
        }

    async def execute_with_hedging(
        self,
        primary_func: Callable,
        hedge_func: Callable | None = None,
        *args,
        **kwargs,
    ) -> Any:
        """Execute request with hedging - fire duplicate request if first is slow"""
        self.metrics["total_hedged_requests"] += 1

        # Use same function for hedge if not specified
        if hedge_func is None:
            hedge_func = primary_func

        # Start primary request
        primary_task = asyncio.create_task(
            self._execute_safely(primary_func, *args, **kwargs)
        )

        try:
            # Wait for hedge timeout
            result = await asyncio.wait_for(
                primary_task, timeout=self.hedge_after_ms / 1000.0
            )

            # Primary completed within hedge timeout
            self.metrics["original_wins"] += 1
            return result

        except TimeoutError:
            # Primary is slow, start hedge request
            logger.debug(
                f"Primary request slow, starting hedge after {self.hedge_after_ms}ms"
            )
            hedge_task = asyncio.create_task(
                self._execute_safely(hedge_func, *args, **kwargs)
            )

            # Race both requests
            done, pending = await asyncio.wait(
                [primary_task, hedge_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Get result from first completed task
            if done:
                completed_task = next(iter(done))
                try:
                    result = await completed_task

                    # Track which one won
                    if completed_task == primary_task:
                        self.metrics["original_wins"] += 1
                    else:
                        self.metrics["hedge_wins"] += 1

                    return result

                except Exception as e:
                    # First completed task failed, try the other
                    if pending:
                        try:
                            other_task = next(iter(pending))
                            other_task.cancel()
                            return await other_task
                        except:
                            pass

                    self.metrics["both_failed"] += 1
                    raise e

            # Both failed
            self.metrics["both_failed"] += 1
            raise Exception("Both primary and hedge requests failed")

    async def _execute_safely(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function safely, handling both sync and async"""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            logger.debug(f"Request failed: {e}")
            raise

    def get_hedging_metrics(self) -> dict[str, Any]:
        """Get hedging performance metrics"""
        total = self.metrics["total_hedged_requests"]
        if total == 0:
            return self.metrics

        return {
            **self.metrics,
            "hedge_win_rate": self.metrics["hedge_wins"] / total,
            "original_win_rate": self.metrics["original_wins"] / total,
            "failure_rate": self.metrics["both_failed"] / total,
        }


class ResilienceManager:
    """Main resilience management system combining all patterns"""

    def __init__(self):
        self.circuit_breaker_manager = CircuitBreakerManager()
        self.bulkhead_manager = BulkheadManager()
        self.request_hedger = RequestHedger()

        # Setup default configurations
        self._setup_default_configurations()

    def _setup_default_configurations(self):
        """Setup default circuit breakers and bulkheads for common services"""

        # Circuit breakers for common services
        self.circuit_breaker_manager.create_circuit_breaker(
            "llm_api",
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                name="LLM API Circuit Breaker",
            ),
            DefaultResponseFallback(
                {
                    "error": "LLM service temporarily unavailable",
                    "fallback": True,
                    "suggestion": "Please try again in a few moments",
                }
            ),
        )

        self.circuit_breaker_manager.create_circuit_breaker(
            "external_api",
            CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60,
                name="External API Circuit Breaker",
            ),
        )

        # Bulkheads for resource categories
        self.bulkhead_manager.create_bulkhead(
            ServiceCategory.LLM_API,
            BulkheadConfig(max_concurrent=5, timeout_seconds=60.0),
        )

        self.bulkhead_manager.create_bulkhead(
            ServiceCategory.EXTERNAL_API,
            BulkheadConfig(max_concurrent=10, timeout_seconds=30.0),
        )

        self.bulkhead_manager.create_bulkhead(
            ServiceCategory.DATABASE,
            BulkheadConfig(max_concurrent=15, timeout_seconds=10.0),
        )

        self.bulkhead_manager.create_bulkhead(
            ServiceCategory.FILE_SYSTEM,
            BulkheadConfig(max_concurrent=8, timeout_seconds=20.0),
        )

        logger.info("Default resilience configurations loaded")

    async def execute_resilient_call(
        self,
        service_name: str,
        func: Callable,
        category: ServiceCategory | None = None,
        use_hedging: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        """Execute a call with full resilience protection"""

        # Determine category if not provided
        if category is None:
            category = self._infer_category(service_name)

        # Execute with bulkhead isolation
        async def bulkhead_protected_call():
            return await self.bulkhead_manager.execute_in_bulkhead(
                category, func, *args, **kwargs
            )

        # Execute with circuit breaker protection
        async def circuit_breaker_protected_call():
            return (
                await self.circuit_breaker_manager.call_with_circuit_breaker(
                    service_name, bulkhead_protected_call
                )
            )

        # Optionally use hedging
        if use_hedging:
            return await self.request_hedger.execute_with_hedging(
                circuit_breaker_protected_call
            )
        else:
            return await circuit_breaker_protected_call()

    def _infer_category(self, service_name: str) -> ServiceCategory:
        """Infer service category from service name"""
        name_lower = service_name.lower()

        if any(
            keyword in name_lower
            for keyword in ["llm", "openai", "anthropic", "gpt"]
        ):
            return ServiceCategory.LLM_API
        elif any(
            keyword in name_lower
            for keyword in ["github", "api", "http", "rest"]
        ):
            return ServiceCategory.EXTERNAL_API
        elif any(
            keyword in name_lower
            for keyword in ["db", "database", "sql", "redis"]
        ):
            return ServiceCategory.DATABASE
        elif any(
            keyword in name_lower for keyword in ["file", "fs", "disk", "io"]
        ):
            return ServiceCategory.FILE_SYSTEM
        else:
            return ServiceCategory.EXTERNAL_API  # Default

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health and resilience status"""
        circuit_breaker_status = (
            self.circuit_breaker_manager.get_circuit_breaker_status()
        )
        bulkhead_status = self.bulkhead_manager.get_bulkhead_status()
        hedging_metrics = self.request_hedger.get_hedging_metrics()

        # Calculate overall health score
        total_services = len(circuit_breaker_status)
        healthy_services = sum(
            1
            for status in circuit_breaker_status.values()
            if status["state"] != "open"
        )

        health_score = healthy_services / max(1, total_services)

        return {
            "health_score": health_score,
            "healthy_services": healthy_services,
            "total_services": total_services,
            "circuit_breakers": circuit_breaker_status,
            "bulkheads": bulkhead_status,
            "hedging": hedging_metrics,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def shutdown(self):
        """Shutdown resilience manager"""
        self.bulkhead_manager.shutdown()
        logger.info("Resilience manager shutdown complete")


# Global instance for easy access
_global_resilience_manager: ResilienceManager | None = None


def get_global_resilience_manager() -> ResilienceManager:
    """Get or create global resilience manager instance"""
    global _global_resilience_manager
    if _global_resilience_manager is None:
        _global_resilience_manager = ResilienceManager()
    return _global_resilience_manager


# Convenience decorators for easy integration
def circuit_breaker(service_name: str, **config_kwargs):
    """Decorator to add circuit breaker protection to a function"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = get_global_resilience_manager()
            return await manager.circuit_breaker_manager.call_with_circuit_breaker(
                service_name, func, *args, **kwargs
            )

        return wrapper

    return decorator


def bulkhead(category: ServiceCategory):
    """Decorator to add bulkhead isolation to a function"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = get_global_resilience_manager()
            return await manager.bulkhead_manager.execute_in_bulkhead(
                category, func, *args, **kwargs
            )

        return wrapper

    return decorator


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import random

    async def test_resilience_manager():
        """Test resilience manager functionality"""
        rm = ResilienceManager()

        # Simulate unreliable service
        async def unreliable_service(fail_rate: float = 0.3):
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate work
            if random.random() < fail_rate:
                raise Exception("Service temporarily failed")
            return {"status": "success", "data": "test_data"}

        print("Resilience Manager Test Results:")
        print("=" * 50)

        # Test multiple calls
        success_count = 0
        total_calls = 20

        for i in range(total_calls):
            try:
                await rm.execute_resilient_call(
                    "test_service",
                    unreliable_service,
                    ServiceCategory.EXTERNAL_API,
                    use_hedging=True,
                    fail_rate=0.4,  # 40% failure rate
                )
                success_count += 1
                print(f"Call {i+1}: ✅ Success")
            except Exception as e:
                print(f"Call {i+1}: ❌ Failed - {e}")

        print(
            f"\nSuccess rate: {success_count}/{total_calls} ({success_count/total_calls*100:.1f}%)"
        )

        # Show system health
        health = rm.get_system_health()
        print(f"\nSystem Health Score: {health['health_score']:.2f}")
        print(
            f"Healthy Services: {health['healthy_services']}/{health['total_services']}"
        )

        # Show detailed metrics
        print("\nCircuit Breaker Status:")
        for service, status in health["circuit_breakers"].items():
            state_emoji = (
                "🟢"
                if status["state"] == "closed"
                else "🔴" if status["state"] == "open" else "🟡"
            )
            print(
                f"  {state_emoji} {service}: {status['state']} (success: {status['success_rate']:.2f})"
            )

        rm.shutdown()

    asyncio.run(test_resilience_manager())
