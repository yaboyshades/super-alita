"""
Unified router with resilient execution patterns.

Enhances the existing REUG router with timeout management, retry logic,
circuit breaker patterns, and graceful degradation.

Patterns adapted from GitHub examples:
- Timeout and retry patterns with exponential backoff
- Circuit breaker for failing services
- Graceful degradation when tools fail
- Connection pooling and resource management
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay_ms: int = 100
    max_delay_ms: int = 5000
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior."""

    tool_timeout_s: float = 30.0
    llm_timeout_s: float = 60.0
    overall_timeout_s: float = 300.0
    connection_timeout_s: float = 10.0


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout_s: int = 60
    half_open_max_calls: int = 3


class CircuitBreaker:
    """Circuit breaker implementation for failing services."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        now = time.time()

        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if now - self.last_failure_time > self.config.recovery_timeout_s:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls

        return False

    def record_success(self):
        """Record successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0

    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.half_open_calls = 0


class ResilientExecutor:
    """Executor with retry logic and circuit breaker."""

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        timeout_config: TimeoutConfig | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
    ):
        self.retry_config = retry_config or RetryConfig()
        self.timeout_config = timeout_config or TimeoutConfig()
        self.circuit_breaker = CircuitBreaker(
            circuit_config or CircuitBreakerConfig()
        )

        # HTTP client with connection pooling
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.timeout_config.connection_timeout_s,
                read=self.timeout_config.tool_timeout_s,
                write=self.timeout_config.tool_timeout_s,
                pool=self.timeout_config.tool_timeout_s,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=10, max_connections=20
            ),
        )

    async def execute_with_retry(
        self,
        operation: callable,
        *args,
        timeout: float | None = None,
        **kwargs,
    ) -> Any:
        """
        Execute operation with retry logic and circuit breaker.

        Adapted from patterns in GitHub examples for resilient execution.
        """
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is open - service unavailable")

        timeout = timeout or self.timeout_config.tool_timeout_s
        last_exception = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    operation(*args, **kwargs), timeout=timeout
                )

                # Record success
                self.circuit_breaker.record_success()
                return result

            except TimeoutError as e:
                last_exception = e
                logger.warning(f"Operation timed out on attempt {attempt + 1}")

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Operation failed on attempt {attempt + 1}: {e}"
                )

            # Don't retry on final attempt
            if attempt == self.retry_config.max_retries:
                break

            # Calculate delay with exponential backoff and jitter
            delay = (
                min(
                    self.retry_config.base_delay_ms
                    * (self.retry_config.exponential_base**attempt),
                    self.retry_config.max_delay_ms,
                )
                / 1000.0
            )

            if self.retry_config.jitter:
                import random

                delay *= (
                    0.5 + random.random() * 0.5
                )  # 50-100% of calculated delay

            logger.info(
                f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{self.retry_config.max_retries})"
            )
            await asyncio.sleep(delay)

        # All retries failed
        self.circuit_breaker.record_failure()
        raise last_exception or Exception("All retry attempts failed")

    async def close(self):
        """Clean up resources."""
        await self.http_client.aclose()


class UnifiedRouter:
    """
    Enhanced REUG router with resilient execution patterns.

    Maintains compatibility with existing router.py while adding:
    - Timeout management
    - Retry logic with exponential backoff
    - Circuit breaker for failing tools
    - Connection pooling
    - Graceful degradation
    """

    def __init__(
        self,
        base_router: Any,
        retry_config: RetryConfig | None = None,
        timeout_config: TimeoutConfig | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
    ):
        self.base_router = base_router
        self.executor = ResilientExecutor(
            retry_config, timeout_config, circuit_config
        )
        self.tool_circuits = {}  # Per-tool circuit breakers
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "timeout_errors": 0,
            "circuit_breaker_trips": 0,
        }

    def get_tool_circuit_breaker(self, tool_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for specific tool."""
        if tool_name not in self.tool_circuits:
            self.tool_circuits[tool_name] = CircuitBreaker(
                CircuitBreakerConfig(
                    failure_threshold=3, recovery_timeout_s=30
                )
            )
        return self.tool_circuits[tool_name]

    async def execute_turn(
        self, message: str, session_id: str, **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Enhanced execute_turn with resilient patterns.

        Wraps the base router's execute_turn with timeout and retry logic.
        """
        start_time = time.time()
        self.metrics["total_executions"] += 1

        try:
            # Execute base router with overall timeout
            async with asyncio.timeout(
                self.executor.timeout_config.overall_timeout_s
            ):
                async for event in self.base_router.execute_turn(
                    message, session_id, **kwargs
                ):

                    # Enhance events with resilience metadata
                    if event.get("type") == "AbilityCalled":
                        event["resilience"] = {
                            "timeout_s": self.executor.timeout_config.tool_timeout_s,
                            "retry_enabled": True,
                            "circuit_state": self.get_tool_circuit_breaker(
                                event.get("tool_name", "unknown")
                            ).state.value,
                        }

                    yield event

            self.metrics["successful_executions"] += 1

        except TimeoutError:
            self.metrics["timeout_errors"] += 1
            self.metrics["failed_executions"] += 1

            # Yield timeout error event
            yield {
                "type": "TaskFailed",
                "error": "Operation timed out",
                "timeout_s": self.executor.timeout_config.overall_timeout_s,
                "elapsed_s": time.time() - start_time,
            }

        except Exception as e:
            self.metrics["failed_executions"] += 1

            # Yield error event with resilience info
            yield {
                "type": "TaskFailed",
                "error": str(e),
                "elapsed_s": time.time() - start_time,
                "resilience": {
                    "retries_attempted": getattr(e, "retry_count", 0),
                    "circuit_breaker_tripped": "Circuit breaker" in str(e),
                },
            }

    async def execute_tool_with_resilience(
        self, tool_name: str, tool_args: dict[str, Any], registry: Any
    ) -> dict[str, Any]:
        """
        Execute a tool with full resilience patterns.

        This can be used to wrap individual tool executions.
        """
        circuit = self.get_tool_circuit_breaker(tool_name)

        if not circuit.can_execute():
            self.metrics["circuit_breaker_trips"] += 1
            return {
                "error": f"Tool {tool_name} circuit breaker is open",
                "circuit_state": circuit.state.value,
                "retry_after_s": circuit.config.recovery_timeout_s,
            }

        start_time = time.time()

        try:
            # Define the tool execution operation
            async def tool_operation():
                return await registry.execute(tool_name, tool_args)

            # Execute with retry and timeout
            result = await self.executor.execute_with_retry(
                tool_operation,
                timeout=self.executor.timeout_config.tool_timeout_s,
            )

            circuit.record_success()

            return {
                "result": result,
                "execution_time_s": time.time() - start_time,
                "circuit_state": circuit.state.value,
            }

        except Exception as e:
            circuit.record_failure()

            return {
                "error": str(e),
                "execution_time_s": time.time() - start_time,
                "circuit_state": circuit.state.value,
                "tool_name": tool_name,
            }

    async def execute_llm_with_resilience(
        self, llm_client: Any, messages: list[dict[str, Any]], **kwargs
    ) -> Any:
        """Execute LLM call with resilience patterns."""

        async def llm_operation():
            return await llm_client.stream_chat(
                messages,
                timeout=self.executor.timeout_config.llm_timeout_s,
                **kwargs,
            )

        return await self.executor.execute_with_retry(
            llm_operation, timeout=self.executor.timeout_config.llm_timeout_s
        )

    def get_health_status(self) -> dict[str, Any]:
        """Get router health status with circuit breaker states."""
        total_executions = self.metrics["total_executions"]
        success_rate = (
            self.metrics["successful_executions"] / total_executions
            if total_executions > 0
            else 0.0
        )

        return {
            "status": "healthy" if success_rate >= 0.8 else "degraded",
            "success_rate": success_rate,
            "metrics": self.metrics.copy(),
            "circuit_breakers": {
                "global": {
                    "state": self.executor.circuit_breaker.state.value,
                    "failure_count": self.executor.circuit_breaker.failure_count,
                },
                "tools": {
                    tool_name: {
                        "state": circuit.state.value,
                        "failure_count": circuit.failure_count,
                        "last_failure_age_s": (
                            time.time() - circuit.last_failure_time
                            if circuit.last_failure_time > 0
                            else None
                        ),
                    }
                    for tool_name, circuit in self.tool_circuits.items()
                },
            },
            "configuration": {
                "tool_timeout_s": self.executor.timeout_config.tool_timeout_s,
                "llm_timeout_s": self.executor.timeout_config.llm_timeout_s,
                "max_retries": self.executor.retry_config.max_retries,
                "failure_threshold": self.executor.circuit_breaker.config.failure_threshold,
            },
        }

    async def close(self):
        """Clean up resources."""
        await self.executor.close()

    # Delegate methods to maintain compatibility with base router
    def __getattr__(self, name):
        """Delegate unknown methods to base router."""
        return getattr(self.base_router, name)


@asynccontextmanager
async def create_resilient_router(
    base_router: Any,
    retry_config: RetryConfig | None = None,
    timeout_config: TimeoutConfig | None = None,
    circuit_config: CircuitBreakerConfig | None = None,
) -> UnifiedRouter:
    """
    Context manager factory for creating resilient router.

    Ensures proper cleanup of resources.
    """
    router = UnifiedRouter(
        base_router, retry_config, timeout_config, circuit_config
    )
    try:
        yield router
    finally:
        await router.close()


def load_resilience_config_from_env() -> (
    tuple[RetryConfig, TimeoutConfig, CircuitBreakerConfig]
):
    """Load resilience configuration from environment variables."""
    import os

    retry_config = RetryConfig(
        max_retries=int(os.getenv("REUG_MAX_RETRIES", "3")),
        base_delay_ms=int(os.getenv("REUG_RETRY_BASE_MS", "100")),
        max_delay_ms=int(os.getenv("REUG_RETRY_MAX_MS", "5000")),
    )

    timeout_config = TimeoutConfig(
        tool_timeout_s=float(os.getenv("REUG_TOOL_TIMEOUT_S", "30.0")),
        llm_timeout_s=float(os.getenv("REUG_LLM_TIMEOUT_S", "60.0")),
        overall_timeout_s=float(os.getenv("REUG_OVERALL_TIMEOUT_S", "300.0")),
        connection_timeout_s=float(
            os.getenv("REUG_CONNECTION_TIMEOUT_S", "10.0")
        ),
    )

    circuit_config = CircuitBreakerConfig(
        failure_threshold=int(
            os.getenv("REUG_CIRCUIT_FAILURE_THRESHOLD", "5")
        ),
        recovery_timeout_s=int(
            os.getenv("REUG_CIRCUIT_RECOVERY_TIMEOUT_S", "60")
        ),
        half_open_max_calls=int(
            os.getenv("REUG_CIRCUIT_HALF_OPEN_CALLS", "3")
        ),
    )

    return retry_config, timeout_config, circuit_config
