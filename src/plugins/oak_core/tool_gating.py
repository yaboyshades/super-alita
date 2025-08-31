"""
Unified Tool Gating System - Circuit breakers and adaptive EV calculation
Provides comprehensive safety mechanisms for tool execution
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from src.core.optimization.nonstationary import NonStationaryBandit


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ToolExecutionResult:
    """Result of tool execution with performance metrics"""

    success: bool
    execution_time: float
    error_message: str = ""
    output: Any = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "output": self.output,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    max_execution_time: float = 30.0
    storm_detection_window: float = 10.0
    storm_threshold: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "success_threshold": self.success_threshold,
            "max_execution_time": self.max_execution_time,
            "storm_detection_window": self.storm_detection_window,
            "storm_threshold": self.storm_threshold,
        }


class AdvancedCircuitBreaker:
    """
    Advanced circuit breaker with adaptive thresholds and storm detection
    """

    def __init__(self, tool_name: str, config: CircuitBreakerConfig = None):
        self.tool_name = tool_name
        self.config = config or CircuitBreakerConfig()

        # Circuit state management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.state_changed_time = datetime.now(UTC)

        # Performance tracking
        self.execution_history: list[ToolExecutionResult] = []
        self.max_history_size = 1000

        # Storm detection
        self.recent_requests: list[datetime] = []
        self.storm_detected = False

        # Adaptive thresholds
        self.adaptive_failure_threshold = self.config.failure_threshold
        self.adaptive_timeout = self.config.recovery_timeout

        # Logging
        self.logger = logging.getLogger(f"circuit_breaker.{tool_name}")

    async def call(
        self, func: Callable, *args, **kwargs
    ) -> ToolExecutionResult:
        """Execute function through circuit breaker protection"""

        # Check for request storms
        if self._detect_storm():
            return ToolExecutionResult(
                success=False,
                execution_time=0.0,
                error_message="Request storm detected - throttling requests",
                metadata={"storm_detected": True},
            )

        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.state_changed_time = datetime.now(UTC)
                self.logger.info(
                    f"Circuit breaker for {self.tool_name} moving to HALF_OPEN"
                )
            else:
                return ToolExecutionResult(
                    success=False,
                    execution_time=0.0,
                    error_message="Circuit breaker is OPEN - tool unavailable",
                    metadata={"circuit_state": self.state.value},
                )

        # Execute the function
        start_time = time.time()
        result = None

        try:
            # Apply timeout
            result = await asyncio.wait_for(
                self._execute_with_monitoring(func, *args, **kwargs),
                timeout=self.config.max_execution_time,
            )

            execution_time = time.time() - start_time

            # Create success result
            execution_result = ToolExecutionResult(
                success=True,
                execution_time=execution_time,
                output=result,
                confidence=self._calculate_confidence(execution_time),
                metadata={
                    "circuit_state": self.state.value,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                },
            )

            # Update circuit state for success
            self._record_success(execution_result)

            return execution_result

        except TimeoutError:
            execution_time = time.time() - start_time
            execution_result = ToolExecutionResult(
                success=False,
                execution_time=execution_time,
                error_message=f"Tool execution timed out after {self.config.max_execution_time}s",
                metadata={"timeout": True, "circuit_state": self.state.value},
            )

            self._record_failure(execution_result)
            return execution_result

        except Exception as e:
            execution_time = time.time() - start_time
            execution_result = ToolExecutionResult(
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                metadata={
                    "exception_type": type(e).__name__,
                    "circuit_state": self.state.value,
                },
            )

            self._record_failure(execution_result)
            return execution_result

    async def _execute_with_monitoring(
        self, func: Callable, *args, **kwargs
    ) -> Any:
        """Execute function with additional monitoring"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _detect_storm(self) -> bool:
        """Detect if there's a request storm"""
        now = datetime.now(UTC)

        # Add current request
        self.recent_requests.append(now)

        # Clean old requests outside the window
        cutoff_time = now - timedelta(
            seconds=self.config.storm_detection_window
        )
        self.recent_requests = [
            req for req in self.recent_requests if req > cutoff_time
        ]

        # Check if storm threshold is exceeded
        if len(self.recent_requests) > self.config.storm_threshold:
            if not self.storm_detected:
                self.storm_detected = True
                self.logger.warning(
                    f"Request storm detected for {self.tool_name}: "
                    f"{len(self.recent_requests)} requests in "
                    f"{self.config.storm_detection_window}s"
                )
            return True
        else:
            if self.storm_detected:
                self.storm_detected = False
                self.logger.info(f"Request storm cleared for {self.tool_name}")
            return False

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset from OPEN state"""
        if self.last_failure_time is None:
            return True

        time_since_failure = datetime.now(UTC) - self.last_failure_time
        return time_since_failure.total_seconds() >= self.adaptive_timeout

    def _record_success(self, result: ToolExecutionResult):
        """Record successful execution and update circuit state"""
        self.success_count += 1
        self.failure_count = 0  # Reset failure count on success

        # Add to history
        self._add_to_history(result)

        # Update circuit state
        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.state_changed_time = datetime.now(UTC)
                self.success_count = 0
                self.logger.info(
                    f"Circuit breaker for {self.tool_name} reset to CLOSED"
                )

        # Adapt thresholds based on recent performance
        self._adapt_thresholds()

    def _record_failure(self, result: ToolExecutionResult):
        """Record failed execution and update circuit state"""
        self.failure_count += 1
        self.success_count = 0  # Reset success count on failure
        self.last_failure_time = datetime.now(UTC)

        # Add to history
        self._add_to_history(result)

        # Update circuit state
        if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
            if self.failure_count >= self.adaptive_failure_threshold:
                self.state = CircuitState.OPEN
                self.state_changed_time = datetime.now(UTC)
                self.logger.warning(
                    f"Circuit breaker for {self.tool_name} OPENED after "
                    f"{self.failure_count} failures"
                )

        # Adapt thresholds based on failure patterns
        self._adapt_thresholds()

    def _add_to_history(self, result: ToolExecutionResult):
        """Add execution result to history with size limit"""
        self.execution_history.append(result)
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[
                -self.max_history_size // 2 :
            ]

    def _calculate_confidence(self, execution_time: float) -> float:
        """Calculate confidence based on execution time and history"""
        if not self.execution_history:
            return 0.5

        # Get recent successful executions
        recent_successes = [
            r for r in self.execution_history[-20:] if r.success
        ]
        if not recent_successes:
            return 0.3

        # Calculate confidence based on execution time consistency
        recent_times = [r.execution_time for r in recent_successes]
        avg_time = np.mean(recent_times)
        std_time = np.std(recent_times)

        # Confidence decreases with longer execution times and higher variance
        time_factor = max(0.1, 1.0 - (execution_time / max(avg_time * 2, 1.0)))
        consistency_factor = max(0.1, 1.0 - (std_time / max(avg_time, 0.1)))

        # Success rate factor
        success_rate = len(recent_successes) / min(
            20, len(self.execution_history)
        )

        confidence = (
            time_factor * 0.3 + consistency_factor * 0.3 + success_rate * 0.4
        )
        return min(1.0, max(0.0, confidence))

    def _adapt_thresholds(self):
        """Adapt circuit breaker thresholds based on performance patterns"""
        if len(self.execution_history) < 10:
            return

        recent_results = self.execution_history[-50:]
        success_rate = sum(1 for r in recent_results if r.success) / len(
            recent_results
        )

        # Adapt failure threshold based on stability
        if success_rate > 0.9:
            # Very stable, can be more lenient
            self.adaptive_failure_threshold = min(
                self.config.failure_threshold * 2,
                self.config.failure_threshold + 3,
            )
        elif success_rate < 0.5:
            # Unstable, be more strict
            self.adaptive_failure_threshold = max(
                self.config.failure_threshold // 2, 2
            )
        else:
            # Moderate stability, use default
            self.adaptive_failure_threshold = self.config.failure_threshold

        # Adapt timeout based on execution times
        if recent_results:
            avg_execution_time = np.mean(
                [r.execution_time for r in recent_results if r.success]
            )
            if avg_execution_time > 0:
                # Adjust timeout based on typical execution time
                self.adaptive_timeout = max(
                    self.config.recovery_timeout, avg_execution_time * 10
                )

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive circuit breaker statistics"""
        recent_results = self.execution_history[-100:]

        stats = {
            "tool_name": self.tool_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "state_changed_time": self.state_changed_time.isoformat(),
            "storm_detected": self.storm_detected,
            "config": self.config.to_dict(),
            "adaptive_thresholds": {
                "failure_threshold": self.adaptive_failure_threshold,
                "timeout": self.adaptive_timeout,
            },
            "performance": {
                "total_executions": len(self.execution_history),
                "recent_executions": len(recent_results),
            },
        }

        if recent_results:
            successful_recent = [r for r in recent_results if r.success]
            stats["performance"].update(
                {
                    "recent_success_rate": len(successful_recent)
                    / len(recent_results),
                    "avg_execution_time": (
                        np.mean([r.execution_time for r in successful_recent])
                        if successful_recent
                        else 0.0
                    ),
                    "avg_confidence": (
                        np.mean([r.confidence for r in successful_recent])
                        if successful_recent
                        else 0.0
                    ),
                }
            )

        return stats

    def reset(self):
        """Reset circuit breaker to initial state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state_changed_time = datetime.now(UTC)
        self.storm_detected = False
        self.recent_requests.clear()
        self.adaptive_failure_threshold = self.config.failure_threshold
        self.adaptive_timeout = self.config.recovery_timeout
        self.logger.info(f"Circuit breaker for {self.tool_name} reset")


class AdaptiveEVCalculator:
    """
    Adaptive Expected Value calculator using multi-armed bandit optimization
    """

    def __init__(self, tool_name: str, learning_rate: float = 0.1):
        self.tool_name = tool_name
        self.learning_rate = learning_rate

        # Initialize bandit for EV calculation
        self.bandit = NonStationaryBandit(
            n_arms=1, learning_rate=learning_rate  # Single tool EV
        )

        # Track execution outcomes
        self.execution_rewards: list[float] = []
        self.execution_costs: list[float] = []
        self.execution_times: list[float] = []

        # EV components
        self.current_ev = 0.0
        self.confidence_in_ev = 0.0
        self.last_updated = datetime.now(UTC)

    def update_ev(
        self, result: ToolExecutionResult, expected_value: float = None
    ) -> float:
        """
        Update expected value based on execution result
        """
        # Calculate reward based on success and execution time
        if result.success:
            # Reward based on speed and confidence
            time_factor = max(
                0.1, 1.0 - (result.execution_time / 30.0)
            )  # Normalize to 30s
            confidence_factor = result.confidence
            reward = (time_factor * 0.5 + confidence_factor * 0.5) * 100
        else:
            # Penalty for failures
            reward = -50  # Fixed penalty

        # Calculate cost (mainly execution time)
        cost = result.execution_time * 10  # Cost per second

        # Net value
        net_value = reward - cost

        # Update bandit with net value
        self.bandit.update(0, net_value)  # Single arm bandit

        # Track history
        self.execution_rewards.append(reward)
        self.execution_costs.append(cost)
        self.execution_times.append(result.execution_time)

        # Limit history size
        max_history = 500
        if len(self.execution_rewards) > max_history:
            self.execution_rewards = self.execution_rewards[
                -max_history // 2 :
            ]
            self.execution_costs = self.execution_costs[-max_history // 2 :]
            self.execution_times = self.execution_times[-max_history // 2 :]

        # Calculate new EV
        if self.execution_rewards:
            recent_rewards = self.execution_rewards[-50:]  # Recent performance
            recent_costs = self.execution_costs[-50:]

            avg_reward = np.mean(recent_rewards)
            avg_cost = np.mean(recent_costs)
            self.current_ev = avg_reward - avg_cost

            # Calculate confidence based on consistency
            reward_std = np.std(recent_rewards)
            cost_std = np.std(recent_costs)
            consistency = 1.0 / (1.0 + reward_std + cost_std)

            # Sample size confidence
            sample_confidence = min(1.0, len(recent_rewards) / 20.0)

            self.confidence_in_ev = consistency * sample_confidence

        self.last_updated = datetime.now(UTC)
        return self.current_ev

    def get_ev_estimate(self) -> dict[str, Any]:
        """Get current expected value estimate with confidence"""
        return {
            "tool_name": self.tool_name,
            "expected_value": self.current_ev,
            "confidence": self.confidence_in_ev,
            "last_updated": self.last_updated.isoformat(),
            "sample_size": len(self.execution_rewards),
            "recent_performance": {
                "avg_reward": (
                    np.mean(self.execution_rewards[-20:])
                    if self.execution_rewards
                    else 0.0
                ),
                "avg_cost": (
                    np.mean(self.execution_costs[-20:])
                    if self.execution_costs
                    else 0.0
                ),
                "avg_execution_time": (
                    np.mean(self.execution_times[-20:])
                    if self.execution_times
                    else 0.0
                ),
            },
        }

    def should_use_tool(self, threshold: float = 0.0) -> bool:
        """Determine if tool should be used based on EV"""
        # Allow tools with no history to run (bootstrap period)
        if len(self.execution_rewards) == 0:
            return True
        return self.current_ev > threshold and self.confidence_in_ev > 0.3


class UnifiedToolGatingSystem:
    """
    Unified tool gating system combining circuit breakers and adaptive EV calculation
    """

    def __init__(self):
        self.circuit_breakers: dict[str, AdvancedCircuitBreaker] = {}
        self.ev_calculators: dict[str, AdaptiveEVCalculator] = {}
        self.global_config = CircuitBreakerConfig()

        # System-wide metrics
        self.total_executions = 0
        self.total_successes = 0
        self.system_start_time = datetime.now(UTC)

        # Logging
        self.logger = logging.getLogger("unified_tool_gating")

    def register_tool(
        self, tool_name: str, config: CircuitBreakerConfig = None
    ) -> bool:
        """Register a new tool with the gating system"""
        try:
            if tool_name not in self.circuit_breakers:
                self.circuit_breakers[tool_name] = AdvancedCircuitBreaker(
                    tool_name, config or self.global_config
                )
                self.ev_calculators[tool_name] = AdaptiveEVCalculator(
                    tool_name
                )
                self.logger.info(f"Registered tool: {tool_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to register tool {tool_name}: {e}")
            return False

    def warm_up_tool(
        self,
        tool_name: str,
        initial_ev: float = 0.5,
        initial_confidence: float = 0.4,
    ) -> bool:
        """Warm up a tool for testing by setting initial EV values"""
        if tool_name in self.ev_calculators:
            ev_calc = self.ev_calculators[tool_name]
            ev_calc.current_ev = initial_ev
            ev_calc.confidence_in_ev = initial_confidence
            self.logger.info(
                f"Warmed up tool {tool_name} with EV={initial_ev}, confidence={initial_confidence}"
            )
            return True
        return False

    async def execute_tool(
        self, tool_name: str, tool_func: Callable, *args, **kwargs
    ) -> ToolExecutionResult:
        """
        Execute tool through unified gating system
        """
        self.total_executions += 1

        # Ensure tool is registered
        if tool_name not in self.circuit_breakers:
            self.register_tool(tool_name)

        circuit_breaker = self.circuit_breakers[tool_name]
        ev_calculator = self.ev_calculators[tool_name]

        # Check if tool should be used based on EV
        if not ev_calculator.should_use_tool():
            self.logger.info(
                f"Tool {tool_name} skipped due to low expected value"
            )
            return ToolExecutionResult(
                success=False,
                execution_time=0.0,
                error_message="Tool execution skipped - low expected value",
                metadata={
                    "reason": "low_ev",
                    "expected_value": ev_calculator.current_ev,
                    "confidence": ev_calculator.confidence_in_ev,
                },
            )

        # Execute through circuit breaker
        result = await circuit_breaker.call(tool_func, *args, **kwargs)

        # Update EV calculation
        ev_calculator.update_ev(result)

        # Update global metrics
        if result.success:
            self.total_successes += 1

        # Add system-wide metadata
        result.metadata.update(
            {
                "tool_name": tool_name,
                "gating_system": "unified_v1.0",
                "expected_value": ev_calculator.current_ev,
                "ev_confidence": ev_calculator.confidence_in_ev,
            }
        )

        return result

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""
        uptime = datetime.now(UTC) - self.system_start_time

        return {
            "system_info": {
                "uptime_seconds": uptime.total_seconds(),
                "total_executions": self.total_executions,
                "total_successes": self.total_successes,
                "overall_success_rate": (
                    self.total_successes / max(1, self.total_executions)
                ),
                "registered_tools": len(self.circuit_breakers),
            },
            "tool_status": {
                tool_name: {
                    "circuit_breaker": cb.get_stats(),
                    "expected_value": self.ev_calculators[
                        tool_name
                    ].get_ev_estimate(),
                }
                for tool_name, cb in self.circuit_breakers.items()
            },
            "global_config": self.global_config.to_dict(),
        }

    def reset_tool(self, tool_name: str) -> bool:
        """Reset circuit breaker and EV calculator for specific tool"""
        if tool_name in self.circuit_breakers:
            self.circuit_breakers[tool_name].reset()
            # Recreate EV calculator
            self.ev_calculators[tool_name] = AdaptiveEVCalculator(tool_name)
            self.logger.info(f"Reset tool gating for: {tool_name}")
            return True
        return False

    def update_global_config(self, config: CircuitBreakerConfig):
        """Update global configuration for all circuit breakers"""
        self.global_config = config
        # Apply to existing circuit breakers
        for cb in self.circuit_breakers.values():
            cb.config = config
        self.logger.info("Updated global circuit breaker configuration")


# Export main classes
__all__ = [
    "UnifiedToolGatingSystem",
    "AdvancedCircuitBreaker",
    "AdaptiveEVCalculator",
    "CircuitBreakerConfig",
    "ToolExecutionResult",
    "CircuitState",
]
