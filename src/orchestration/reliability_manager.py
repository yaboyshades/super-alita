"""Unified Reliability Manager for the Orchestrator.

Lightweight reliability primitives for the unified orchestrator:

* Retry (exponential / linear) + optional deterministic jitter
* Error classification (transient vs permanent)
* Circuit breaker (open after consecutive failures, auto-reset)
* Reliability events (retry attempt/scheduled, circuit open/close)
* Per-stage statistics snapshot

Public API::

        ReliabilityManager(config).execute_with_retries(
                stage, coro_fn, timeout_s, emit_cb
        )

Returned dict keys:
    status, attempts, retries, output|error, classified_error,
    latency_ms, circuit_state

Events (emitted via emit_cb):
    ReliabilityRetryAttempt, ReliabilityRetryScheduled,
    ReliabilityCircuitOpened, ReliabilityCircuitClosed
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from .error_taxonomy import ErrorCode, classify_exception, normalize_error_code

# ----------------------------- Configuration ---------------------------- #


@dataclass(slots=True)
class ReliabilityConfig:
    max_retries: int = 2  # total attempts = 1 + max_retries
    base_backoff_ms: int = 200
    backoff_strategy: str = "exponential"  # exponential|linear
    enable_jitter: bool = False
    jitter_fraction: float = 0.25
    circuit_break_threshold: int = 3
    circuit_reset_after_s: int = 60
    classify_timeout_as_transient: bool = True
    classify_oserror_as_transient: bool = True


@dataclass(slots=True)
class StageStats:
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_latency_ms: int | None = None
    total_latency_ms: int = 0
    count: int = 0
    circuit_open_until: float | None = None

    def record_success(self, latency_ms: int) -> None:  # noqa: D401 - self doc
        self.successes += 1
        self.consecutive_failures = 0
        self.last_latency_ms = latency_ms
        self.total_latency_ms += latency_ms
        self.count += 1

    def record_failure(self, latency_ms: int) -> None:  # noqa: D401 - self doc
        self.failures += 1
        self.consecutive_failures += 1
        self.last_latency_ms = latency_ms
        self.total_latency_ms += latency_ms
        self.count += 1

    def is_circuit_open(self, now: float) -> bool:
        return bool(self.circuit_open_until and now < self.circuit_open_until)

    def maybe_close_circuit(self, now: float) -> bool:
        if self.circuit_open_until and now >= self.circuit_open_until:
            self.circuit_open_until = None
            self.consecutive_failures = 0
            return True
        return False


# ---------------------------- Reliability Core --------------------------- #


class ReliabilityManager:
    """Retry + circuit breaker logic for orchestrator stages.

    Single-threaded async design (no locking). Internal failures are
    swallowed to avoid cascading errors.
    """

    def __init__(self, config: ReliabilityConfig | None = None):
        self.config = config or ReliabilityConfig()
        self._stats: dict[str, StageStats] = {}

    # Public API --------------------------------------------------------- #
    async def execute_with_retries(
        self,
        stage: str,
        coro_fn: Callable[[], Awaitable[Any]],
        timeout_s: int,
        emit_cb: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        *,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        stats = self._stats.setdefault(stage, StageStats())
        now = time.time()
        if stats.is_circuit_open(now):
            return {
                "status": "skipped",
                "attempts": 0,
                "retries": 0,
                "circuit_state": "open",
                "reason": "circuit_open",
                "correlation_id": correlation_id,
            }

        if stats.maybe_close_circuit(now) and emit_cb:
            await self._safe_emit(
                emit_cb,
                {
                    "type": "ReliabilityCircuitClosed",
                    "stage": stage,
                    "time": now,
                },
                correlation_id=correlation_id,
            )

        attempts = 0
        last_error: Exception | None = None
        classified_error: str | None = None
        t_start = time.time()
        total_allowed = 1 + max(0, self.config.max_retries)

        while attempts < total_allowed:
            attempts += 1
            attempt_start = time.time()
            if attempts > 1 and emit_cb:
                await self._safe_emit(
                    emit_cb,
                    {
                        "type": "ReliabilityRetryAttempt",
                        "stage": stage,
                        "attempt": attempts,
                        "max_attempts": total_allowed,
                    },
                    correlation_id=correlation_id,
                )
            try:
                async with asyncio.timeout(timeout_s):  # per-attempt timeout
                    result = await coro_fn()
                latency_ms = int((time.time() - attempt_start) * 1000)
                stats.record_success(latency_ms)
                return {
                    "status": "success",
                    "output": result,
                    "attempts": attempts,
                    "retries": attempts - 1,
                    "classified_error": None,
                    "latency_ms": latency_ms,
                    "circuit_state": "closed",
                    "correlation_id": correlation_id,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                classified_error = self._classify_error(exc)
                latency_ms = int((time.time() - attempt_start) * 1000)
                stats.record_failure(latency_ms)

                if not self._should_retry(
                    classified_error,
                    attempts,
                    total_allowed,
                ):
                    break
                backoff_s = self._compute_backoff(attempts)
                if emit_cb:
                    await self._safe_emit(
                        emit_cb,
                        {
                            "type": "ReliabilityRetryScheduled",
                            "stage": stage,
                            "attempt": attempts,
                            "delay_s": backoff_s,
                            "classified_error": classified_error,
                        },
                        correlation_id=correlation_id,
                    )
                await asyncio.sleep(backoff_s)

        # Exhausted attempts -> failure
        if stats.consecutive_failures >= self.config.circuit_break_threshold:
            stats.circuit_open_until = (
                time.time() + self.config.circuit_reset_after_s
            )
            if emit_cb:
                await self._safe_emit(
                    emit_cb,
                    {
                        "type": "ReliabilityCircuitOpened",
                        "stage": stage,
                        "open_until": stats.circuit_open_until,
                        "failures": stats.consecutive_failures,
                    },
                    correlation_id=correlation_id,
                )

        total_latency_ms = int((time.time() - t_start) * 1000)
        return {
            "status": "failed",
            "error": str(last_error) if last_error else "unknown",
            "attempts": attempts,
            "retries": attempts - 1,
            "classified_error": classified_error,
            "latency_ms": total_latency_ms,
            "circuit_state": (
                "open" if stats.is_circuit_open(time.time()) else "closed"
            ),
            "correlation_id": correlation_id,
        }

    # Internals ---------------------------------------------------------- #
    async def _safe_emit(
        self,
        emit_cb: Callable[[dict[str, Any]], Awaitable[None]],
        event: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        # Swallow any emission errors; reliability shouldn't cascade
        if correlation_id is not None and "correlation_id" not in event:
            event["correlation_id"] = correlation_id
        with suppress(Exception):  # pragma: no cover - defensive
            await emit_cb(event)

    def _should_retry(
        self,
        classified_error: str | None,
        attempts: int,
        total_allowed: int,
    ) -> bool:
        if attempts >= total_allowed or classified_error is None:
            return False
        code = normalize_error_code(classified_error)
        if code == ErrorCode.TIMEOUT:
            return self.config.classify_timeout_as_transient
        if code == ErrorCode.NETWORK_FAILURE:
            return self.config.classify_oserror_as_transient
        return code in {ErrorCode.RATE_LIMIT, ErrorCode.ABILITY_FAILURE}

    def _classify_error(self, exc: Exception) -> str:
        code = classify_exception(exc)
        return code.value

    def _compute_backoff(self, attempts: int) -> float:
        retry_index = max(1, attempts - 1)
        base = self.config.base_backoff_ms / 1000.0
        if self.config.backoff_strategy == "linear":
            delay = base * retry_index
        else:  # exponential
            delay = base * (2 ** (retry_index - 1))
        if self.config.enable_jitter:
            # Deterministic pseudo-jitter via integer hash (no random import)
            frac = ((retry_index * 9301) + 49297) % 233280 / 233280.0
            delay += delay * self.config.jitter_fraction * frac
        return delay

    # Expose stats ------------------------------------------------------- #
    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            k: {
                "successes": v.successes,
                "failures": v.failures,
                "consecutive_failures": v.consecutive_failures,
                "circuit_open_until": v.circuit_open_until,
                "last_latency_ms": v.last_latency_ms,
                "avg_latency_ms": (
                    int(v.total_latency_ms / v.count) if v.count else None
                ),
                "total_latency_ms": v.total_latency_ms,
                "count": v.count,
            }
            for k, v in self._stats.items()
        }

    def get_avg_latency_ms(self, stage: str) -> int | None:
        stats = self._stats.get(stage)
        if not stats or stats.count == 0:
            return None
        return int(stats.total_latency_ms / stats.count)


__all__ = ["ReliabilityManager", "ReliabilityConfig", "StageStats"]
