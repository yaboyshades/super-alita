# Reliability Error Classification

This document defines the canonical error taxonomy used by the Unified Orchestrator’s ReliabilityManager.

## Classes

- transient: Errors considered recoverable with retry (e.g., timeouts, temporary network failures)
- permanent: Errors considered non-recoverable by retry (e.g., validation errors, logic errors)

## Default Heuristics

- TimeoutError / asyncio.TimeoutError -> transient (configurable)
- OSError and subclasses -> transient (configurable)
- Message contains any of: ["temporarily", "unreachable", "connection reset", "timeout"] -> transient
- Otherwise -> permanent

## Circuit Breaker

- Opens after N consecutive failures (default 3)
- Resets automatically after configurable window (default 60s)

## Implementation Notes

- See `src/orchestration/reliability_manager.py` `_classify_error` for mapping.
- Emitted events: ReliabilityRetryAttempt, ReliabilityRetryScheduled, ReliabilityCircuitOpened, ReliabilityCircuitClosed.
- Per-stage stats expose counts and latency metrics; orchestrator can export telemetry periodically.

## Future Extensions

- Pluggable classifier interface (e.g., structured error types)
- Domain-specific labeling
- Telemetry-enriched classification (rate-limited failures -> network issue)
