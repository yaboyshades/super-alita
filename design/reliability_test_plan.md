# Reliability Unified Test Plan

Status: Draft
Date: 2025-09-11
Scope: UnifiedReliabilityManager (FAST, BALANCED, STRICT)

## Objectives
Validate correctness, performance characteristics, and degradation behaviors of unified reliability layer.

## Test Categories
1. Core Idempotency
2. Bloom Fast Path Accuracy
3. Circuit Breaker Lifecycle
4. Dead Letter Queue Retry Semantics
5. Backpressure Activation & Release
6. Metrics Integrity
7. Mode Behavior Parity (no regression vs legacy)
8. Concurrency & Race Safety
9. Failure Injection & Recovery

## Fixtures
- Redis test instance (flush DB per test class)
- Synthetic event factory (unique payload generator)
- Time control helper (monkeypatch time.time / asyncio.sleep where needed)

## Test Cases
### 1. Core Idempotency
- T1.1 First event processes successfully (status=published)
- T1.2 Duplicate event (same payload hash) returns status=duplicate
- T1.3 Duplicate after TTL expiration processes again

### 2. Bloom Fast Path Accuracy
- T2.1 Non-duplicate not present in bloom (fast negative path)
- T2.2 Duplicate hits bloom then confirmed by Redis (no false negatives)
- T2.3 False positive rate sampling stays < configured error rate (statistical)

### 3. Circuit Breaker Lifecycle
- T3.1 Remains CLOSED under success sequence
- T3.2 Opens after threshold consecutive failures
- T3.3 Transitions to HALF_OPEN after recovery window
- T3.4 Closes after successful probe in HALF_OPEN
- T3.5 Re-opens if probe fails (reset recovery timer)

### 4. Dead Letter Queue Retry
- T4.1 Failure enqueues into DLQ with retry metadata
- T4.2 Exponential backoff intervals respected
- T4.3 Max retries exceeded => permanent failure status recorded
- T4.4 Successful retry removes original entry

### 5. Backpressure
- T5.1 Queue utilization below low watermark => inactive
- T5.2 Surpasses high watermark => active=true, new events queued or rejected depending policy
- T5.3 Utilization drops below low watermark => deactivates

### 6. Metrics Integrity
- T6.1 Metrics snapshot returns required top-level keys
- T6.2 Duplicate increments duplicates counter only once
- T6.3 Circuit trip increments trips count
- T6.4 DLQ enqueue increments depth
- T6.5 Backpressure activation toggles active flag

### 7. Mode Behavior
- T7.1 FAST: No DLQ/backpressure fields in metrics
- T7.2 BALANCED: Circuit present, DLQ/backpressure absent
- T7.3 STRICT: All components active and reported

### 8. Concurrency & Race Safety
- T8.1 Parallel duplicate submissions => only one published
- T8.2 Parallel failures trigger single circuit open transition

### 9. Failure Injection
- T9.1 Redis transient error during idempotency confirm => graceful fallback
- T9.2 Redis outage => circuit opens, events fail predictably
- T9.3 DLQ processing error => entry re-scheduled with incremented retry count

## Performance Benchmarks (Targets)
- FAST median latency < 1ms (local Redis)
- BALANCED median latency < 2.5ms
- STRICT median latency < 5ms under light load (<= 200 eps)

## Tooling & Helpers
- Stopwatch decorator for latency capture
- Bloom filter sampler utility
- Retry scheduler test hook to force run

## Coverage Assertions
- Function coverage >= 90% for reliability module
- Branch coverage >= 80% for circuit breaker & DLQ logic

## Open Items
- Determine load generation strategy (async task swarm vs explicit loop)
- Decide on p95 latency measurement approach (histogram vs record + quantile)

-- End Draft --
