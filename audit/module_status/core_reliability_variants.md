# Core Reliability Variants Audit

Date: 2025-09-11
Scope: `src/core/reliability.py` vs `src/core/reliability_optimized.py`
Objective: Analyze overlap, performance intent, and recommend unification strategy.

## Summary Classification
- `reliability.py`: FULL FEATURE BASE (Idempotency, Circuit Breaker, Dead Letter Queue, Backpressure, Integrated Manager). Production semantics oriented, richer guarantees, higher overhead.
- `reliability_optimized.py`: PERFORMANCE-OPTIMIZED SUBSET (Fast-path idempotency, adaptive circuit breaker, configuration levels FAST/BALANCED/STRICT). Omits DLQ, backpressure, and integrated multi-pattern orchestration; focuses on throughput & latency minimization.

## Feature Matrix
| Feature | reliability.py | reliability_optimized.py |
|---------|----------------|---------------------------|
| Idempotency | Redis keyed + content hash + TTL | Bloom + selective Redis (fast negative path) |
| Circuit Breaker | Standard w/ HALF_OPEN transitions | Optimized + adaptive thresholds |
| Dead Letter Queue | Yes (retries + TTL) | No |
| Backpressure | Yes (queue thresholds) | No |
| Reliability Modes | No | Yes (FAST/BALANCED/STRICT) |
| Metrics Depth | Rich (multiple counts, latencies) | Lightweight EMA + efficiency ratios |
| Integrated Manager | `ReliabilityManager` (composition) | `OptimizedReliabilityManager` (conditional components) |
| Retry Handling | DLQ-based exponential backoff | None (caller responsibility) |
| Drop Policy | Backpressure-based drop or enqueue | Not implemented |

## Overlap & Divergence
- Idempotency implementations differ in algorithmic approach: canonical (explicit Redis marking post-success) vs bloom-first (fast negative detection). Both valuable; can be layered.
- Optimized variant lacks DLQ/backpressure; cannot fully replace base in high reliability scenarios.
- Separate metrics schemas complicate unified monitoring.

## Risks
| Risk | Impact |
|------|--------|
| Divergent managers selected inconsistently | Unpredictable reliability guarantees |
| Bloom filter in-memory set growth | Memory bloat over long-running sessions |
| Missing DLQ/backpressure in optimized path | Event loss under failure spikes |
| Duplicate concept naming (ReliabilityManager vs OptimizedReliabilityManager) | Cognitive load and import errors |

## Recommended Unification Strategy
1. Introduce `reliability_unified.py` exporting `UnifiedReliabilityManager` with pluggable strategy:
   - Modes: `fast`, `balanced`, `strict` mapping to capability bundles.
   - Compose feature flags: `use_bloom_fastpath`, `enable_dlq`, `enable_backpressure`, `enable_circuit_breaker`.
2. Merge best-of implementations:
   - Use bloom fast negative path + Redis confirm from optimized variant.
   - Retain DLQ and backpressure from full variant (gated by mode).
   - Single metrics model with optional extended fields.
3. Provide deprecation wrappers in existing two modules re-exporting unified manager with warnings.
4. Add memory guard for bloom filter (size check + periodic purge by TTL buckets).
5. Add tests covering: duplicate suppression success path, circuit open→half_open→closed lifecycle, DLQ retry after simulated failure, backpressure drop metrics.

## Immediate Action Items
- [ ] Create audit test plan document linking required test cases.
- [ ] Draft unified interface design (pydantic settings model for mode flags).
- [ ] Add deprecation banners to variant modules before refactor.

## Mode Mapping Proposal
| Mode | Features Enabled |
|------|------------------|
| fast | bloom idempotency only (Redis confirm) |
| balanced | bloom + Redis idempotency, circuit breaker, lightweight metrics |
| strict | balanced + DLQ + backpressure + extended metrics |

## Metrics Unification Sketch
```yaml
reliability_metrics:
  mode: fast|balanced|strict
  events:
    total: int
    duplicates_prevented: int
    dropped: int
  performance:
    avg_latency_ms: float
    cache_efficiency: float
  circuit_breaker?:
    state: CLOSED|OPEN|HALF_OPEN
    trips: int
    failure_count: int
  dlq?:
    queued: int
    retries: int
  backpressure?:
    queue_utilization: float
    active: bool
  timestamp: float
```

## Decision Log
- Keep both modules temporarily; unify into a new file to avoid breaking import chains prematurely.
- Single public import target planned: `from src.core.reliability_unified import UnifiedReliabilityManager`.

---
Updated after unified implementation draft.
