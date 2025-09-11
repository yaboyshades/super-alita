# Unified Reliability Interface Design

Status: Draft
Date: 2025-09-11
Author: Copilot Agent

## Objective
Unify `reliability.py` and `reliability_optimized.py` into a single configurable reliability layer `UnifiedReliabilityManager` supporting performance and durability trade‑offs via modes.

## Design Principles
1. Zero feature regression (strict mode >= current full reliability)
2. Composability (feature flags inject components lazily)
3. Observability (single, extensible metrics schema)
4. Deterministic degradation (clear documented behavior per mode)
5. Backward compatibility (deprecated shims emit warnings for one release)

## Public API (Proposed)
```python
class ReliabilityMode(StrEnum):
    FAST = "fast"      # minimal idempotency (bloom + async Redis confirm)
    BALANCED = "balanced"  # FAST + adaptive circuit breaker + lightweight metrics
    STRICT = "strict"  # BALANCED + DLQ + backpressure + extended metrics

@dataclass
class ReliabilitySettings:
    mode: ReliabilityMode = ReliabilityMode.BALANCED
    bloom_capacity: int = 500_000
    bloom_error_rate: float = 0.001
    idempotency_ttl_seconds: int = 3600
    circuit_failure_threshold: int = 5
    circuit_recovery_time: float = 30.0
    dlq_max_retries: int = 5
    dlq_retry_backoff_base: float = 0.5
    backpressure_max_queue: int = 10_000
    backpressure_high_watermark: float = 0.85
    backpressure_low_watermark: float = 0.65
    metrics_extended: bool | None = None  # auto-enable when STRICT if None

class UnifiedReliabilityManager:
    def __init__(self, redis: Redis, settings: ReliabilitySettings): ...
    async def process_event(self, event: BaseEvent, publish_cb: Callable[[BaseEvent], Awaitable[Any]], *, enable_idempotency: bool = True) -> dict[str, Any]: ...
    def get_metrics(self) -> dict[str, Any]: ...
    def get_mode(self) -> ReliabilityMode: ...
```

## Internal Components
| Component | Always? | Mode Activation | Notes |
|-----------|---------|-----------------|-------|
| BloomFastPath | Yes | FAST/BALANCED/STRICT | Probabilistic prefilter before Redis check |
| RedisIdempotencyStore | Conditional | when enable_idempotency | Durable dedupe confirmation |
| CircuitBreaker | BALANCED+ | BALANCED/STRICT | Adaptive thresholds (EMA failure rate) |
| DeadLetterQueue | STRICT | STRICT | Redis list + retry schedule index |
| BackpressureController | STRICT | STRICT | Queue size + moving average throughput |
| MetricsCollector | Yes | Extended fields when STRICT or metrics_extended True | Unified schema |

## Processing Pipeline (Sequence)
1. (Optional) Backpressure check (STRICT) – may enqueue or reject
2. (Optional) Idempotency fast path (Bloom) – short circuit duplicate
3. (Optional) Redis idempotency confirm – definitive duplicate check
4. (Optional) Circuit breaker guard – may raise open state
5. Execute publish callback
6. (STRICT) Success: acknowledge; Failure: DLQ enqueue (with retry metadata)
7. Metrics update & structured result assembly

## Result Schema (Draft)
```json
{
  "status": "published|duplicate|circuit_open|queued|failed|retry_scheduled",
  "event_id": "str",
  "mode": "fast|balanced|strict",
  "latency_ms": 0.42,
  "retries": 0,
  "circuit": {"state": "CLOSED", "trips": 1} ,
  "backpressure": {"active": false, "queue_utilization": 0.12},
  "dlq": {"queued": false, "retry_eta": null},
  "metrics_snapshot": {...}
}
```

## Metrics Schema (YAML)
```yaml
reliability:
  mode: fast|balanced|strict
  totals:
    processed: int
    duplicates: int
    failed: int
    dlq_enqueued: int
    dlq_retried: int
  performance:
    avg_latency_ms: float
    p95_latency_ms: float
    bloom_hit_rate: float
    idempotency_confirm_rate: float
  circuit?:
    state: CLOSED|OPEN|HALF_OPEN
    trips: int
    recent_failure_rate: float
  backpressure?:
    queue_size: int
    queue_utilization: float
    active: bool
  dlq?:
    depth: int
    in_retry_window: int
  timestamp: float
```

## Backward Compatibility
- `reliability.py` → import `UnifiedReliabilityManager` and emit `DeprecationWarning`
- `reliability_optimized.py` → same shim
- Provide helper: `create_reliability_manager(redis, mode="balanced")`

## Migration Path
1. Introduce new manager + tests
2. Add shims with warnings
3. Replace internal imports (event bus, reliable event bus)
4. Remove legacy modules after grace period

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Bloom memory growth | Periodic rebuild + capacity monitoring |
| Circuit thrash in BALANCED under burst failures | Add min open duration & exponential backoff |
| DLQ starvation | Retry scheduler with capped concurrency |
| Backpressure rejection floods | Token bucket smoothing + low/high watermark hysteresis |

## Open Questions
- Should STRICT enforce mandatory idempotency? (Leaning yes)
- Do we persist DLQ metadata across restarts? (Redis sorted set with score=next_eta)

## Next Steps
1. Approve interface
2. Implement component scaffolds (no mock placeholders, full code)
3. Write tests (fast path, duplicate, circuit open/half-open lifecycle, DLQ retry, backpressure trigger, metrics)
4. Integrate with ReliableEventBus or replace it

-- End Draft --
