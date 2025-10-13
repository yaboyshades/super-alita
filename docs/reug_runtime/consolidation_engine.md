# Intelligence Consolidation Engine (REUG)

## 1. Architecture Spec
### Target Scope
- **target_scope:** `consolidation_engine` (Intelligence Consolidation Engine within REUG post-turn pipeline).
- **Responsibilities:**
  - Normalize reasoning + action artifacts after each REUG turn.
  - Extract stable patterns and consolidate contextual memory patches.
  - Emit versioned consolidation telemetry to guardian observers.
  - Orchestrate downstream ACE/context evolution via `AbilityRegistry.execute()` and `ACE.store.update_context()` hooks.
- **Out of Scope / Non-Responsibilities:**
  - Real-time turn streaming, ability execution scheduling, or policy enforcement handled elsewhere in `src/reug_runtime/`.
  - Long-horizon planning beyond post-turn consolidation windows.
  - Direct persistence beyond invoking the ACE store adapter.

### Purpose & Goals
- Provide a guarded, idempotent consolidation step triggered post-turn, capped at ≤20 ms average execution latency.
- Maintain backward compatibility by defaulting to a no-op when feature flag `fea.consolidation.post_turn` is disabled.
- Supply observable, contract-first outputs that downstream guardians and dashboards can reason about.

### Invariants & Constitutional Rules
- **INV-1 Idempotent Updates:** Repeated consolidation with identical envelopes must not mutate ACE context more than once. Implemented via deterministic dedupe keys inside `ConsolidationEnvelope.deduplication_key` and replay protection in adapters.
- **INV-2 Fail-Closed Flag:** If `fea.consolidation.post_turn` is disabled or flag service unavailable, consolidation exits without side effects and records a `skip_reason`.
- **INV-3 Schema Stability:** `ConsolidationEvent.v1` payload fields remain backwards compatible; breaking changes require new version topic.
- **INV-4 Constitutional Validation:** All candidate patches must pass `constitutional_checks.pre_consolidation()` before ACE updates. Non-compliance escalates to guardian queue.
- **INV-5 Timing Guard:** Internal stopwatch ensures post-turn average <20 ms, otherwise emits `consolidation_latency_violation` alert and short-circuits further processing for the turn.

### Trust Boundaries
- **Inbound:**
  - REUG Orchestrator hands off `ConsolidationEnvelope` (contains reasoning transcript + tool outcomes).
  - Feature flag provider (may be remote); treat as untrusted input with timeout + fail-closed semantics.
- **Outbound:**
  - ACE store via `ACE.store.update_context()` (privileged write) — guarded by validation + idempotency tokens.
  - `AbilityRegistry.execute()` for optional follow-up abilities (treated as untrusted; wrapped with circuit breaker).
  - `EventBus.publish()` to broadcast consolidation telemetry (append-only, tolerant to failures).

### Data Contracts
- **ConsolidationEnvelope (domain model):**
  ```python
  class ConsolidationEnvelope(BaseModel):
      session_id: str
      turn_id: str
      timestamp: datetime
      agent_snapshot: dict[str, Any]
      reasoning_trace: list[dict[str, Any]]
      tool_outputs: list[dict[str, Any]]
      metadata: dict[str, Any] = Field(default_factory=dict)
      deduplication_key: str
  ```
- **ConsolidationResult:** structured summary emitted to orchestrator.
- **ConsolidationPatch:** diff instructions for ACE update (json-serializable mapping).
- **ConsolidationEvent.v1:** `event_type="ConsolidationEvent"`, topic `reug.consolidation.v1`, payload fields `session_id`, `turn_id`, `patterns`, `validation`, `ace_patch`, `latency_ms`, `status`, `skip_reason` (optional), `trace_id`.

### Failure Modes & Mitigations
| Failure Mode | Detection | Mitigation |
| --- | --- | --- |
| Flag service timeout | Latency timer on flag fetch | Default to disabled, emit structured log w/ reason `flag_timeout` |
| Invalid envelope schema | Pydantic validation error | Return `ConsolidationResult(status="rejected")`, log `validation_error`, no outbound calls |
| ACE update failure | Raised exception | Retry once (configurable), then emit `ConsolidationEvent` with `status="ace_update_failed"` and trigger guardian escalation |
| Event bus unavailable | Publish raises | Increment `consolidation_events_dropped_total`, continue (non-fatal) |
| Latency >20 ms | Stopwatch check | Mark result `status="degraded"`, set `latency_breach=true`, emit alert metric |
| Ability execution failure | AbilityRegistry error | Capture error via structured log, propagate sanitized message in result, no retries |

## 2. Interfaces
- **Domain Service:**
  ```python
  class ConsolidationEngine:
      async def consolidate_post_turn(
          self,
          envelope: ConsolidationEnvelope,
          *,
          request_context: ConsolidationRequestContext,
      ) -> ConsolidationResult: ...
  ```
- **Feature Flag Provider Protocol:** `def is_enabled(key: str, default: bool = False) -> bool`.
- **Event Publisher Protocol:** `async def publish(event: ConsolidationEvent) -> None`.
- **ACE Store Adapter Protocol:** `async def apply_patch(patch: ConsolidationPatch, *, dedupe_key: str) -> ACEUpdateReceipt`.
- **Ability Registry Adapter:** `async def execute(name: str, payload: Mapping[str, Any], *, correlation_id: str) -> AbilityExecutionRecord`.
- **Versioned Event Topic:** `reug.consolidation.v1` (JSON payload abiding `ConsolidationEventPayloadV1`). Next breaking change => `reug.consolidation.v2`.

## 3. Integration Plan
- **REUG Hooks:**
  - `post_turn` hook in `src/reug_runtime/loop.py` instantiates `ConsolidationEngine` via app wiring when feature flag on.
  - Optional `pre_ability` hook to pre-validate ability responses (flagged via `fea.consolidation.pre_ability` future extension — default false).
- **Feature Flag Behavior:**
  - `fea.consolidation.post_turn` default `False`. Flag failure → treat as disabled.
  - Kill-switch override via environment `CONSOLIDATION_FORCE_DISABLED=true`.
- **Migration:**
  - Phase 0: deploy scaffolding (this change) — no behavioral impact.
  - Phase 1: enable in shadow mode on staging via partial sampling; monitor metrics.
  - Phase 2: gradually increase rollout percentage.
- **Rollback:**
  - Toggle flag off → immediate revert to no-op.
  - Remove hook registration to fallback to prior pipeline (documented in `app/consolidation_engine/hooks.py`).

## 4. Safety & Policy Alignment
- **Checks:**
  - `constitutional_checks.pre_consolidation(envelope)` ensures reasoning trace respects constitutional boundaries.
  - Validate ability outputs before invoking ACE store using `constitutional_checks.validate_patch(patch)`.
  - Denylist for sensitive tool outputs enforced prior to telemetry emission.
- **Escalation Paths:**
  - On repeated ACE update failures (>3/10min) escalate to Guardian queue via `guardian.raise_incident(...)` event.
  - Policy violations trigger structured audit log with `severity="policy_violation"` and notify oversight channel.
- **Auditability:**
  - Persist audit trails to structured log sink with correlation_id = `{session_id}:{turn_id}`.
  - Version all emitted events with schema hash for diffing.

## 5. Observability Plan
- **Metrics:**
  - `consolidation_latency_ms` (histogram, target p95 ≤ 25 ms, alert at p95 > 40 ms for 5 min).
  - `patterns_extracted_total` (counter labeled by `status`).
  - `consolidation_skips_total` (counter labeled by `reason`).
  - `ace_patch_idempotent_total` (counter increments on dedupe hits).
- **Structured Logs:** JSON logs at `info` for success, `warning` for skips, `error` for failures. Include `session_id`, `turn_id`, `latency_ms`, `status`, `feature_flag_state`.
- **Traces:** Span `reug.consolidation.post_turn` with child spans for validation, pattern extraction, ACE update. Inject trace_id into events.
- **SLOs:** 99% of consolidations finish <40 ms, error budget <0.5% failure rate. Alert if budget burn >10% hourly.
- **Alerts:**
  - Latency violation alert (PromQL): `histogram_quantile(0.95, rate(consolidation_latency_ms_bucket[5m])) > 0.04`.
  - Error rate alert: `rate(consolidation_failures_total[10m]) / rate(consolidation_attempts_total[10m]) > 0.01`.

## 6. Pytest Suite Blueprint
- **Fixtures:**
  - `flag_provider` fixture w/ controllable enabled/disabled states.
  - `dummy_event_publisher` capturing published events.
  - `ace_store_stub` verifying dedupe behavior.
- **Test Matrix:**
  - **Correctness:** Validate that valid envelope yields deterministic result when flag enabled (xfail until implementation).
  - **Guardrails:** Ensure constitutional rejection raises `PolicyError` and no side effects.
  - **Routing:** Confirm ability registry invoked only when configured.
  - **Multi-Agent:** Parametrize with multiple session_ids verifying dedupe keys unique.
  - **Failure Handling:** Simulate event bus failure → increments dropped counter and still returns result.
  - **Negative:** Invalid envelope raises `ValidationError`.
  - **Property Tests:** Idempotency property ensures repeated calls produce same ACE patch + event payload.

## 7. Code Scaffold Summary
```
domain/
  consolidation_engine/
    __init__.py
    models.py
    service.py
adapters/
  consolidation_engine/
    __init__.py
    feature_flags.py
    event_publisher.py
    ability_registry.py
app/
  consolidation_engine/
    __init__.py
    hooks.py
tests/
  consolidation_engine/
    test_consolidation_engine.py
schema/
  reug_subsystem_manifest.schema.json
manifests/
  consolidation_engine_manifest.json
```
- Skeletons expose protocols, dataclasses, and guardrail placeholders without side effects on import.

## 8. Risk & Validation
- **Assumptions:**
  - Feature flag infrastructure offers synchronous `is_enabled` lookup with ≤5 ms latency.
  - ACE store can accept idempotency tokens for dedupe.
  - Event bus `publish` API returns awaitable; network failures are rare but non-fatal.
- **Open Questions:**
  - Should consolidation operate during streaming (mid-turn) for long turns?
  - What retention period for dedupe cache ensures balance between correctness and memory footprint?
  - How will consolidation interact with multi-session shared memory (team agents)?
- **Red-Team Scenarios:**
  1. Inject adversarial tool output containing prompt injection payload → verify constitutional filter rejects patch and emits incident.
  2. Flood system with duplicate envelopes using unique turn_ids but same dedupe key → ensure dedupe prevents ACE drift.
  3. Force event bus latency spike (>50 ms) → confirm consolidation auto-skips to preserve turn latency budget and logs warning.

## 9. Manifest
- Generated manifest stored at `manifests/consolidation_engine_manifest.json` and validated against `schema/reug_subsystem_manifest.schema.json`.
