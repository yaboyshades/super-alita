# Unified Orchestration P0 Event Schema & Stage Contract Specification

Status: Draft (P0 Implementation Spec)
Version: 0.1.0
Target Deployment Mode: Shadow (read-only; no prod path changes)
Last Updated: 2025-09-11

## 1. Scope & Objectives (P0)

P0 delivers a canonical event schema and minimal supporting contracts to unify all existing orchestration/streaming paths without breaking legacy clients.

Objectives:

- Define canonical event model (strongly typed, versioned, forward-extensible)
- Provide stage contract definitions for the unified orchestrator
- Introduce minimal run ledger record shape (opt-in persistence) for audit/replay
- Add mapping adapter: legacy REUG streaming → canonical events
- Establish constitutional gating hook points (stub scoring in P0)
- Define error taxonomy & event emission rules
- Specify observability (log/metrics/tracing) field set
- Provide test & validation strategy plus shadow rollout criteria

Out of Scope (Deferred > P0):

- Full replay execution engine
- Advanced persistence indexing / query APIs
- Rich constitutional scoring algorithms (stub only now)
- Cross-run optimization heuristics
- Advanced semantic diff telemetry

Success Criteria (Exit P0):

- Shadow mode runs produce canonical event stream + ledger entries (when enabled)
- No regression in legacy `/v1/chat/stream` semantics
- 100% coverage of event field population for happy path + primary error path
- Contract tests locking JSON shape for each event type

---

## 2. Stakeholders & Assumptions

Stakeholders:

- Orchestrator Runtime (unified)
- Legacy REUG Router / Streaming endpoint
- SDD Workflow Endpoints (`/specify`, `/plan`, `/tasks`)
- Ability / Tool Registry (producers & consumers of ability invocation events)
- Observability & Security modules

Assumptions:

- Python 3.11+, FastAPI, async event emission
- Backward compatibility is mandatory (no rename/removal of existing legacy fields)
- New canonical events can be consumed independently by new clients
- Event publishing is in-process (no external broker introduced P0)
- JSON serialization via `orjson` or standard `json` (implementation detail)
- Monotonic run-scoped sequence numbers

---

## 3. Architecture Overview (Current → Target)

Current:

- Multiple orchestration paths (legacy REUG router vs unified orchestrator)
- Event schemas implicit / loosely structured (string markers, partial payloads)
- No canonical error taxonomy; errors often surfaced as generic failure messages
- No standardized persistence ledger; ad hoc logging only

Target (P0):

- Single canonical event contract produced by unified orchestrator
- Legacy path wrapped by adapter translating to canonical events (shadow only)
- Structured, versioned event payloads (v1) with explicit `kind` discriminator
- Optional ledger append capturing essential run facts (event summary roll-up)
- Constitutional gating hooks (pre-stage, post-stage) returning placeholder scores
- Observability fields consistent across all events (correlation IDs, timestamps)

---

## 4. Canonical Event Schema (v1)

All events share a base envelope:

```text
BaseEvent {
  version: "v1"                    // Schema version (string)
  kind: <EventKind>                 // Discriminator (enum)
  run_id: str                       // UUID for run invocation
  sequence: int                     // Monotonic >= 0 within run
  timestamp: str                    // RFC 3339 UTC
  correlation_id: str               // Stable ID linking related events (stage / ability)
  parent_correlation_id: str|null   // Parent correlation (nesting) or null
  stage: str|null                   // Logical stage name if applicable
  trace_id: str|null                // External tracing correlation (OpenTelemetry) optional
  constitutional_score: float|null  // Present only on gating boundary events (0-1) or null
  meta: { [k: string]: any }        // Non-contract extension map (stable keys discouraged)
}
```

Event Kinds (P0 set):

1. `RunStarted`
2. `StageStarted`
3. `StageCompleted`
4. `AbilityInvocationStarted`
5. `AbilityInvocationChunk` (streamed partial output)
6. `AbilityInvocationCompleted`
7. `RunLog` (structured diagnostic message; not human-freeform logs)
8. `RunError` (non-fatal stage/ability scoped error)
9. `RunTerminated` (graceful end; success flag + summary)
10. `RunFailed` (fatal termination)

Kind-Specific Payloads (embedded under `data` key):

```text
RunStarted.data {
  input_summary: str        // Truncated (≤ 200 chars) synopsis of user input
  config: {                 // Selected normalized config snapshot (filtered safe)
    stages: [str]
    abilities: [str]
    ledger_enabled: bool
  }
}

StageStarted.data {
  name: str
  index: int                 // Zero-based ordering
}

StageCompleted.data {
  name: str
  index: int
  duration_ms: int
  output_summary: str|null   // Truncated representation (< 200 chars) or null
  status: "ok"|"skipped"|"partial"
}

AbilityInvocationStarted.data {
  ability: str
  args_hash: str             // Stable hash (sha256 hex first 16 chars) of sanitized args
}

AbilityInvocationChunk.data {
  ability: str
  chunk: str                 // Stream fragment (UTF-8 text)
  index: int                 // Zero-based chunk index per ability invocation
  is_final: bool             // True only if last chunk and Completed will follow immediately
}

AbilityInvocationCompleted.data {
  ability: str
  duration_ms: int
  result_preview: str|null   // Truncated (< 200 chars) output preview
  status: "ok"|"error"
  error_type: str|null       // Populated if status=error
}

RunLog.data {
  level: "DEBUG"|"INFO"|"WARN"|"ERROR"
  message: str               // Structured, deterministic phrasing
  context: { [k: string]: any } // Safe diagnostic context
}

RunError.data {
  scope: "stage"|"ability"|"system"
  stage: str|null
  ability: str|null
  error_type: str            // Canonical taxonomy code
  message: str               // Safe summarized message
  retryable: bool
}

RunTerminated.data {
  success: bool              // Always true here (graceful completion)
  total_duration_ms: int
  stages_executed: int
  abilities_invoked: int
}

RunFailed.data {
  fatal_error_type: str
  message: str
  last_stage: str|null
  total_duration_ms: int
}
```

Reserved / Invariant Rules:

- `sequence` strictly increments per emitted event (gaps allowed only if future-reserved; not used P0)
- `correlation_id` for stage events: stable UUID per stage; for ability events: nested under stage, parent_correlation_id = stage correlation
- `RunFailed` or `RunTerminated` MUST be the final event in a run
- Exactly one of `RunFailed` or `RunTerminated` is emitted
- `AbilityInvocationChunk.is_final=true` implies a subsequent `AbilityInvocationCompleted` with same correlation chain

Versioning Strategy:

- Additive only in v1 (new optional fields) — breaking changes bump `version`
- Consumers must ignore unknown fields

---

## 5. Stage Contracts

Stages (illustrative ordering — actual enabled subset defined in config):

1. `ingest` – Normalize input, produce internal request object
2. `plan` – (Optional) Generate task decomposition / plan doc
3. `ability_selection` – Resolve abilities required for downstream execution
4. `execute` – Invoke abilities / tools (may stream)
5. `aggregate` – Consolidate outputs into final artifact
6. `gate` – Apply constitutional gating (placeholder P0)
7. `finalize` – Prepare response, compute run metrics

Each stage function signature (conceptual):

```python
async def stage(ctx: OrchestrationContext) -> StageResult:
    ...
```

Preconditions / Postconditions (selected examples):

- ingest: requires raw user input; produces normalized `ctx.request`
- plan: requires `ctx.request`; outputs `ctx.plan` (structured) else skipped if disabled
- ability_selection: requires `ctx.plan` or `ctx.request`; sets `ctx.selected_abilities`
- execute: requires `ctx.selected_abilities`; populates `ctx.ability_results`
- aggregate: requires partial / full ability results; sets `ctx.final_output`
- gate: requires `ctx.final_output`; sets `ctx.constitutional_score`
- finalize: aggregates metrics, emits termination event

Skipping Logic:

- Stage may emit `StageCompleted` with `status="skipped"` when disabled
- Partial status used if recoverable errors occurred but downstream continuity preserved

---

## 6. Error Taxonomy & Handling

Canonical Error Codes (initial set):

- `INGEST_VALIDATION_ERROR`
- `PLAN_GENERATION_ERROR`
- `ABILITY_SELECTION_ERROR`
- `ABILITY_INVOCATION_ERROR`
- `EXECUTION_TIMEOUT`
- `AGGREGATION_ERROR`
- `CONSTITUTIONAL_GATE_ERROR`
- `FINALIZATION_ERROR`
- `SYSTEM_UNEXPECTED_ERROR`

Mapping Rules:

- Stage failure without recovery → emit `RunError` then `RunFailed`
- Ability failure (single) with fallback → emit `RunError` (retryable maybe), continue
- Timeout inside ability invocation → `RunError` with `retryable=false`
- Unexpected exception (uncaught) → escalate to `RunFailed`

Retry Semantics P0:

- No automatic retries (future extension). `retryable` indicates theoretical possibility

---

## 7. Run Ledger (Minimal Persistence)

Opt-in flag: `ledger_enabled` (config boolean). When true, append record at run completion.

Record Shape (JSON line per run):

```text
RunLedgerRecord {
  version: "v1"
  run_id: str
  started_at: str
  ended_at: str
  success: bool
  total_duration_ms: int
  stages: [ { name: str, status: str, duration_ms: int } ]
  abilities: [ { name: str, status: str, duration_ms: int } ]
  constitutional_score: float|null
  final_output_preview: str|null  // Truncated (< 200 chars)
  error: { type: str, message: str } | null
}
```

Storage P0: newline-delimited JSON file: `data/run_ledger.ndjson` (create if absent, append atomic). Future: pluggable store.

---

## 8. Constitutional Gating Integration (P0 Stub)

Hook Points:

- Post-aggregate pre-finalize (`gate` stage)
- Optional per-stage lightweight heuristic (not enforced P0)

Behavior P0:

- Compute placeholder score in [0.0, 1.0] (e.g., fixed 0.82) or simple heuristic length/content
- Attach to `StageCompleted` for `gate` and propagate to `RunTerminated` / `RunFailed`
- Non-blocking (no fail on low score in P0)

Future (>P0): thresholds, remediation loops, violation events.

---

## 9. Security Requirements

- All dynamic code execution must use sandbox (`src/sandbox/exec_sandbox.py`)
- No `subprocess` direct calls; use `src/core/proc.py`
- Redact secrets: detect keys by heuristic (`API_KEY`, `TOKEN`, length > 20) before event emission
- Truncate large outputs (≤ 200 chars) in summaries
- `args_hash` computed after sanitizing PII (drop keys: password, secret, token, key)
- Ledger file permissions: create with mode 600 equivalent (Windows: restrict ACL if feasible)

---

## 10. Observability & Metrics

Standard Log Fields (structured): `run_id`, `correlation_id`, `stage`, `event_kind`, `message`, `duration_ms` (when applicable)

Metrics (counter / timer names):

- `orchestrator_runs_total{success=bool}`
- `orchestrator_stage_duration_ms{stage}` (histogram)
- `orchestrator_ability_duration_ms{ability}` (histogram)
- `orchestrator_errors_total{error_type}`
- `constitutional_score_gauge`

Tracing:

- Inject `trace_id` from inbound request header `X-Trace-Id` if present

---

## 11. Testing & Validation Plan

Test Categories:

1. Unit: stage functions yield expected context mutations
2. Contract: snapshot canonical JSON for each event kind (pytests with stable ordering)
3. Integration: full run through orchestrator with ledger enabled (shadow)
4. Adapter: legacy router stream → canonical mapping produces identical sequence lengths
5. Error Path: induced ability exception -> `RunError` then continuation (if non-fatal)
6. Fatal Path: injected system exception -> `RunFailed` final event
7. Security: redaction of secret-like fields in emitted events / ledger

Fixtures:

- Mock ability returning streaming chunks
- Ability raising deterministic exception

Coverage Target: ≥85% for new orchestration event module

---

## 12. Acceptance Criteria & Rollout

P0 Done When:

- Event dataclasses module implemented & passes contract tests
- Unified orchestrator emits canonical events for each stage path
- Legacy adapter produces canonical event mirror in shadow mode
- Ledger appends correct record for success & failure scenarios
- Constitutional score (stub) present in gate stage + termination events
- All tests green; no regression in legacy endpoint manual smoke

Rollout Phases:

1. Shadow (emit canonical alongside legacy; ledger optional)
2. Dual (new clients consume canonical; legacy remains default)
3. Promote (canonical becomes default; legacy behind flag)

Abort Conditions:

- >2% increase in average run latency (baseline vs shadow)
- Any unhandled exception causing missing termination event

---

## 13. Implementation Checklist (Traceability)

- [ ] events.py module with dataclasses + validation helpers
- [ ] orchestrator instrumentation refactor for StageCompleted + termination
- [ ] legacy adapter wrapper emitting canonical
- [ ] ledger writer utility with atomic append
- [ ] constitutional gating stub
- [ ] redaction + hashing helpers
- [ ] tests: unit, contract, integration, error, security
- [ ] CI inclusion (ruff, mypy, pytest selection)

---

## 14. Open Questions / Future Extensions

- Should AbilityInvocationChunk include token usage metadata? (defer)
- Introduce dedicated Violation events for constitutional gating? (future)
- Move ledger to SQLite for query efficiency? (P1 candidate)
- Add per-ability structured cost metrics? (after token accounting integration)

End of Spec.
