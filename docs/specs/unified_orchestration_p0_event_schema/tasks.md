# Tasks: Unified Orchestration P0 Event Schema

**Spec Source**: docs/specs/unified_orchestration_p0_event_schema_spec.md
**Generated**: 2025-09-18

> Tests must exist and fail before implementation begins. `[P]` marks work streams that can proceed independently.

## Tests-First Backlog (author before code)
- [ ] T001 [P] Add canonical event contract snapshots covering all EventKind payloads and base envelope validation in `tests/orchestration/test_canonical_event_contracts.py` (ensure version, sequence, correlation fields populated).
- [ ] T002 [P] Add NDJSON ledger contract tests using `tmp_path` in `tests/orchestration/test_run_ledger_contract.py` (atomic append ordering, 0600-equivalent permissions, truncation of long previews).
- [ ] T003 [P] Add unified orchestrator stream integration test in `tests/orchestration/test_unified_orchestrator_event_stream.py` (assert ordered emission RunStarted -> ... -> RunTerminated/RunFailed with constitutional score propagation).
- [ ] T004 [P] Add legacy REUG adapter parity test in `tests/reug_runtime/test_legacy_canonical_adapter.py` (compare legacy chunks vs canonical events for identical runs).
- [ ] T005 [P] Add redaction and hashing security tests in `tests/orchestration/test_event_redaction.py` (mask secrets, truncate >200 char fields, stable args hash).

## Event Schema & Sanitization Implementation
- [ ] T010 Implement canonical event dataclasses and EventKind enum matching spec section 4 in `src/orchestration/event_schemas.py` (base envelope plus `data` payload models with type hints).
- [ ] T011 Implement schema builders/validators with `orjson` serialization helpers in `src/orchestration/event_schemas.py` (ensure test fixtures can call `.to_dict()`/`.to_json()`).
- [ ] T012 [P] Introduce sanitization helpers for config redaction, preview truncation, and args hashing in `src/orchestration/event_sanitizer.py`.

## Ledger & Persistence
- [ ] T020 Create append-only run ledger writer with atomic file handling in `src/orchestration/run_ledger.py` (open-with, newline-delimited JSON, Windows ACL note per spec section 7).
- [ ] T021 [P] Wire feature flags/env toggles for ledger plus canonical stream in `src/core/settings.py` and `src/orchestration/config.py` (default shadow mode, opt-in persistence).

## Orchestrator & Adapter Wiring
- [ ] T030 Update `src/orchestration/unified_orchestrator.py` to emit canonical events at each stage boundary, invoke sanitizers, and forward to ledger when enabled.
- [ ] T031 [P] Implement constitutional gating stub producing placeholder score in `src/orchestration/constitutional_gate_stub.py` and integrate with stage completion events.
- [ ] T032 [P] Add canonical event adapter for legacy REUG streaming in `src/reug_runtime/canonical_event_adapter.py` (map legacy tokens to event envelope).
- [ ] T033 Update `src/reug_runtime/router.py` to mirror canonical events via the adapter while preserving legacy stream behavior.

## Observability, Reliability, and Error Taxonomy
- [ ] T040 Extend error taxonomy plus mapping utilities in `src/core/reliability.py` (or new `src/orchestration/errors.py`) to feed `RunError`/`RunFailed` payloads.
- [ ] T041 [P] Publish metrics/log hooks for canonical events in `src/core/telemetry/collector.py` and `src/orchestration/observability.py` (counters, histograms listed in spec section 10).
- [ ] T042 [P] Ensure reliability manager retries/annotations populate `meta` and correlation IDs in `src/orchestration/reliability_manager.py`.

## Documentation & Validation
- [ ] T050 [P] Update developer documentation with canonical event schema and ledger instructions in `docs/specs/unified_orchestration_p0_event_schema_spec.md` and add quickstart snippet in `docs/sdd/README.md`.
- [ ] T051 [P] Add runbook/checklist for enabling shadow-mode ledger in `docs/unified_runtime_spec.md` or new `docs/orchestration/run_ledger_quickstart.md`.
- [ ] T052 Capture validation script to diff legacy vs canonical streams in `scripts/compare_legacy_canonical.py` (used by T004 integration test).
- [ ] T053 [P] Final verification task: run contract plus integration suites (`pytest -q -k "canonical_event or ledger"`) and document results in `docs/specs/unified_orchestration_p0_event_schema/validation.md`.
