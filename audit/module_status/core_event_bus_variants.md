# Core Event Bus Variants Audit (Pre-Consolidation)

Date: 2025-09-11
Scope: `src/core/event_bus.py`, `src/core/event_bus_clean.py`, `src/core/event_bus_old.py`
Objective: Identify overlap, functional deltas, risks, and propose a unification direction.

## Summary Classification
- `event_bus.py`: ACTIVE (Feature-rich, Redis URL flexibility, fallback to in-memory, throughput metrics).
- `event_bus_clean.py`: DUPLICATE / PARTIAL (Singleton pattern, cleaner structure, lacks some dynamic emit logic + fallback paths present in `event_bus.py`).
- `event_bus_old.py`: LEGACY (In-memory async queue model, semantic filtering, history/search; no Redis integration).

## Feature Matrix (Condensed)
| Feature | event_bus.py | event_bus_clean.py | event_bus_old.py |
|---------|--------------|--------------------|------------------|
| Redis/Memurai PubSub | Yes (URL, auth) | Yes (host/port) | No |
| Pattern Subscription | Yes (psubscribe '*') | Yes | N/A |
| In-memory Fallback | Yes (env-controlled) | No | Native (queue) |
| Throughput Metrics | Yes (eps, counts) | Yes (similar) | Basic stats |
| Wildcard Handlers | Yes | Yes | Yes ('*') |
| Structured Event Serialization | Pydantic model_dump | Custom serializer via get_serializer | create_event/serialize_event |
| Semantic Filtering | No | No | Yes (numpy embeddings) |
| Event History / Export | No | No | Yes |
| Reconnection Logic | Partial (reconnect attempt) | Minimal | N/A |
| Backpressure Strategy | Passive (listener sleep) | Passive | Queue bounded by memory |
| Duplicate Handler Guard | Yes | Yes | Implicit (list mgmt) |
| Async Dispatch Mode | Inline sequential | Background tasks set | gather async tasks |
| Fallback Raw Publish | Yes (emit fallback) | No | N/A |

## Gaps & Overlaps
- Two Redis implementations (`event_bus.py` and `event_bus_clean.py`) compete; both maintain metrics & subscription guards.
- Legacy bus (`event_bus_old.py`) offers semantic filtering + history not present in Redis variants.
- Missing consolidated abstraction that supports: Redis transport + optional semantic routing + event history + unified metrics.

## Risks
- Divergent code paths for publish/subscribe semantics increases maintenance burden.
- Redis vs in-memory behavior differences may cause subtle test/environment drift.
- Legacy semantic filtering code may bit-rot; embedding normalization logic not reused elsewhere.
- Multiple public "EventBus" classes could cause accidental import drift (wrong variant used in modules/imports).

## Recommended Unification Strategy (Phaseable)
1. Select `event_bus.py` as the canonical transport base (it supports URL auth + fallback + emit enrichment).
2. Extract reusable mixins:
   - `SemanticFilteringMixin` (from legacy) providing optional embedding-based subscription filters.
   - `EventHistoryMixin` for history, search, export (ported from legacy).
3. Integrate background async dispatch pattern from `event_bus_clean.py` to avoid sequential handler blocking.
4. Provide adapter `LegacyEventBus` shim (deprecated) re-exporting canonical `EventBus` to maintain import stability.
5. Add explicit deprecation warnings in `event_bus_clean.py` and `event_bus_old.py` pointing to unified implementation.
6. Add conformance tests: publish/subscribe, wildcard subscription, semantic filter (if embedding provided), history size cap, metrics increments, in-memory fallback path.
7. Remove deprecated files after two release cycles (mark with `@deprecated` docstring banner now).

## Immediate Action Items
- [ ] Create `src/core/event_bus_unified.py` (or refactor `event_bus.py` directly) with TODO markers for semantic + history integration.
- [ ] Introduce semantic filter interfaces (non-operative stubs) to avoid breaking future integration.
- [ ] Add deprecation headers to `event_bus_clean.py` and `event_bus_old.py`.
- [ ] Add audit test skeleton: `tests/core/test_event_bus_unified.py`.

## Deferred Considerations
- Pluggable serializer strategies (JSON vs MsgPack) with negotiation.
- Metrics export to Prometheus via existing `metrics_registry`.
- Backpressure: bounded async Queue + drop policy metrics.
- Tracing correlation: integrate `correlation_id` + `trace_id` propagation hooks.

## Decision Log
- Canonical base chosen: `event_bus.py` (most feature-complete + fallback support).
- Legacy semantic & history features considered valuable; will be modularized instead of copied inline.
- Consolidation prioritized before broader orchestration refactors to reduce surface area of race conditions.

---
This document will be updated as consolidation progresses.
