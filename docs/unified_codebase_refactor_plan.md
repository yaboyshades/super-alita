# Unified Codebase Refactor Master Plan

## 🎯 Objective
Deliver a cohesive refactor roadmap that converges the runtime, planning, and enhancement subsystems into a single, reliable experience without breaking the REUG streaming invariants or degrading developer ergonomics. The plan focuses on:

- Eliminating duplicated orchestration logic across demos, launchers, and legacy entry points.
- Consolidating ability/tool registration into a single contract-first workflow.
- Standardising telemetry, event lifecycles, and knowledge graph provenance.
- Improving documentation, developer onboarding, and operational readiness.

## ✅ What We Have Going For Us

| Capability | Current Assets | Notes |
| --- | --- | --- |
| **Streaming Runtime** | `src/reug_runtime/router.py`, `reug_runtime/config.py`, rich telemetry tests in `tests/runtime/` | Single-turn server streaming with dynamic tool support and resilient retries. |
| **Ability Ecosystem** | `src/abilities/`, ability registry schemas, dynamic tool synthesis flow | Supports registration + health checks, but duplicated wrappers exist in demos. |
| **Knowledge Graph + Telemetry** | Event bus, KG emitters, `telemetry.jsonl`, `docs/telemetry.md` | Comprehensive event vocabulary (STATE_TRANSITION, Ability events, ArtifactCreated, etc.). |
| **SDD + Mangle + Copilot Integration** | `src/sdd/`, `src/copilot/`, Mangle bridges, unified orchestrator docs (`SYSTEM_UNIFICATION_PLAN.md`) | Conceptually unified, but code paths still branch across demos/launchers. |
| **Operational Tooling** | Makefile targets (`make deps`, `make run`, `make test`), validation scripts | Repeatable workflows already defined. |
| **Testing Framework** | Focused runtime tests, integration demos, multiple pytest entry points | Provides regression coverage for router invariants and ability interactions. |

## ⚠️ Gaps & Fragmentation (What Still Needs Work)

1. **Multiple Entry Points:** `start.py`, `start_super_alita.py`, demo launchers, and CLI utilities all bootstrap different subsets of abilities and integrations.
2. **Duplicated Enhancement Logic:** Copilot, SDD, and Mangle helpers exist in both `src/` and top-level demo scripts, leading to divergent behaviour.
3. **Telemetry Divergence:** Some demos emit partial event lifecycles (missing STATE_TRANSITION or Ability events) which breaks downstream analytics.
4. **Configuration Drift:** Environment variables and feature flags are defined across `.env.example`, docs, and scripts without a single authoritative schema.
5. **Testing Coverage Gaps:** Non-runtime tests lag behind runtime guarantees; property-based coverage is minimal outside `tests/runtime/`.
6. **Documentation Scatter:** Guidance lives across many README variants; onboarding lacks a canonical unification story.

## 🛠️ Refactor Pillars

1. **Runtime Core Consolidation**
   - Promote `src/reug_runtime/router.py` as the single orchestrator entry.
   - Deprecate redundant streaming implementations in demos; refactor them to import the core router.
   - Centralise ability registry wiring in `src/abilities/__init__.py` with a documented contract.

2. **Ability & Tool Unification**
   - Define a canonical ability manifest (JSON or Python config) consumed by runtime and demos.
   - Ensure all dynamic tool creation paths emit TOOL_REGISTERED and SchemaBypass consistently.
   - Add property-based tests around ability argument schemas and retries.

3. **Observability Standardisation**
   - Adopt a shared telemetry emitter that enforces STATE_TRANSITION → Task events sequencing.
   - Provide decorators/utilities for emitting AbilityCalled/Succeeded/Failed with consistent span IDs.
   - Extend `docs/telemetry.md` with new lifecycle diagrams.

4. **Configuration Governance**
   - Introduce a typed settings module (e.g., `src/config/settings.py`) that reads env vars with defaults + validation.
   - Generate documentation from schema (auto docs section in README or `docs/runtime_env_check.md`).

5. **Developer Experience Streamlining**
   - Collapse README variants into a single canonical onboarding doc with per-role appendices.
   - Replace ad-hoc demo scripts with parameterised CLI commands routed through unified runtime.
   - Introduce smoke-test CLI for verifying ability availability and telemetry emission.

## 📅 Phased Implementation Roadmap

### Phase 0 — Discovery & Baseline (Week 0)
- Catalogue all entry points and classify by usage (production, demo, deprecated).
- Capture baseline telemetry from representative runs to establish success metrics.
- Confirm failing/fragile tests and prioritise coverage gaps.

### Phase 1 — Core Runtime Hardening (Weeks 1–2)
- Refactor launchers (`start.py`, `start_super_alita.py`, demo scripts) to import a single runtime bootstrap function.
- Introduce integration tests ensuring every launcher emits the required STATE_TRANSITION and Task events.
- Implement property-based tests for retry + circuit breaker policies using fakes in `tests/runtime/`.

### Phase 2 — Ability Registry Convergence (Weeks 3–4)
- Create canonical ability manifest; update ability registration code to consume it.
- Delete or migrate duplicate ability definitions in demo files.
- Add regression tests verifying manifest parity with runtime registry (round-trip load → register → emit).

### Phase 3 — Observability & Configuration (Weeks 4–5)
- Build shared telemetry helper module; migrate existing emitters.
- Wire typed settings module and replace ad-hoc `os.getenv` usage.
- Document new telemetry lifecycle and settings schema in `docs/telemetry.md` and `docs/runtime_env_check.md`.

### Phase 4 — Experience & Documentation (Weeks 5–6)
- Consolidate READMEs into canonical `README.md` + role-specific appendices.
- Publish operational playbooks: rollout checklist, rollback procedure, and smoke-test script instructions.
- Record video or screenshot walkthroughs (optional) for onboarding.

### Phase 5 — Validation & Rollout (Week 7)
- Execute full `make lint && make test` gate plus targeted integration suites.
- Perform shadow deployment using unified launcher; monitor telemetry for parity with baseline.
- Collect feedback, iterate on gaps, then deprecate legacy scripts.

## 📊 Success Metrics

- **Runtime Stability:** Zero regression in `tests/runtime/` suite; added coverage for manifest + telemetry helpers.
- **Telemetry Completeness:** 100% of orchestrated runs emit full lifecycle events with correlation/span IDs.
- **Developer Flow:** Single documented command (`make run` or equivalent) to start full stack.
- **Documentation Adoption:** Canonical onboarding doc referenced by other guides; reduced duplicate READMEs.

## 🧪 Testing & Quality Gates

- Maintain existing instructions: run `pre-commit run --all-files` and `pytest -q tests/runtime` on each change set.
- Introduce new targeted suites:
  - `tests/runtime/test_manifest_parity.py` — ensures manifest and registry stay aligned.
  - `tests/runtime/test_telemetry_helpers.py` — property-based coverage for event sequencing.
  - `tests/runtime/test_settings_contract.py` — validates env parsing with fuzzed inputs.
- Stress tests: leverage fakes to simulate load spikes on event bus and ensure no telemetry loss.

## 📡 Observability & Telemetry Actions

- Enforce `STATE_TRANSITION` emissions for top-level phases (AWAITING_INPUT → RESPONDING_SUCCESS/TaskFailed).
- Ensure dynamic tools emit TOOL_REGISTERED with hash and artifact caps generate ArtifactCreated events.
- Align telemetry schema with knowledge graph ingestion expectations.

## 🔄 Change Management & Rollback

- Maintain feature flags for new registry + telemetry helpers until validated.
- Provide rollback script to restore legacy launcher bindings if unified bootstrap fails.
- Document fallback plan in deployment guide and ensure monitoring alerts on schema mismatches.

## 🗂️ Backlog & Next Actions

1. Draft canonical ability manifest and validate with runtime team (Owner: Runtime Eng).
2. Prototype typed settings module and integrate into `start.py` (Owner: Platform).
3. Design telemetry helper API and spike tests under `tests/runtime/` (Owner: Observability).
4. Create consolidated onboarding README outline (Owner: Developer Experience).
5. Audit demo scripts for redundant logic and propose merge/deprecation (Owner: Runtime Eng).
6. Schedule shadow deployment rehearsal after Phase 3 completion.

---

**Status:** Planning complete. Awaiting stakeholder sign-off to begin Phase 0 discovery.
