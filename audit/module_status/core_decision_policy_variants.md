# Core Decision Policy Variants Audit

Date: 2025-09-11
Scope: `src/core/decision_policy.py` and `src/core/decision_policy_v1.py`
Objective: Determine duplication, functional drift, and consolidation path.

## Summary Classification
- `decision_policy.py`: ACTIVE CURRENT (Uses `Dict`/`List`/`Optional` imports but missing from typing – bug risk; richer PlanStep dataclass present but unused downstream; includes PlanStep & ExecutionPlan separation with nested DSL builder; type hints partially inconsistent; some methods reference names that are not imported (e.g., `Dict`, `List`, `Optional`).)
- `decision_policy_v1.py`: DUPLICATE (Near structural clone with minor syntactic differences; slightly more explicit type hints with Python 3.11 `|` unions; removes unused PlanStep dataclass; simpler GoalSynthesizer signature (`_ctx` placeholder)).

## Key Diff Highlights
| Aspect | decision_policy.py | decision_policy_v1.py |
|--------|--------------------|------------------------|
| PlanStep dataclass | Present (unused externally) | Removed |
| ExecutionPlan fields | Same semantic content | Same |
| Type Hint Style | Mix of legacy typing (`Dict`, `List`, `Optional`) but missing imports | Uses native `dict[str, Any]` etc. consistently |
| GoalSynthesizer.synthesize | `(intent, slots, ctx)` uses ctx | `(intent, slots, _ctx)` ignoring context |
| Schema compatibility method | Inlined in candidate loop (no dedicated `schema_compatible` method) | Adds `schema_compatible` method |
| Fallback plan strings | Slight formatting differences | Slight formatting differences |
| Utility function comments | Similar | Similar |
| Parallel/Delegate builders | Same no-op placeholders | Same |
| Risk calculation semantics | Equivalent | Equivalent |
| Unused PlanStep logic | Present | Absent |

## Issues / Risks
1. Two near-identical modules risk divergence; future patches might land in only one.
2. Missing imports (`Dict`, `List`, `Optional`) in `decision_policy.py` would raise NameError at runtime → indicates code path potentially unexecuted.
3. Redundant PlanStep abstraction not integrated with orchestrator / execution path (dead weight).
4. Both versions reimplement basic text similarity & schema matching; no shared utility.
5. Absence of structured validation for plan DSL increases downstream coupling risk.

## Recommended Consolidation
1. Designate `decision_policy_v1.py` as canonical base (cleaner typing, fewer unused constructs).
2. Merge any genuinely needed constructs from `decision_policy.py` (none critical: PlanStep currently unused; ignore unless future DSL expansion required).
3. Create alias module `decision_policy.py` that re-exports `DecisionPolicyEngine` from `decision_policy_v1.py` with deprecation warning OR vice versa (choose final filename for stability—prefer keeping `decision_policy.py` as canonical public path to minimize import churn).
4. Introduce `policy_interface.py` defining minimal protocol for orchestrator: `register_capability`, `decide_and_plan`.
5. Add validation for candidate capability graph size & bandit stat integrity.
6. Add light unit tests for: fallback path (no candidates), single best vs sequential strategy selection thresholds, parallel threshold edge case, risk escalation guardrail.

## Immediate Actions
- [ ] Replace contents of `decision_policy.py` with thin wrapper importing from `decision_policy_v1` plus `DeprecationWarning` if imported directly (or inverse based on chosen canonical file).
- [ ] Add tests under `tests/core/test_decision_policy.py` covering selection logic edges.
- [ ] Remove dead `PlanStep` dataclass unless DSL expansion planned in next milestone.
- [ ] Centralize text similarity & schema fitness helpers (future cross-module extraction candidate: `capability_utils.py`).

## Decision Log
- Canonical public import path will remain `src.core.decision_policy.DecisionPolicyEngine`.
- Internal implementation sourced from refined `decision_policy_v1.py` version.
- Dead abstractions trimmed to reduce cognitive overhead pre-further orchestration work.

---
Pending execution after initial audit batch merges (event bus, plan executor).
