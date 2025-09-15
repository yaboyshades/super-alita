# Implementation Sync Mini-Protocol

Follow this procedure after the specification mini-protocol succeeds and before merging code that claims compliance with the updated spec. The objective is to prove spec freshness, preserve telemetry guarantees, and keep downstream consumers confident that code, tests, and docs share the same truth.

---

## 1. Purpose & Inputs

- **Goal**: land code changes that trace back to the latest spec revision while emitting deterministic telemetry for future audits.
- **Primary Inputs**:
  - Spec hash + correlation ID produced by `sync_spec.md` protocol.
  - Implementation plan or TODO slice tagged with the same correlation ID.
  - Baseline code metrics (coverage, lint status) for regression comparison.
- **Primary Outputs**:
  - Passing test + lint reports referencing the spec hash.
  - Updated code/doc artifacts annotated with protocol metadata.
  - Telemetry events confirming closed-loop completion (`TaskSucceeded`).

---

## 2. Preconditions Checklist

- [ ] `sync_spec.md` protocol finished with `TaskSucceeded` (spec hash available).
- [ ] Implementation TODOs linked to spec atoms are assigned and in "ready" state.
- [ ] Environment variables for runtime safety toggles set (`REUG_MAX_TOOL_CALLS`, `REUG_SCHEMA_ENFORCE`, etc.).
- [ ] Local branch rebased on latest `main`/`master` to avoid stale diffs.

> Missing any precondition requires rerunning the spec protocol or refreshing the branch before continuing.

---

## 3. Closed-Loop Execution Sequence

| Stage | What Happens | Tooling / Telemetry | Exit Condition |
| --- | --- | --- | --- |
| **Event** | Record `STATE_TRANSITION` → `CODE_SYNC_INTENT` with `correlation_id`, `spec_hash`, and target modules. Attach TODO IDs. | `telemetry.emit_state_transition(...)` prior to coding session. | Intent logged and acked. |
| **Atom/Bond** | Update KG with new code atoms (`Implementation`, `TestCase`, `DocUpdate`) and bond them to the originating spec atoms. | `kg_writer.link_code_changes(...)` reading Git diff + coverage JSON. | Each modified file linked to at least one spec atom. |
| **Energy** | Run regression risk scoring (`energy = coverage_delta + defect_history`). High energy triggers expanded testing or review. | `EnergyComputed` event including `pytest` coverage stats. | Energy within tolerance or mitigation TODO created. |
| **TODO** | Mark TODOs as "in-progress" → "done" as code & tests complete; push updates to `.vscode/todos.json` and issue tracker. | `todo_sync.py --close <todo_ids>` | TODO ledger shows closed items referencing commit SHA. |
| **Bandit** | Feed execution metrics (test duration, flake rate, review findings) into adaptive planner to tune future task selection. | `bandit.update_policy(...)` with code telemetry. | Planner acknowledges update (`AbilitySucceeded`). |
| **Reward** | Emit final `TaskSucceeded` summarizing coverage numbers, lint status, artifacts, and spec hash. Attach commit + PR links. | `telemetry.emit_task_succeeded(...)` plus artifact upload. | Protocol complete; merge allowed. |

---

## 4. Required Commands & Evidence

Run these commands and capture artifacts (store logs or HTML reports alongside the PR):

1. `pre-commit run --all-files` — attach summary log with spec hash in header.
2. `pytest -q tests/runtime` (extend selection if other areas touched) — save `pytest` output + coverage delta.
3. `python tools/spec_diff.py --verify <spec_hash>` — optional guard verifying no spec drift since coding began. _(planned implementation: script not yet available)_
4. `python scripts/todo_sync.py --export` — snapshot TODO ledger after updates.

Include command outputs or artifact paths in the PR description under "Verification".

---

## 5. Observability Contract

- Every stage must emit both `STATE_TRANSITION` and success/failure events (`Ability*`, `Task*`).
- Attach `spec_hash`, `commit_sha`, and `todo_ids` to each telemetry payload.
- Large artifacts (coverage HTML, lint logs) should be persisted via `ArtifactCreated` with SHA-256 checksums.
- Circuit breakers (`ToolCircuitOpen`) pause the protocol; resolve root cause before resuming.

---

## 6. Failure Handling & Rollback Triggers

| Scenario | Immediate Action |
| --- | --- |
| Tests fail or coverage drops | Emit `TaskFailed` with `reason="test_regression"`; revert or fix before rerunning protocol. |
| Spec hash mismatch (drift) | Abort, rerun spec protocol, cherry-pick implementation onto refreshed branch. |
| KG ingestion error | Execute `kg_writer.replay()` with previous snapshot; do not merge until deterministic IDs confirmed. |
| Telemetry gap detected | Regenerate events from local logs and attach as remediation artifact before closing TODOs. |

---

## 7. Merge Checklist

- [ ] Protocol ID `sync-code-v1` referenced in commit body or PR.
- [ ] Spec hash + correlation ID present in PR template.
- [ ] Tests, lint, and verification artifacts uploaded and linked.
- [ ] TODO ledger synced and reviewers acknowledged completion.
- [ ] Observability review signed off (runtime impact + telemetry summary).

Once all boxes are checked, proceed with merge and notify stakeholders that the spec-code loop has closed.

