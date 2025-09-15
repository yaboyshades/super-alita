# Specification Sync Mini-Protocol

The specification sync mini-protocol keeps written requirements, constitutional gates, and the knowledge graph aligned before any implementation work begins. Run this playbook whenever a feature spec is created, revised, or re-certified after drift detection.

---

## 1. Purpose & Inputs

- **Goal**: guarantee Article XI (Spec-Code Integrity) by broadcasting spec intent through the closed-loop cognitive model before code changes occur.
- **Primary Inputs**:
  - Updated specification markdown (SDD or minimal template).
  - Previous canonical spec revision (for drift comparison).
  - Context ledger entries (open TODOs, recent telemetry snapshots).
- **Primary Outputs**:
  - Stored diff artifact (artifact bucket / `.alita/spec_diffs/`).
  - Knowledge Graph atom/bond updates capturing new requirements.
  - Refreshed TODO plan, tagged with the spec revision hash.

---

## 2. Preconditions Checklist

- [ ] Source spec passes `SocraticTestingEngine` with readiness ≥ 0.75.  
  <sub><sup>ℹ️ <b>SocraticTestingEngine</b> is an automated spec review tool that scores requirement clarity and completeness. See [SocraticTestingEngine documentation](https://github.com/alita-ai/socratic-testing-engine) for usage and scoring details.</sup></sub>
- [ ] Drift analysis completed (`tools/spec_diff.py` (planned) or Git diff) and summarized.
- [ ] Related feature TODOs in backlog tagged with existing correlation IDs.
- [ ] Stakeholder sign-off recorded (async approval, issue comment, or telemetry event).

> Failing any item aborts the protocol. Resolve gaps, then restart at Stage 0.

---

## 3. Closed-Loop Sync Sequence

| Stage | What Happens | Tooling / Telemetry | Exit Condition |
| --- | --- | --- | --- |
| **Event** | Emit `STATE_TRANSITION` → `SPEC_SYNC_INTENT` with `correlation_id` + `spec_hash`. Capture motivation and impacted features. | `reug_runtime.router`, `telemetry.emit_state_transition(...)` | Intent acknowledged and persisted. |
| **Atom/Bond** | Parse spec sections → create/update KG atoms (`SpecRequirement`, `AcceptanceCriteria`) and bonds to existing tasks. | `kg_writer.ingest_spec(...)`, `ArtifactCreated` for diff snapshot. | All deltas represented in KG with deterministic IDs. |
| **Energy** | Score change impact using `spec_energy_scanner`. High energy flags (drift, dependency breakage) raise TODO severity. | `EnergyComputed` event with `energy_delta`. | Energy ≤ agreed threshold **or** mitigation tasks queued. |
| **TODO** | Materialize or update implementation TODOs referencing the new atoms; sync with `.vscode/todos.json` and planner backlog. | `todo_sync.py --spec <hash>` | TODO ledger reflects revision hash + dependency links. |
| **Bandit** | Update adaptive planner rewards so future task selection prefers freshly aligned specs; log via `BanditPolicyUpdated`. | `bandit.update_policy(...)` with spec telemetry. | Policy ack + `AbilitySucceeded` on update step. |
| **Reward** | Broadcast `TaskSucceeded` summarizing ready-to-code state, linking spec artifact, TODO IDs, and reviewer approvals. | `telemetry.emit_task_succeeded(...)` | Mini-protocol complete; downstream code sync may begin. |

---

## 4. Artifacts to Capture

- Spec diff artifact (markdown or HTML) under `.alita/spec_diffs/<spec_hash>.md`.
- KG import log (JSON) proving deterministic atom/bond creation.
- TODO sync report showing revisions inserted into `.vscode/todos.json`.
- Protocol transcript (events emitted) for auditability.

Store artifact paths inside the telemetry payload for traceability.

---

## 5. Failure & Drift Handling

| Scenario | Response |
| --- | --- |
| Missing stakeholder approval | Emit `TaskFailed` with `reason="approval_missing"`; halt downstream automation. |
| Socratic score < 0.75 | Auto-create TODO: `raise_spec_readiness`; re-run Socratic engine after remediation. |
| KG write conflict (duplicate atom) | Trigger `SchemaBypass` review; manually reconcile IDs, then replay Atom/Bond stage. |
| TODO sync mismatch | Re-run `todo_sync.py`; if persistent, escalate via `ToolCircuitOpen` to pause automation. |

---

## 6. Hand-Off Notes for Implementers

- Reference this protocol ID (`sync-spec-v1`) inside commit messages touching the spec.
- Include spec hash and telemetry correlation ID in PR descriptions for downstream traceability.
- Pass artifact handles to the code sync mini-protocol (see `sync_code.md`) so that implementation can assert spec freshness before merging.

