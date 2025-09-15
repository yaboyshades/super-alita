# Codecraft Addendum (CMA v5.3)

## Purpose
This addendum extends the [Super-Alita Constitutional Framework](../../.github/CONSTITUTION.md) with Codecraft-specific operating notes. It preserves the constitutional articles while clarifying how Specification-Driven Development (SDD) powers every Codecraft engagement.

## Mandated Workflow
- **SDD is the default**. All Codecraft work must begin with `specify → plan → tasks` using the constitutional gates that ship with `src/sdd/`.
- **No bypassing phases.** Do not generate code before a compliant specification, implementation plan, and task graph are produced.
- **Constitutional gates stay on.** Requests must keep `constitutional_gates=True`; treat any `compliance_threshold_met=False` result as blocking until the artifacts are revised.

## Tooling Alignment
- **CLI:** `python -m src.sdd.sdd_cli ...` remains the canonical interface for Codecraft automation. The CLI exposes `specify`, `plan`, `tasks`, and validation helpers wired to the enhanced framework.
- **Shell helpers:**
  - `scripts/lib/constitutional-gates.sh` checks specifications and plans for CMA rule coverage (feature IDs, DoD, etc.).
  - `scripts/lib/sdd-common.sh` provides shared shell helpers (for example, canonical slug generation) so bespoke Codecraft scripts stay consistent with the Python implementation.
- **Smoke tests:** `scripts/smoke_sdd_specify.py` issues a minimal SDD `/sdd/specify` request against a running runtime to confirm that the constitutional gates and FastAPI wiring are intact.

## Telemetry and Reporting
- Emit the standard REUG events for every Codecraft turn: `STATE_TRANSITION`, `TaskStarted`, `AbilityCalled/Succeeded/Failed`, and `TaskSucceeded/Failed`. The streaming router must still respect the single terminal event rule.
- Persist the generated SDD artifacts under `specs/` and capture their hashes in telemetry or task tracking systems so downstream automation can detect drift.

## Change Management
- Update this addendum whenever CMA articles change or new automation scripts alter the required SDD phase sequencing.
- Run `pre-commit run --all-files` and `pytest -q tests/runtime` before shipping any Codecraft-facing modification. Treat failures as constitutional blockers until resolved.
