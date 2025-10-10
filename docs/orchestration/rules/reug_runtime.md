# REUG Runtime Routing Rules

These rules activate when tasks touch the streaming runtime (`src/reug_runtime`).

- Respect state machine invariants defined in `src/reug_runtime/router.py`.
- Mirror telemetry schemas from `tests/runtime/fakes.py` when adding new events.
- Update resilience toggles in `reug_runtime/config.py` and document defaults in `README.md`.
- Trigger `.github/workflows/performance-monitoring.yml` after major router changes.
