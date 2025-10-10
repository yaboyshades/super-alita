# Runtime Agent Operating Manual

- Execute plans through `src/reug_runtime/router.py` while maintaining streaming contracts.
- Ensure sandboxed tool calls emit `AbilityCalled`/`AbilitySucceeded` events with correlation IDs.
- Backfill property-based regression tests under `tests/runtime/` for every new execution path.
