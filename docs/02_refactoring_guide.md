# Refactoring Guide

## Overview
This guide maps the changes that introduced the single-turn, server-side streaming orchestrator wired into REUG. It documents how to update the streaming endpoint without breaking invariants such as event telemetry and dynamic tool registration.

## Scope
- `/v1/chat/stream` FastAPI router implementing a single server-turn tool loop with client streaming.
- Observability: state transitions, correlation IDs, event hashing, and telemetry lifecycles.
- Dynamic tools: contract synthesis, health checks, registration, and execution.
- Safety and limits: parser buffer caps, schema enforcement, result capping, retries, circuit breaking, and client disconnect handling.

## Files
- `.pre-commit-config.yaml`
- `pytest.ini`
- `reug_runtime/config.py`
- `reug_runtime/router.py`
- `tests/runtime/`

## Guidance
When refactoring these components:
- Maintain the streaming contract and telemetry events.
- Ensure new behavior includes focused tests under `tests/runtime/`.
- Run `pre-commit run --all-files` and `pytest -q tests/runtime` before committing.

## Further Reading
- [Architectural Overview](./01_architectural_overview.md)
- [Agentic Workflows](./03_agentic_workflows.md)
- [Advanced Development Patterns](./04_advanced_patterns.md)
- [Unified Codebase Refactor Master Plan](./unified_codebase_refactor_plan.md)
