# Project Context

## Purpose
Super Alita is an event-driven autonomous agent platform that fuses MuZero-inspired planning, reinforcement learning telemetry, and memory chaining to deliver resilient AI operations. The project focuses on orchestrating multi-model toolchains with constitutional guardrails so research prototypes can graduate into production-ready automations.

## Tech Stack
- Python 3.11+
- FastAPI + Uvicorn for orchestration endpoints (`src/main.py`, `app.py`)
- Redis/Memurai-backed event bus for streaming coordination
- Pydantic models for typed contracts
- pytest for unit, integration, and regression suites
- Ruff, Black (88 columns), and MyPy for code quality gates
- npm/TypeScript utilities for MCP and prompt pipeline tooling when required

## Project Conventions

### Code Style
- Enforce Ruff + Black 88 for linting/formatting; run `make lint` or `ruff check .` then `black .` before commits.
- Use explicit type hints everywhere and prefer `pathlib.Path` to raw `os.path` APIs.
- Double quotes are standard; keep functions small, pure, and side-effect aware.
- Never use `eval`/`exec`; dynamic execution must go through `src/sandbox/exec_sandbox.py` and subprocesses through `src/core/proc.py` without `shell=True`.

### Architecture Patterns
- Event-driven neural fabric with Atoms/Bonds knowledge graph captured in `src/core` and `memory/`.
- Unified orchestrator (`src/orchestration/unified_orchestrator.py`) coordinates SDD, REUG runtime, and plugin dispatch.
- Plugins inherit from `PluginInterface` and communicate over the event bus with telemetry hooks in `src/telemetry/`.
- Specification-driven development (Spec-Kit + OpenSpec) governs changes; specs in `openspec/specs` define source-of-truth capabilities mirrored by FastAPI endpoints and REUG abilities.

### Testing Strategy
- Primary command `pytest -q`; comprehensive suites via `make test` or `pytest -q -m <marker>` for focused runs.
- Mirror `src/` directory hierarchy under `tests/`; include regression coverage for new behaviors and maintain ≥70% coverage for touched surfaces.
- Use dedicated fixtures in `tests/conftest.py` and sandbox utilities for deterministic execution.

### Git Workflow
- Branch from `main`; keep changes scoped with verb-led branch names aligned to OpenSpec change IDs.
- Commit messages follow `[module] Short description` (e.g., `[sandbox] Harden exec policy`).
- Run lint, type-checks, and targeted tests before opening a PR; CI enforces Ruff, Black, MyPy, and pytest gates.

## Domain Context
- MuZero-inspired reinforcement learning loops feed decision policies in `src/planning/` and `src/reug_runtime/`.
- Memory chaining persists cognitive atoms in `memory-system/` to support long-horizon reasoning.
- Reliability telemetry is critical: REUG runtime and `src/telemetry/` modules emit traces consumed by guardian workflows.
- Spec-Kit + OpenSpec proposals align research features with constitutional guardrails captured in `memory/sdd/` and `docs/orchestration/` playbooks.

## Important Constraints
- All dynamic code execution must be sandboxed; never bypass `src/sandbox` policies or guardian checks.
- Protect credentials by reading from environment variables or `.env`; secrets must not be committed.
- Keep changes observable: update OpenSpec specs and `tasks.md` checklists before implementation and validate with `openspec validate --strict`.
- Prefer simple, auditable solutions (<100 LOC) unless performance/scale data justifies complexity.

## External Dependencies
- Redis/Memurai for the event bus (`REDIS_URL` in `.env`).
- LLM providers (Gemini, OpenAI, Anthropic) configured via API keys; fallback routing managed in `src/adapters/`.
- MCP tooling integrations (VS Code, Cursor, etc.) orchestrated through `mcp_server/` and `prompt-pipeline` assets.
- Optional observability stacks (e.g., telemetry collectors) triggered via workflows in `.github/workflows/`.
