# Orchestration Nervous System

AGENTS is the orchestration nervous system for this repository: it connects planning intent, runtime execution, and protective g
uardrails so every agent behaves as part of a cohesive whole. Treat this document as the living wiring diagram for routing signal
s, coordinating swarms, and hardening continuous improvement loops.

## Dynamic Context Routing

The task router adapts orchestration context based on code, telemetry, and workflow signals. The references below point to docume
nted guardrails that now live inside the repo.

```yaml
task_router:
  version: 2
  default_context:
    instructions: docs/orchestration/rules/global.md
    context_window: 4096
  routes:
    - name: sdd_pipeline
      triggers:
        files:
          - "src/sdd/**"
          - "templates/sdd/**"
        events:
          - type: ability
            name: SDDPlanner
      instructions: docs/orchestration/rules/sdd_pipeline.md
      memory:
        - memory/sdd/constitutional_sdd_framework.md
      escalations:
        - condition: plan_requires_runtime_changes
          workflow: .github/workflows/sdd-validation.yml
    - name: reliability_runtime
      triggers:
        files:
          - "src/reug_runtime/**"
          - "tests/runtime/**"
        events:
          - type: telemetry
            name: STATE_TRANSITION
      instructions: docs/orchestration/rules/reug_runtime.md
      escalations:
        - condition: latency_ms > 750
          workflow: .github/workflows/performance-monitoring.yml
  audit:
    owner: runtime-eng@super-alita
    cadence: weekly
```

## Agent Swarm Configuration

Swarm roles, responsibilities, and hand-offs are codified so every autonomous cycle has clear ownership.

```yaml
agents:
  architect:
    instructions: docs/orchestration/agents/architect.md
    focus: planning
    tools:
      - planning.generate_plan
      - planning.summarize_requirements
  runtime:
    instructions: docs/orchestration/agents/runtime.md
    focus: execution
    tools:
      - runtime.execute_stream
      - runtime.emit_telemetry
  guardian:
    instructions: docs/orchestration/agents/guardian.md
    focus: assurance
    tools:
      - quality.audit_telemetry
      - quality.rollback_assessment
handoffs:
  architect_to_runtime:
    playbook: docs/orchestration/handoffs/architect_to_runtime.md
    workflow: .github/workflows/workflow-orchestrator.yml
  runtime_to_guardian:
    playbook: docs/orchestration/handoffs/runtime_to_guardian.md
    workflow: .github/workflows/enhanced-quality-gates.yml
```

## Learning & Improvement

Runtime intelligence is reinforced by the knowledge assets captured in the learning library.

```yaml
learning:
  effective_patterns:
    source: docs/learning/effective_patterns.md
    cadence: per_sprint
    owner: runtime-architects
  common_errors:
    source: docs/learning/common_errors.md
    issue_label: runtime-regression
  quality_gates:
    source: docs/learning/quality_gates.md
    workflows:
      - .github/workflows/enhanced-quality-gates.yml
      - .github/workflows/capability-validate.yml
```

## External Integrations

Key API and data integrations are wired through explicit workflows, scripts, and env-scoped secrets.

```yaml
integrations:
  apis:
    github:
      base_url: https://api.github.com
      secret: env:REUG_GITHUB_TOKEN
      workflow: .github/workflows/auto-sync.yml
      script: scripts/smoke_github_integration_spec.py
    openai:
      base_url: https://api.openai.com/v1
      secret: env:OPENAI_API_KEY
      workflow: .github/workflows/prompt-regression.yml
      script: scripts/validate_capabilities.py
  database:
    telemetry:
      dsn: env:REUG_TELEMETRY_DSN
      migrations: scripts/todo_manager.py
      healthcheck: scripts/telemetry_dashboard.py
      backup_workflow: .github/workflows/performance-monitoring.yml
```

## Autonomous Workflows

Each autonomous workflow sequences scripts and CI automation to protect reliability while accelerating delivery.

```yaml
autonomous_workflows:
  new_feature:
    steps:
      - scripts/check-task-prerequisites.sh
      - scripts/create-new-feature.sh
      - scripts/run_all.py
      - .github/workflows/deps-and-tests.yml
  incident_response:
    steps:
      - scripts/monitor-dev.ps1
      - scripts/final_validation.py
      - scripts/bidirectional_telemetry.py
      - .github/workflows/workflow-orchestrator.yml
```

## Event Triggers

File and git events drive automation that keeps routing metadata current and CI guards engaged.

```yaml
event_triggers:
  git_push_main:
    event: git.push
    workflow: .github/workflows/ci.yml
  runtime_files_changed:
    files:
      - "src/reug_runtime/**"
      - "tests/runtime/**"
    script: scripts/update_agents_md.py
    workflow: .github/workflows/update-agents-md.yml
  docs_updated:
    files:
      - "docs/orchestration/**"
    script: scripts/collect_validation_messages.sh
    workflow: .github/workflows/validate-config.yml
```

## Dynamic Context Sources

- `telemetry.jsonl` — rolling event log consumed by `scripts/telemetry_dashboard.py`.
- `scripts/bidirectional_telemetry.py` — bridge that synchronizes runtime signals into the knowledge graph.
- `.github/workflows/performance-monitoring.yml` — scheduled job that refreshes live latency and retry dashboards.
- `memory/feedback/metrics.md` — curated metrics referenced by the guardian and compliance micro-agents.

## Micro-Agent Library

```yaml
micro_agents:
  compliance_audit:
    instructions: docs/orchestration/micro_agents/compliance_audit.md
    entrypoint: scripts/bidirectional_telemetry.py
    outputs:
      - memory/feedback/metrics.md
  trace_normalizer:
    instructions: docs/orchestration/micro_agents/reliability_review.md
    entrypoint: scripts/export_schema.py
    outputs:
      - telemetry.jsonl
composite_tasks:
  reliability_review:
    instructions: docs/orchestration/micro_agents/reliability_review.md
    components:
      - compliance_audit
      - guardian
    workflow: .github/workflows/performance-monitoring.yml
```

## Continuous Improvement

Feedback loops close the gap between execution telemetry and roadmap intent.

```yaml
feedback_loops:
  cadence:
    weekly:
      retro_template: memory/feedback/retro_template.md
      metrics: memory/feedback/metrics.md
      knowledge_base:
        - memory/feedback/learning_log.md
        - docs/learning/effective_patterns.md
    realtime:
      signals:
        - telemetry.jsonl
        - scripts/telemetry_dashboard.py
      triage_workflow: .github/workflows/performance-monitoring.yml
  ownership:
    lead: runtime steering group
    escalation: docs/orchestration/handoffs/runtime_to_guardian.md
```

---

## Appendix A: Repository Guidelines

### A.1 Project Structure & Module Organization
- Source code: `src/` (planner, sandbox, plugins, telemetry, orchestration). Entry point: `src/main.py`.
- API dev app: `app.py` (served by `uvicorn` via `make run`).
- Tests: `tests/` plus top-level `test_*.py`; mirror `src/` layout.
- Config/docs/tools: `config/`, `docs/`, `extensions/`, `tools/`, `docker/`.

### A.2 Build, Test, and Development Commands
- Install deps: `uv pip install -r requirements.txt -c constraints.txt` (or `make deps`).
- Run runtime server: `python -m src.main`.
- FastAPI dev server: `make run` (serves `app:app` on port 8080).
- Tests:
  - `pytest -q`
  - `pytest -q -k "expr"`
  - `pytest -q -m integration_redis`
- Lint/format: `ruff check .` and `black . -l 88` (or `pre-commit run --all-files`).
- Type-check: `mypy --strict src core` (focus on `src/core`, `src/sandbox`; add `app.py` as needed).

### A.3 Coding Style & Naming Conventions
- Python 3.11+. Use 4-space indentation, double quotes, and explicit type hints.
- Keep functions small and pure; avoid side effects.
- No raw `eval/exec`; use `src/sandbox/exec_sandbox.py` for dynamic code.
- Subprocess/YAML: use `src/core/proc.py` (no `shell=True`) and `src/core/yaml_utils.py`.
- Naming: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.

### A.4 Testing Guidelines
- Framework: `pytest`; target ≥70% coverage for changes.
- Naming: files `test_*.py`; structure tests to mirror `src/` packages.
- Useful patterns: `pytest -k name`, `pytest -m integration_redis`.
- Write unit tests for new modules and critical paths; prefer fast, isolated tests.

### A.5 Commit & Pull Request Guidelines
- Commits: `[module] Short description` (e.g., `[sandbox] Harden exec policy`).
- Before PR: run hooks, type-check, and tests; CI enforces lint/type/test/coverage.
- PRs: include summary, rationale, linked issues, and updated docs/config when applicable.
- Secrets: never commit keys; manage via env or `.env` (see `.env.example`).

### A.6 Security & Run Modes
- All dynamic execution must be sandboxed; do not bypass policy guards.
- Process/YAML must go through repository utilities (`proc.py`, `yaml_utils.py`).
- Modes via `SUPER_ALITA_MODE`: `shadow` (plan), `act` (sandboxed act), `batch` (replay).

### A.7 Spec‑Kit SDD Workflow (Integrated)
Spec‑Driven Development (Spec‑Kit) is a first‑class workflow in this repo. It provides a consistent path from specification → plan
→ tasks, with constitutional validation and test‑first gates.

#### What’s included
- FastAPI endpoints:
  - `POST /sdd/specify`
  - `POST /sdd/plan`
  - `POST /sdd/tasks`
- Key runtime files:
  - `src/sdd/router.py` — FastAPI routes for SDD
  - `src/sdd/models.py` — Pydantic request/response models
  - `src/sdd/enhanced_sdd_framework.py` — SDD pipeline logic (with Mangle integration)
  - `src/sdd/config.py` — SDD configuration and defaults
  - `src/sdd/validators.py` — Constitutional compliance checks
  - `src/orchestration/unified_orchestrator.py` — Orchestrator wired for SDD + reliability
- Templates & memory:
  - `templates/sdd/spec-template.md`
  - `templates/sdd/plan-template.md`
  - `templates/sdd/tasks-template.md`
  - `memory/sdd/constitutional_sdd_framework.md`

#### How to run (Windows PowerShell)
```powershell
uvicorn app:app --reload --port 8080
```

Call SDD endpoints:
```powershell
# /sdd/specify
curl -X POST "http://127.0.0.1:8080/sdd/specify" `
  -H "Content-Type: application/json" `
  -d '{
    "user_input": "Add an SDD pipeline with constitutional validation gates.",
    "context": {"priority": "high"}
  }'

# /sdd/plan
curl -X POST "http://127.0.0.1:8080/sdd/plan" `
  -H "Content-Type: application/json" `
  -d '{
    "feature_id": "feat-sdd-pipeline"
  }'

# /sdd/tasks
curl -X POST "http://127.0.0.1:8080/sdd/tasks" `
  -H "Content-Type: application/json" `
  -d '{
    "feature_id": "feat-sdd-pipeline"
  }'
```

CLI helpers:
```powershell
python -m src.sdd.sdd_cli specify "Implement streaming SDD endpoints" --context '{"owner":"platform"}'
python -m src.sdd.sdd_cli plan feat-sdd-pipeline
python -m src.sdd.sdd_cli tasks feat-sdd-pipeline
```

#### VS Code tasks (quick checks)
- SDD: Validate Environment — ensures key env vars are present
- SDD: Check Runtime — simple health check against the running server
- Run Prompt Pipeline — executes the prompt pipeline for ad‑hoc testing

Use from Command Palette: “Tasks: Run Task”.

#### Quality gates and policies
- Constitutional threshold: overall compliance score ≥ 0.75
- Test‑first convention: unit tests for new modules and critical paths
- Simplicity Gate: small, focused functions; avoid unnecessary complexity
- Integration‑first verification for orchestrated flows
- Security: dynamic execution via `src/sandbox/exec_sandbox.py`; subprocess via `src/core/proc.py` (no `shell=True`); YAML via `s
rc/core/yaml_utils.py` (safe loading)

#### Notes
- The SDD pipeline is integrated into the unified orchestrator and uses the reliability manager (retries, backoff, classification)
 under the hood.
- If repo‑wide linting is noisy due to tools/examples, scope checks to `src/` and core tests first.
