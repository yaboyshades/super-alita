# Repository Guidelines

## Project Structure & Module Organization

- Source code: [src/](vscode://file/src) (planner, sandbox, plugins, telemetry, orchestration). Entry point: [src/main.py](vscode://file/src/main.py:1).
- API dev app: [app.py](vscode://file/app.py:1) (served by `uvicorn` via `make run`).
- Tests: [tests/](vscode://file/tests) plus top-level `test_*.py`; mirror [src/](vscode://file/src) layout.
- Config/docs/tools: [config/](vscode://file/config), [docs/](vscode://file/docs), [extensions/](vscode://file/extensions), [tools/](vscode://file/tools), [docker/](vscode://file/docker).

## Build, Test, and Development Commands

- Install deps: `uv pip install -r requirements.txt -c constraints.txt` (or `make deps`) ([requirements.txt](vscode://file/requirements.txt:1), [constraints.txt](vscode://file/constraints.txt:1)).
- Run runtime server: `python -m src.main`.
- FastAPI dev server: `make run` (serves `app:app` on port 8080).
- Tests:
  - `pytest -q`
  - `pytest -q -k "expr"`
  - `pytest -q -m integration_redis`
- Lint/format: `ruff check .` and `black . -l 88` (or `pre-commit run --all-files`).
- Type-check: `mypy --strict src core` (focus on [src/core](vscode://file/src/core), [src/sandbox](vscode://file/src/sandbox); add [app.py](vscode://file/app.py:1) as needed).

## Coding Style & Naming Conventions

- Python 3.11+. Use 4-space indentation, double quotes, and explicit type hints.
- Keep functions small and pure; avoid side effects.
- No raw `eval/exec`; use [src/sandbox/exec_sandbox.py](vscode://file/src/sandbox/exec_sandbox.py:1) for dynamic code.
- Subprocess/YAML: use [src/core/proc.py](vscode://file/src/core/proc.py:1) (no `shell=True`) and [src/core/yaml_utils.py](vscode://file/src/core/yaml_utils.py:1).
- Naming: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.

## Testing Guidelines

- Framework: `pytest`; target >=70% coverage for changes.
- Naming: files `test_*.py`; structure tests to mirror `src/` packages.
- Useful patterns: `pytest -k name`, `pytest -m integration_redis`.
- Write unit tests for new modules and critical paths; prefer fast, isolated tests.

## Commit & Pull Request Guidelines

- Commits: `[module] Short description` (e.g., `[sandbox] Harden exec policy`).
- Before PR: run hooks, type-check, and tests; CI enforces lint/type/test/coverage.
- PRs: include summary, rationale, linked issues, and updated docs/config when applicable.
- Secrets: never commit keys; manage via env or `.env` (see [.env.example](vscode://file/.env.example:1)).

## Security & Run Modes

- All dynamic execution must be sandboxed; do not bypass policy guards.
- Process/YAML must go through repository utilities ([src/core/proc.py](vscode://file/src/core/proc.py:1), [src/core/yaml_utils.py](vscode://file/src/core/yaml_utils.py:1)).
- Modes via `SUPER_ALITA_MODE`: `shadow` (plan), `act` (sandboxed act), `batch` (replay).

## Spec-Kit SDD Workflow (Integrated)

Spec-Driven Development (Spec-Kit) is a first-class workflow in this repo. It provides a consistent path from specification -> plan -> tasks, with constitutional validation and test-first gates.

### What's included

- FastAPI endpoints:
  - `POST /sdd/specify`
  - `POST /sdd/plan`
  - `POST /sdd/tasks`
- Key runtime files:
  - [src/sdd/router.py](vscode://file/src/sdd/router.py:1) - FastAPI routes for SDD
  - [src/sdd/models.py](vscode://file/src/sdd/models.py:1) - Pydantic request/response models
  - [src/sdd/enhanced_sdd_framework.py](vscode://file/src/sdd/enhanced_sdd_framework.py:1) - SDD pipeline logic (with Mangle integration)
  - [src/sdd/config.py](vscode://file/src/sdd/config.py:1) - SDD configuration and defaults
  - [src/sdd/validators.py](vscode://file/src/sdd/validators.py:1) - Constitutional compliance checks
  - [src/orchestration/unified_orchestrator.py](vscode://file/src/orchestration/unified_orchestrator.py:1) - Orchestrator wired for SDD + reliability
- Templates & memory:
  - [templates/sdd/spec-template.md](vscode://file/templates/sdd/spec-template.md:1)
  - [templates/sdd/plan-template.md](vscode://file/templates/sdd/plan-template.md:1)
  - [templates/sdd/tasks-template.md](vscode://file/templates/sdd/tasks-template.md:1)
  - [memory/sdd/constitutional_sdd_framework.md](vscode://file/memory/sdd/constitutional_sdd_framework.md:1)

### How to run (Windows PowerShell)

1. Start the API (development):

```powershell
uvicorn app:app --reload --port 8080
```

1. Call SDD endpoints:

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

1. Use the CLI (sync wrappers around async SDD calls):

```powershell
# Specify → Plan → Tasks
python -m src.sdd.sdd_cli specify "Implement streaming SDD endpoints" --context '{"owner":"platform"}'
python -m src.sdd.sdd_cli plan feat-sdd-pipeline
python -m src.sdd.sdd_cli tasks feat-sdd-pipeline
```

### PowerShell Profile Setup for Spec Kit Integration

To enable seamless Spec Kit workflows in PowerShell, add the following to your PowerShell profile (`$PROFILE`):

```powershell
function Invoke-SpecKitWorkflow {
    param([string]$FeatureName, [string]$Phase = "all")
    switch ($Phase) {
        "constitution" { uvx spec-kit constitution $FeatureName }
        "specify" { uvx spec-kit specify $FeatureName }
        "plan" { uvx spec-kit plan $FeatureName }
        "tasks" { uvx spec-kit tasks $FeatureName }
        "implement" { uvx spec-kit implement $FeatureName }
        "all" {
            uvx spec-kit constitution $FeatureName
            uvx spec-kit specify $FeatureName
            uvx spec-kit plan $FeatureName
            uvx spec-kit tasks $FeatureName
            uvx spec-kit implement $FeatureName
        }
    }
}
Set-Alias spec Invoke-SpecKitWorkflow
```

This allows you to run SDD workflows with simple commands like:

- `spec "Add user authentication" plan` - Generate plan for user authentication feature
- `spec "Improve API performance" all` - Run full SDD pipeline for API performance improvement

### VS Code tasks (quick checks)

- SDD: Validate Environment — ensures key env vars are present
- SDD: Check Runtime — simple health check against the running server
- Run Prompt Pipeline — executes the prompt pipeline for ad-hoc testing

Use from Command Palette: “Tasks: Run Task”.

### Quality gates and policies

- Constitutional threshold: overall compliance score >= 0.75
- Test-first convention: unit tests for new modules and critical paths
- Simplicity Gate: small, focused functions; avoid unnecessary complexity
- Integration-first verification for orchestrated flows
- Security: dynamic execution via [src/sandbox/exec_sandbox.py](vscode://file/src/sandbox/exec_sandbox.py:1); subprocess via [src/core/proc.py](vscode://file/src/core/proc.py:1) (no `shell=True`); YAML via [src/core/yaml_utils.py](vscode://file/src/core/yaml_utils.py:1) (safe loading)

## Unified Intelligence Layer Integration

The Unified Intelligence Layer is integrated into the Super-Alita ecosystem:

- **Contracts**: [src/unified_intelligence/contracts.yaml](vscode://file/src/unified_intelligence/contracts.yaml:1) - Defines request/response schemas
- **Orchestrator**: [src/unified_intelligence/orchestrator.py](vscode://file/src/unified_intelligence/orchestrator.py:1) - Implements fusion logic and canonical orchestration
- **Golden Fixtures**: [src/unified_intelligence/golden_fixtures.py](vscode://file/src/unified_intelligence/golden_fixtures.py:1) - Contains test fixtures for validation
- **Telemetry**: [src/unified_intelligence/telemetry.py](vscode://file/src/unified_intelligence/telemetry.py:1) - Handles telemetry collection and envelopes
- **Validation Checklist**: [src/unified_intelligence/validation_checklist.py](vscode://file/src/unified_intelligence/validation_checklist.py:1) - Comprehensive quality assurance

## Mangle Reasoning Engine Integration

The Mangle Reasoning Engine provides code analysis capabilities:

- **Code Ingester**: [src/unified_intelligence/code_reasoning/ingester.py](vscode://file/src/unified_intelligence/code_reasoning/ingester.py:1) - AST-based symbol extraction
- **Rule Engine**: [src/unified_intelligence/code_reasoning/rules.py](vscode://file/src/unified_intelligence/code_reasoning/rules.py:1) - Datalog-like rule application
- **Models**: [src/unified_intelligence/code_reasoning/models.py](vscode://file/src/unified_intelligence/code_reasoning/models.py:1) - Data structures for analysis
- **Facts Database**: [src/unified_intelligence/code_reasoning](vscode://file/src/unified_intelligence/code_reasoning) - SQLite storage for code relationships and dependencies

## Quality Assurance Integration

- **Constitutional Gates**: >=75% compliance required for all changes
- **Code Quality Gates**: Complexity <10, 0 circular dependencies, >=70% test coverage
- **Performance Gates**: Algorithmic complexity documented and validated
- **Integration Gates**: Compatibility with Super-Alita ecosystem verified

## Development Workflow Integration

- **SDD Pipeline**: Integrated with unified intelligence for specification -> plan -> tasks
- **Test-First**: Unit tests for new modules and critical paths
- **Contract-First**: Interface definitions before implementation
- **Fusion Logic**: Score combination for intelligent decision routing

### Notes

- The unified intelligence layer uses the reliability manager for retries/backoff
- Mangle reasoning integrates with the sandbox for secure code analysis
- All dynamic execution must be sandboxed; never bypass policy guards
- Telemetry flows through the unified collector for observability
- The SDD pipeline is integrated into the unified orchestrator and uses the reliability manager (retries, backoff, classification) under the hood.
- If repo-wide linting is noisy due to tools/examples, scope checks to `src/` and core tests first.

# Using `vscode://` URIs to Streamline Agent Workflows

To empower AI agents (like Codex) and developers alike with instant, context-aware navigation in VS Code, embed **`vscode://`** URIs in your AGENTS.md or Codex prompts. These links let agents and humans jump directly to files, symbols, and lines in your workspace.

## 1. Enable VS Code URI Handling
1. Ensure VS Code allows URI handling. Add in your user settings (`settings.json`):
   ```json
   {
     "window.openFilesInNewWindow": "off",
     "workbench.editor.enablePreview": false
   }
   ```
2. Register the file opener in Codex's config (global defaults):
   ```toml
   file_opener = "vscode"
   ```

## 2. Constructing `vscode://` Links
The general format is:
```text
vscode://file/{absolute-or-relative-path}:{line}:{column}
```

- **file/**: Required prefix.
- **Path**: Can be absolute (`/Users/alice/project/src/index.ts`) or relative to workspace root (`src/index.ts`).
- **Line & column**: Optional; defaults to line 1, column 1.

### Examples
- Open a file at a specific line:
  ```md
  [Open main.ts at line 42](vscode://file/src/main.ts:42)
  ```
- Jump to a symbol definition (uses VS Code's command URI):
  ```md
  [Go to `initializeAgent` definition](vscode://command:editor.action.revealDefinition?%7B%22resource%22%3A%22file%3Asrc/agent.ts%22%7D)
  ```

## 3. Embedding in AGENTS.md
In your AGENTS.md, situate links under relevant sections:

```markdown
## Dev environment tips
- Use `pnpm dlx turbo run where <project>` to locate a package.
- Jump directly to the agent entrypoint:
  [Open `agent.ts`](vscode://file/src/agent.ts:1)

## Testing instructions
- To inspect failing tests in `agent.test.ts`:
  [Open test file at failure line](vscode://file/src/__tests__/agent.test.ts:37)

## PR instructions
- When reviewing PRs, jump to the diff file:
  [Open PR diff](vscode://command:git.openChange?%7B%22path%22%3A%22src/agent.ts%22%2C%22line%22%3A15%7D)
```

## 4. Automating Link Generation in Codex
Enhance prompts or scripts to have Codex emit URIs automatically:

```shell
codex --profile super-alita-dev \
  -c "print('Open file at failure:', 'vscode://file/' + failed_file + ':' + str(failed_line))"
```

Or use an MCP tool that formats diagnostics into clickable URIs:

```jsonc
{
  "id": "mcp_servers.vscode-links",
  "command": "node",
  "args": ["generate-vscode-links.js"],
  "tool_timeout_sec": 10
}
```

## 5. Benefits of `vscode://` Integration
- **Instant navigation**: Agents and users can jump to the exact location without manual search.
- **Context-rich prompts**: Codex can reference and open code in follow-up turns.
- **Streamlined reviews**: Clickable PR diffs and test failures speed up triage.

By weaving `vscode://` URIs into your AGENTS.md, Codex prompts, and MCP tools, you'll create a **seamless, click-driven** development experience—amplifying productivity for both AI agents and human collaborators.
