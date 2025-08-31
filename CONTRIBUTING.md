#
# /CONTRIBUTING.md
#
# Description: A guide for contributors, encapsulating the development philosophy,
# quality gates, and procedures from the "Codex Agent" instructions.
#

# Contributor Guide: REUG Runtime Engineer Procedure

This document outlines the standard operating procedure for all contributors. Following these guidelines ensures the repository remains buildable, testable, and shippable.

## 🎯 Core Goals
1.  **Stability:** Keep the application buildable, testable, and shippable at all times.
2.  **Orchestration:** Prefer single-turn, streaming architectures and preserve event telemetry.
3.  **Minimalism:** Make surgical edits rather than broad refactors. Add focused tests for every change.
4.  **Configuration over Code:** If a task can be solved by adjusting configuration, prompts, or schemas, prefer that over writing new code.

## ⚙️ Development Workflow

### 1. Plan
- **Read:** Before coding, review `PATCHMAP.md` and relevant `docs/*.md`.
- **Identify:** Pinpoint all impacted modules and tests in advance.
- **Prioritize:** Check if the goal can be met by tuning prompts, tool contracts, or environment variables first.

### 2. Environment
- Use `Makefile` targets for standard operations:
  - `make deps`: Install dependencies.
  - `make run`: Serve the application.
  - `make test`: Run the full test suite.
  - `make lint`: Format code and run static analysis.

### 3. Making Changes
- **Minimize Scope:** Edit the fewest files necessary.
- **Preserve Contracts:** Maintain streaming and event contracts.
- **Emit Telemetry:** Ensure all significant state transitions and actions emit structured events (e.g., `AbilityCalled`, `TaskSucceeded`).
- **Test:** Add or adjust tests for any new or modified behavior.

### 4. Quality Gate (Pre-Commit)
Before submitting any change, you **must** run and pass the following checks:
1.  `pre-commit run --all-files`
2.  `pytest -q tests/`

If any tests fail, loop on fixing and re-testing until the suite is green.

### 5. Commits & Pull Requests
- **Commit Format:**
  - **Subject:** `<scope>: <change>` (e.g., `lsp: add completion for bonds`)
  - **Body:** Describe what changed, why, any risks, and test coverage.
- **PR Description:** Must include the following sections:
  - **Summary:** High-level overview.
  - **Changes:** Detailed list of modifications.
  - **Verification:** Exact commands to run to verify the fix, with pass/fail notes.
  - **Runtime Impact:** Notes on latency, retries, or schema enforcement.
  - **Observability:** Changes to event shapes or new telemetry fields.
  - **Rollback:** Instructions for reverting the change if it causes issues.

### 6. Safety & Security
- **No Secrets:** Never commit API keys, credentials, or user data.
- **Use Toggles:** Gate features and policies with configuration flags (e.g., `REUG_MAX_TOOL_CALLS`, `REUG_SCHEMA_ENFORCE`).
- **Cap Outputs:** Large data blobs should be emitted as artifacts with an `ArtifactCreated` event, not inlined in logs.

### 7. Integrations
- **Use Fakes:** Provide fake adapters in tests for external services (see `tests/fakes.py` pattern).
- **Use Markers:** Gate tests requiring live external services with pytest markers (e.g., `@pytest.mark.integration_redis`) and ensure they are skipped by default in CI.
