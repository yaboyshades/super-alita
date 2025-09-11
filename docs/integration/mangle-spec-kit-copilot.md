Complete Integration Guide: MANGLE Reasoning + GitHub Spec Kit + GitHub Copilot

This document mirrors your comprehensive guide and adapts it to the Super-Alita repo conventions (repo-first, Python MCP server, SDD workflow). It introduces reasoning-enhanced templates, MCP tool stubs, and CI validation.

- Spec Kit: living specs structure and templates
- MANGLE: deductive reasoning integration points
- Copilot: repo-first guidance and SDD commands

See templates under `tools/templates/reasoning-enhanced/` and MCP tools in `mcp_server_wrapper.py` (tools: `mangle_spec_reason`, `mangle_plan_validate`, `mangle_task_optimize`, `mangle_cross_phase_verify`, `mangle_living_doc_update`). These are safe stubs ready to wire to a MANGLE engine via `src/core/proc.py`.

Note: We intentionally avoided adding a Node MCP server; the repo already runs a Python MCP server. If you want a Node server too, we can add it alongside without interfering.

Quick Start
- VS Code MCP servers are already configured for the Python MCP in `.vscode/settings.json`.
- Use SDD commands in the Alita extension:
  - Alita: SDD: Specify Intent (/specify)
  - Alita: SDD: Technical Planning (/plan)
  - Alita: SDD: Break Down Tasks (/tasks)
  - Alita: SDD: View Current State
- Use MCP tools (from chat/CLI) to emulate reasoning:
  - mangle_spec_reason
  - mangle_plan_validate
  - mangle_task_optimize
  - mangle_cross_phase_verify
  - mangle_living_doc_update
- Add your MANGLE CLI path and wire invocation when ready:
  - Prefer `src/core/proc.py` to execute binaries safely (no shell=True).
  - Gate with an env var (e.g., `MANGLE_PATH`) and provide graceful fallback.

CI Validation
- `.github/workflows/validate-config.yml` ensures `.github` configs avoid hardcoded secrets and that YAML files are syntactically valid.

Future Enhancements
- Implement real fact extraction from specs/plans/tasks and invoke MANGLE in the MCP tools.
- Add an orchestrator to run cross-phase checks on repo events.
- Extend Copilot instructions with exact prompts for reasoning passes.
