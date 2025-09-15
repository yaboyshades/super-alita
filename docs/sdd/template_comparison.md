# SDD Template Harmonization Audit

This note compares the long-standing Spec-Driven Development (SDD) templates under `templates/sdd/` with the newer minimal templates stored at the root of `templates/`. It captures structural differences, highlights downstream dependencies, and recommends next steps for converging on a single canonical set.

## Template Sets

- **Canonical SDD templates** – `templates/sdd/spec-template.md`, `templates/sdd/plan-template.md`, and `templates/sdd/tasks-template.md` power the `/specify`, `/plan`, and `/tasks` phases in the runtime pipeline.【F:templates/sdd/spec-template.md†L1-L190】【F:templates/sdd/plan-template.md†L1-L342】【F:templates/sdd/tasks-template.md†L1-L398】
- **Minimal templates** – `templates/spec-template.md` and `templates/plan-template.md` live at the root and are consumed by helper scripts/CLI tooling outside of the strict SDD workflow.【F:templates/spec-template.md†L1-L182】【F:templates/plan-template.md†L1-L270】

## Specification Template Comparison

| Aspect | `templates/sdd/spec-template.md` | `templates/spec-template.md` | Impact |
| --- | --- | --- | --- |
| Metadata fields | Uses feature identifiers and compliance placeholders (`{feature_id}`, score stub) that line up with SDD pipeline IDs.【F:templates/sdd/spec-template.md†L3-L16】 | Uses mustache-style metadata (`{{feature_name}}`, `{{creation_date}}`) and omits compliance score slot.【F:templates/spec-template.md†L3-L16】 | Switching between sets requires adapting templating logic (string formatting vs. mustache) before reuse. |
| Requirements framing | Enforces a “Problem Statement” header and three structured user stories with per-story acceptance checklists.【F:templates/sdd/spec-template.md†L10-L54】 | Provides a single “Primary User Story” block and a free-form secondary section.【F:templates/spec-template.md†L18-L31】 | Minimal template yields lighter scaffolding but removes explicit requirement for three stories, weakening SDD validation rules. |
| Functional section style | Captures `Given/When/Then` scaffolding and separates business rules from functional requirements.【F:templates/sdd/spec-template.md†L61-L80】 | Focuses on enumerated requirements plus API requirements list.【F:templates/spec-template.md†L33-L48】 | Merging would need reconciliation between scenario-driven vs. API-oriented guidance. |
| Constitutional coverage | Aligns strictly to the six constitutional articles with general quality checks.【F:templates/sdd/spec-template.md†L129-L170】 | Includes Article VIII (Anti-Abstraction) and Article VI reframed as “Implicit Knowledge Codification,” plus readiness gates and constitutional architect approval steps.【F:templates/spec-template.md†L96-L176】 | Direct replacement would break current scoring logic that assumes six-article structure; the extra gates would need new validation rules. |
| Lifecycle guidance | Ends with revision history and “Next Steps” that point users back into `/plan`.【F:templates/sdd/spec-template.md†L173-L190】 | Adds an “Implementation Readiness” checklist and constitutional authority metadata.【F:templates/spec-template.md†L164-L182】 | Keeping both may be warranted: SDD for automated scoring, minimal template for human-driven readiness reviews. |

## Implementation Plan Template Comparison

| Aspect | `templates/sdd/plan-template.md` | `templates/plan-template.md` | Impact |
| --- | --- | --- | --- |
| Metadata & scoring | Mirrors spec template metadata with compliance score placeholder and spec reference.【F:templates/sdd/plan-template.md†L3-L15】 | Uses mustache variables without compliance score fields.【F:templates/plan-template.md†L3-L16】 | Again, formatting differences prevent drop-in substitution. |
| Architecture framing | Provides component breakdown, data flow, and explicit library-first dependency analysis.【F:templates/sdd/plan-template.md†L19-L118】 | Emphasizes project count, anti-abstraction checklist, and CLI distribution mandates.【F:templates/plan-template.md†L18-L188】 | Minimal template pushes constitutional Article VII/VIII enforcement; SDD template favors broader architecture documentation. |
| Execution phases | Summarizes four phases with duration placeholders plus explicit test-first compliance checklist.【F:templates/sdd/plan-template.md†L69-L100】 | Details test-first, CLI, and integration phases with explicit task lists for each (Red/Green/Refactor, CLI work, etc.).【F:templates/plan-template.md†L49-L171】 | Consolidation requires deciding whether SDD should inherit CLI-specific mandates or keep broader phase buckets. |
| Quality & deployment | Adds Quality Assurance, Deployment, Documentation plans, and per-article scorecards.【F:templates/sdd/plan-template.md†L216-L345】 | Tracks constitutional verification via Articles I–VIII and defines success metrics plus task generation commands.【F:templates/plan-template.md†L201-L264】 | The scorecard structure differs; merging demands new schema for compliance evidence vs. binary checklists. |

## Task Template Coverage

Only the SDD tree ships a tasks template; no minimal counterpart exists under the root `templates/` directory (only spec/plan are present there).【F:templates/sdd/tasks-template.md†L1-L398】【a1c8ed†L1-L2】 Meanwhile, CLI tooling like `detect_ai_cli.py` still references `templates/tasks-template.md`, so the fallback path would fail if Copilot/OpenAI output is unavailable.【F:detect_ai_cli.py†L160-L219】 Creating a minimal tasks template (or pointing the CLI to `templates/sdd/tasks-template.md`) remains an outstanding gap.

## Downstream Usage Map

| Consumer | Template path(s) | Notes |
| --- | --- | --- |
| `src/sdd` runtime config | `templates/sdd/spec-template.md`, `plan-template.md`, `tasks-template.md` via `templates_dir="templates/sdd"` | Canonical for streaming pipeline and constitutional scoring.【F:src/sdd/config.py†L25-L110】 |
| `extensions/alita-language-tools` scripts | Copy `templates/spec-template.md` and `templates/plan-template.md` into new feature branches | Keeps CLI/bootstrap flows aligned with minimal templates.【F:extensions/alita-language-tools/scripts/create-new-feature.sh†L70-L96】【F:extensions/alita-language-tools/scripts/setup-plan.sh†L1-L44】 |
| `detect_ai_cli.py` helper | Requests `spec-template.md`, `plan-template.md`, and `tasks-template.md` from root `templates/` | Needs consistent naming or fallback to the SDD directory for tasks.【F:detect_ai_cli.py†L160-L259】 |
| Prompt configuration | `src/config/prompts/integrated_system_prompts.json` references `templates/sdd/specification.md`/`plan.md`/`tasks.md` | These filenames no longer exist, indicating another cleanup opportunity when templates are consolidated.【F:src/config/prompts/integrated_system_prompts.json†L31-L61】 |

## Recommendations

1. **Keep SDD templates as the authoritative set for automated workflows** until scoring logic can ingest Article VIII and readiness gates; the structural assumptions in `src/sdd` depend on the current six-article layout.【F:src/sdd/config.py†L25-L110】
2. **Move or rename the minimal templates into an explicit `templates/minimal/` namespace** and update CLI scripts accordingly, clarifying that they are lightweight human-facing alternatives rather than the canonical SDD artifacts.【F:extensions/alita-language-tools/scripts/create-new-feature.sh†L70-L96】【F:extensions/alita-language-tools/scripts/setup-plan.sh†L1-L44】
3. **Provide a minimal tasks template (or retarget consumers to the SDD version)** so that helper tooling no longer references a non-existent file.【F:detect_ai_cli.py†L202-L259】【a1c8ed†L1-L2】
4. **Audit prompt/config references** such as `integrated_system_prompts.json` to point at actual filenames once the directory structure is finalized, preventing runtime attempts to load missing templates.【F:src/config/prompts/integrated_system_prompts.json†L31-L61】

These steps let us avoid breaking the existing SDD constitutional checks while making it clear where the new minimal variants live and which downstream integrations must change when we consolidate the template hierarchy.
