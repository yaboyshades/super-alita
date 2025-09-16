# research.md - Constitutional Mastery Architect v5.3

**Spec**: specs/020-constitutional-mastery-architect/feature-spec.md

**Generated**: 2025-09-16

## Purpose

Resolve NEEDS_CLARIFICATION items from the feature spec and provide recommendations for the implementation plan.

## Resolved Decisions

1. Preferred language/runtime for tooling and generated libraries

   - Decision: Default to Python 3.11 for tooling scripts and CLI. Reason: existing repo uses Python tooling; wide ecosystem and developer familiarity.
   - Alternatives considered: Node.js (TypeScript) for CLI; Rust for performance-critical components.
   - Rationale: Python allows rapid iteration and easy integration with existing scripts in the repo.

2. CI environment expectations

   - Decision: Support Linux runners as primary CI, with Windows runner checks for script compatibility.
   - Rationale: Most CI providers use Linux runners; Windows-specific scripts will be validated in separate jobs.

3. Persistence for ArchitecturalDecisionRegistry

   - Decision: Use YAML file-based registry stored in `memory/ArchitecturalDecisionRegistry.yaml` with optional export to JSON for tooling consumption.
   - Rationale: File-based YAML is simple, human-editable, supports versioning, and matches the repo patterns.

4. LLM/assistant integrations to include by default

   - Decision: Include adapters for Gemini, Copilot (GitHub), and an abstract adapter for Claude; make adapters pluggable.
   - Rationale: Supports multiple provider ecosystems and allows selection per deployment.

## Research Notes & Links

- Best practices for script-driven SDD tooling in Python: prefer small CLI utilities, use click/typer for CLI, pytest for tests, and dot-env or env vars for secrets.

- Bidirectional spec-code sync approaches: generate unified patch files (.patch) with provenance metadata; use git diffs and structured JSON mapping between spec sections and code modules.

- ArchitecturalDecisionRegistry formats: YAML schema with fields: id, spec_ref, decision, rationale, alternatives, date, contributors.

- CI patterns: add a 'constitution-check' job to CI that runs lightweight linting, constitutional gates, and spec validation before merging.

## Next Steps

- Create `data-model.md` mapping `FeatureSpec` and `DecisionRegistryEntry` to concrete attributes and example YAML snippets.

- Create `/contracts` placeholders for CLI contract tests.

- Create `quickstart.md` with exact commands and environment setup.

