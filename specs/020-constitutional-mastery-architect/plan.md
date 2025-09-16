# Implementation Plan: Constitutional Mastery Architect v5.3

**Branch**: `020---json` | **Date**: 2025-09-16 | **Spec**: specs/020-constitutional-mastery-architect/feature-spec.md
**Input**: Feature specification from `specs/020-constitutional-mastery-architect/feature-spec.md`

## Execution Flow (/plan command scope)

1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command

---

## Summary

Primary requirement: create a script-driven SDD pipeline and a Constitutional Mastery Architect specification-first workflow that can generate spec → plan → tasks artifacts. The plan focuses on producing constitutionally-compliant artifacts (research.md, data-model.md, contracts/, quickstart.md) and preparing for task generation.

Technical approach: Use repository templates in `.specify/templates/` to generate required artifacts. Enforce constitution checks before design; prioritize test-first development, library-first structure, and integration-first testing.

---

## Technical Context

**Language/Version**: NEEDS CLARIFICATION (the blueprint is language-agnostic; default to Python 3.11 for tooling scripts)
**Primary Dependencies**: NEEDS CLARIFICATION (tools: scripts rely on Python, jq; generated libraries should specify dependencies per plan)
**Storage**: N/A (primarily spec and code artifacts)
**Testing**: pytest (default) unless plan indicates otherwise
**Target Platform**: Linux-compatible developer environments and CI; Windows-compatible scripts provided
**Project Type**: Single project with `src/`, `tests/` layout by default
**Performance Goals**: NEEDS CLARIFICATION
**Constraints**: Must pass Constitutional checks; follow TDD and integration-first testing
**Scale/Scope**: Project-level tooling and templates; expected to support multiple feature specs

---

## Constitution Check

> GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.

**Simplicity**:

- Projects: 1 (tooling + templates) — PASS
- Using framework directly? Yes — PASS
- Single data model? n/a — PASS
- Avoiding patterns? Keep minimal — PASS

**Architecture**:

- EVERY feature as library? Yes — plan mandates library-first
- Libraries listed: `sdd_core` (spec generation), `sdd_templates` (templates), `sdd_tools` (scripts)
- CLI per library: planned — `sdd_core` will expose `specify`, `plan`, `tasks` commands
- Library docs: llms.txt planned — NEEDS CLARIFICATION on format

**Testing (NON-NEGOTIABLE)**:

- RED-GREEN-Refactor enforced — PASS (plan requires tests before implementation)
- Git commits show tests before implementation — PASS (workflow prescribes this)
- Order: Contract→Integration→E2E→Unit enforced — PASS
- Real dependencies used? Where possible — NEEDS CLARIFICATION for CI

**Observability**:

- Structured logging included in generated tooling — PASS (to be implemented)

**Versioning**:

- Versioning strategy: Semantic SemVer for libraries — PASS (details to be documented)

---

## Project Structure

(specs/020-constitutional-mastery-architect/)
- plan.md
- research.md
- data-model.md
- quickstart.md
- contracts/
- tasks.md (created by /tasks)

Source layout: default to single project `src/`, `tests/`.

---

## Phase 0: Outline & Research

Unknowns to resolve (from feature spec):
- Preferred language/runtime for generated libraries and tooling (defaults to Python 3.11) — [NEEDS CLARIFICATION]
- CI environment expectations (Linux vs Windows runners) — [NEEDS CLARIFICATION]
- Persistence needs for ArchitecturalDecisionRegistry (file-based YAML vs DB) — [NEEDS CLARIFICATION]
- Preferred LLM/assistant integrations to include by default (CLAUDE/GEMINI/COPILOT) — [NEEDS CLARIFICATION]

Research tasks:
- Research: Best practices for script-driven SDD tooling in Python 3.11
- Research: How to implement bidirectional spec-code sync (patch generation)
- Research: Recommended formats for ArchitecturalDecisionRegistry (YAML/JSON)
- Research: CI patterns for enforcing constitutional checks

Create `research.md` consolidating decisions and rationale.

---

## Phase 1: Design & Contracts

Data model (high level):
- FeatureSpec: { title, branch, created_date, status, input_description, sections, needs_clarification }
- DecisionRegistryEntry: { id, rule, rationale, contributors, date }

Contracts:
- CLI contracts: `sdd-core` CLI exposing `specify`, `plan`, `tasks` commands with JSON input/output
- REST/HTTP endpoints: None by default; optional for remote orchestration

Contract tests (to be created later):
- Contract tests for CLI JSON output formats (spec creation, plan generation)

Quickstart:
- Steps to run locally:
  1. Ensure Python 3.11 and jq are installed
  2. Activate virtualenv
  3. Run `python -m src.sdd.cli specify "Constitutional Mastery Architect v5.3"`
  4. Inspect `specs/020-constitutional-mastery-architect/feature-spec.md`

---

## Phase 2: Task Planning Approach

Task generation approach: Use `templates/tasks-template.md`, generate tasks per contract and entities, mark [P] where parallelizable, follow TDD ordering.

---

## Complexity Tracking

No constitution violations identified that require exceptions. Any future violations must be documented in `Complexity Tracking` in this plan.

---

## Progress Tracking

- [x] Phase 0: Research complete (draft)
- [x] Phase 1: Design complete (draft)
- [ ] Phase 2: Task planning described
- [ ] Phase 3: Tasks generated (/tasks)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

---

## Outputs generated

- specs/020-constitutional-mastery-architect/plan.md
- specs/020-constitutional-mastery-architect/research.md (next)
- specs/020-constitutional-mastery-architect/data-model.md (next)
- specs/020-constitutional-mastery-architect/quickstart.md (next)
- specs/020-constitutional-mastery-architect/contracts/ (next)

