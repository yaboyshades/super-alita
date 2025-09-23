# data-model.md - Constitutional Mastery Architect v5.3

## Entities

### FeatureSpec

- title: string
- branch: string
- created_date: iso8601 date
- status: enum [draft, approved, implemented]
- input_description: text
- sections: map of section_name -> text
- needs_clarification: list of strings (questions or markers)

### DecisionRegistryEntry

- id: string (uuid)
- spec_ref: path to spec file or spec id
- decision: text
- rationale: text
- alternatives: list of text
- contributors: list of strings
- date: iso8601 date

## Example YAML snippets

FeatureSpec example:

```yaml
title: "Constitutional Mastery Architect v5.3"
branch: "020---json"
created_date: "2025-09-16"
status: draft
input_description: |
  Constitutional Mastery Architect v5.3: The Definitive Blueprint (Script-Driven Edition)
sections:
  user_scenarios: |
    A product or engineering lead invokes the Constitutional Mastery Architect to generate a full SDD pipeline.
needs_clarification:
  - "Preferred language/runtime for generated libraries?"
```

DecisionRegistryEntry example:

```yaml
- id: "dec-0001"
  spec_ref: "specs/020-constitutional-mastery-architect/feature-spec.md"
  decision: "Use YAML-based registry"
  rationale: "Human-editable, versioned, easy to integrate"
  alternatives:
    - "Use a small SQLite DB"
  contributors:
    - "alice"
  date: "2025-09-16"
```
### NextStepItem

Represents a single actionable follow-up entry surfaced during `/specify`.

- action: string (imperative statement captured in the spec)
- owner: string (assigned owner or `unassigned` if not yet claimed)
- linked_artifact: string (path or identifier for the supporting evidence)
- gate: enum [`library_first`, `test_first`, `simplicity`, `integration_first`, `clarity`, `counterfactual`]
- status: enum [`pending`, `in_progress`, `complete`]
- rationale: string (context for why the action matters)
- source: enum [`clarification`, `artefact`, `command`, `reminder`]

### NextStepGuidance

Structured metadata persisted alongside the spec (e.g. `specs/<feature>/next_steps.yaml`).

- generated_at: iso8601 timestamp
- feature_id: string (matches spec directory prefix)
- clarifications: list of NextStepItem with `source = clarification`
- artefacts: list of NextStepItem with `source = artefact`
- commands: list of NextStepItem with `source = command`
- constitutional_alignment: list of objects
  - gate: enum (same as `NextStepItem.gate`)
  - summary: string describing how the next steps close the gate
  - evidence: string (pointer to the artefact or decision log)

Example YAML:

```yaml
feature_id: "020"
generated_at: "2025-09-16T10:03:21Z"
clarifications:
  - action: "Confirm which authentication providers must be supported"
    owner: "unassigned"
    linked_artifact: "specs/020-constitutional-mastery-architect/spec.md#clarifications"
    gate: clarity
    status: pending
    rationale: "Spec contains [NEEDS CLARIFICATION: auth method] marker"
    source: clarification
artefacts:
  - action: "Create data model sketch for FeatureSpec entity"
    owner: "platform-architecture"
    linked_artifact: "specs/020-constitutional-mastery-architect/data-model.md"
    gate: simplicity
    status: pending
    rationale: "Research file requested a data-model mapping"
    source: artefact
commands:
  - action: "Run `/plan` once clarifications are resolved and evidence is linked"
    owner: "feature-owner"
    linked_artifact: "specs/020-constitutional-mastery-architect/spec.md"
    gate: test_first
    status: pending
    rationale: "Test-first gate requires verified acceptance criteria"
    source: command
constitutional_alignment:
  - gate: library_first
    summary: "Clarify reuse opportunities before committing to new tooling"
    evidence: "research.md#resolved-decisions"
```
