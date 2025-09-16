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
