# Research Agent Specification
**Branch**: [feature-branch]
**Created**: [date]
**Status**: Draft

## Objective
- Problem statement: [NEEDS CLARIFICATION]
- Audience: [NEEDS CLARIFICATION]
- Success criteria:
  - Metric 1: [NEEDS CLARIFICATION]
  - Metric 2: [NEEDS CLARIFICATION]

## Queries
- Q1: "[Primary research question]"
- Q2: "[Secondary research question]"
- Q3: "[Optional depth research question]"

> All queries MUST be measurable. Add `[NEEDS CLARIFICATION: …]` when intent is ambiguous.

## Constraints
- Search provider: `${SEARXNG_BASE_URL}`
- Timeout: 10s (per attempt)
- Attempts: 3 (exponential backoff + jitter)
- Minimum authoritative sources per query: 3

## Acceptance Scenarios
1. Given query Q1, when the pipeline runs, then the resulting JSON validates against `contracts/research_query.schema.json` and `stats.count >= 3`.
2. Given a blocked scheme (e.g., `ftp://`), when executed, then the pipeline fails fast with `egress blocked` logged.
3. Given repeated queries, when executed, then latency histogram updates in telemetry.

## Review Checklist
- [ ] Queries map to specification requirements
- [ ] Success criteria measurable and tied to schema fields
- [ ] Edge cases documented (blocked domains, empty results)
- [ ] Collaboration notes captured for follow-up tasks
