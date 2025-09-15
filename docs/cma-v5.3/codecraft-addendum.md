# CMA v5.3.1 — Codecraft Addendum (Generalized, Repo-Agnostic)

Article XV — Coding Definition of Done (DoD)
A change is not done unless all hold:
• Observability
• Structured logs (JSON or logfmt) with consistent context keys (e.g., trace_id, span_id, component, operation, attempt).
• Metrics: counters for request totals & retries; histograms for latency; gauges for saturation.
• Tracing spans around external I/O and critical sections.
• Resilience
• Retries with exponential backoff and jitter; single owner of retries (no layered/double retries).
• Circuit breaker for failing dependencies.
• Bulkhead limits (per dependency or resource class).
• Explicit timeouts: per-attempt and overall wall-clock cap.
• Security
• Egress/SSRF guardrails (deny non-http(s), link-local/private/metadata by default; allow-lists configurable).
• Secrets never logged; retrieved via a secret resolution abstraction; least-privilege by default.
• Pre-dispatch policy check ("can we call this?") that's testable and auditable.
• Contracts
• JSON Schema (or equivalent) validated at registration.
• Backward/forward/full compatibility rules codified and enforced.
• Property-based generators derived from the schema for test inputs.
• Testing
• Unit + property-based + snapshot (for sequences) + chaos (fault injection) where relevant.
• Integration tests use ephemeral ports/resources and deterministic synchronization.
• Mutation testing targets meaningful surfaces (≥80% on critical modules where feasible).
• Docs & Tooling
• Each change includes short "what/why/risk" notes (ChangeLog or PR footer).
• Linting, type checks, and basic security scanning enabled and enforced in CI.

Article XVI — Golden Implementation Patterns (Abstract, Reusable)
• HTTP/Remote Call Pattern
• Use a single retry owner (either the client wrapper or the transport—never both).
• Expose per-attempt timeout and overall cap.
• Map error classes into a minimal, structured error envelope (status, category, message).
• Emit logs/metrics/spans with correlation IDs and retry attempt number.
• Egress Guard Pattern
• Parse target, resolve hostnames, and block private/link-local/loopback/multicast/metadata ranges by default.
• Allow-list via configuration; log a structured "policy_denied" event with reason (no secrets).
• Policy & Secrets Pattern
• Policy: can_call(subject, resource, purpose, context) returns allow/deny + reason.
• Secrets: SecretResolver.get(ref) returns credentials/keys; never serialize to logs/errors.
• Event Emission Pattern
• Emit "CallStarted/CallFinished" (or ToolCall/ToolResult) with consistent fields; guard emissions if bus/tracer is absent (no hard dependency).
• Do not leak payload data by default; include only metadata unless explicitly safe.
• Contract Pattern
• Register schema with validation; cache compiled validators.
• Compatibility checker provides explicit, human-actionable diffs (paths + migration hints).
• Property generators derived from schema for fuzzing + stateful tests.

Article XVII — Code Generation Protocol (for Copilot/LLMs/Tools)
When asking an assistant to write code, prepend this protocol:
CMA v5.3.1 Codecraft Addendum compliance is required. This code must:
• Include structured logging (JSON/logfmt) with trace_id, component, operation context.
• Implement retries with exponential backoff + jitter; circuit breakers; timeouts.
• Add SSRF guardrails for egress; use SecretResolver for credentials (never log secrets).
• Validate all I/O against JSON Schema; emit compatibility diffs.
• Include unit + property + snapshot + chaos tests where applicable.
• Document what/why/risk in code comments or ChangeLog.
If uncertain about any requirement, ask for clarification before proceeding.

Article XVIII — Contract Validation (Schemas & Compatibility)
• Schema Registry
• All API inputs/outputs, configuration, and inter-service messages use registered schemas.
• Schemas include version, description, and examples.
• Backward, forward, and full compatibility rules defined per schema.
• Validation Pipeline
• Input validation at API boundaries with detailed error responses.
• Output validation in development/staging (warning in production).
• Schema evolution tracked with automated compatibility analysis.
• Breaking Change Protocol
• Breaking changes require explicit opt-in (version headers, feature flags).
• Migration path documented with before/after examples.
• Rollback plan defined and tested.

Article XIX — Testing Mandate (Comprehensive Coverage)
• Test Categories (All Required)
• Unit: Test individual functions/methods in isolation.
• Property-based: Generate inputs from schema; test invariants.
• Snapshot: Test sequences/workflows with deterministic outputs.
• Chaos: Inject faults (network, disk, memory) and verify graceful degradation.
• Integration: Test cross-service interactions with ephemeral resources.
• Test Quality Standards
• ≥80% line coverage for critical modules.
• ≥70% branch coverage overall.
• Mutation testing on business logic (≥60% mutation kill rate).
• All tests run in <2 minutes for rapid feedback.
• Test Infrastructure
• Deterministic test data with predictable IDs/timestamps.
• Ephemeral resources (ports, databases, external services) for isolation.
• Parallel execution with proper resource cleanup.

Article XX — Developer Experience (DevEx) Requirements
• Rapid Feedback Loops
• Code changes validated in <30 seconds (lint + quick tests).
• Full CI pipeline completes in <10 minutes.
• Hot reload for local development where applicable.
• Clear Error Messages
• Validation errors include field path, expected vs. actual, fix suggestions.
• Runtime errors include trace ID, component context, next steps.
• Logs structured for easy filtering and correlation.
• Documentation Standards
• API documentation auto-generated from schemas with examples.
• "Getting Started" guide executable in <5 minutes.
• Architecture decisions recorded with rationale and alternatives considered.
• Debugging & Observability
• Structured logs with consistent field names across services.
• Metrics dashboards for SLIs (latency, error rate, saturation).
• Distributed tracing for request flow visualization.
• Local debugging setup documented and automated.