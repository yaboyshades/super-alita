<!--
Sync Impact Report - Constitution v1.0.0
========================================
Version Change: INITIAL → 1.0.0 (Initial ratification)
Rationale: MINOR bump - New constitutional framework establishing Super Alita governance principles

Modified Principles:
- NEW: Article I - Library-First Principle
- NEW: Article II - Test-First Imperative  
- NEW: Article III - Simplicity Gate
- NEW: Article IV - Integration-First Testing
- NEW: Article V - Clarity and Unambiguity
- NEW: Article VI - Neural Architecture Preservation
- NEW: Article VII - Event Sourcing Foundation
- NEW: Article VIII - Constitutional Compliance Gates

Added Sections:
- Core Constitutional Articles (8 principles)
- Neural Architecture Requirements
- Development Workflow Standards
- Governance and Amendment Process

Templates Requiring Updates:
✅ .specify/templates/spec-template.md - Added constitutional compliance checklist
✅ .specify/templates/plan-template.md - Added constitutional validation gates
✅ .specify/templates/tasks-template.md - Added compliance verification tasks
✅ .github/copilot-instructions.md - Referenced constitutional framework

Follow-up TODOs:
- None: All critical constitutional principles defined
- Future: Consider adding Article IX for Mangle Integration if code quality rules expand

Commit Message Suggestion:
docs: ratify Super Alita constitution v1.0.0 (establish constitutional AI governance framework)
-->

# Super Alita Constitution
**Specification-Driven Development with Constitutional AI Governance**

## Core Constitutional Articles

### Article I: Library-First Principle

Every feature MUST be designed as a standalone, reusable library with clean API interfaces. Libraries must be self-contained, independently testable, and properly documented with clear purpose statements.

**Non-Negotiable Requirements**:
- No hardcoded application-specific dependencies
- Importable as independent Python modules
- Clean separation of concerns with single responsibility
- Comprehensive docstrings for all public interfaces

**Rationale**: Modular architecture enables composition, testing isolation, and reuse across Super Alita's multi-agent ecosystem. Constitutional compliance threshold: ≥0.75.

### Article II: Test-First Imperative

Testable acceptance criteria MUST be defined before implementation begins. Test scenarios MUST be identified for all user stories with clear success/failure conditions documented. Test data requirements MUST be specified upfront.

**Non-Negotiable Requirements**:
- Red-Green-Refactor cycle strictly enforced
- Minimum 70% test coverage for new code
- Integration tests with real service dependencies preferred over mocks
- Tests execute before implementation approval

**Rationale**: Test-first development prevents architectural degradation and ensures constitutional compliance validation. Constitutional compliance threshold: ≥0.75.

### Article III: Simplicity Gate

Minimal project structure MUST be maintained (≤3 projects per feature). Complexity MUST be justified in writing with architectural decision records. No speculative future-proofing without documented requirements. Simple solutions MUST be chosen over complex alternatives.

**Non-Negotiable Requirements**:
- Function complexity ≤10 (cyclomatic complexity)
- Clear, descriptive naming conventions
- YAGNI principles applied rigorously
- Documented justification for any complexity >10

**Rationale**: Cognitive simplicity reduces maintenance burden and prevents technical debt accumulation. Constitutional compliance threshold: ≥0.75.

### Article IV: Integration-First Testing

Integration tests MUST use real services when practical. Mocks/stubs MUST be minimized to isolation requirements only. End-to-end smoke tests MUST be defined for critical paths. Real Redis-backed event bus MUST be preferred over in-memory alternatives.

**Non-Negotiable Requirements**:
- Contract tests for all new library integrations
- Cross-component communication validated in realistic environments
- Event sourcing patterns tested with actual Redis Streams
- Neural atom bonding validated with real storage backends

**Rationale**: Realistic testing ensures Super Alita's event-driven architecture functions correctly under production conditions. Constitutional compliance threshold: ≥0.75.

### Article V: Clarity and Unambiguity

All TBDs MUST be resolved before implementation begins. Glossary of terms MUST be provided for domain-specific language. Spec-by-example MUST be included for complex behaviors. Edge cases MUST be enumerated with expected outcomes.

**Non-Negotiable Requirements**:
- No placeholder or ambiguous requirements in specifications
- Clear acceptance criteria with testable conditions
- Comprehensive documentation of assumptions
- Examples provided for all non-trivial features

**Rationale**: Unambiguous specifications prevent interpretation errors and ensure constitutional AI agents can generate compliant code. Constitutional compliance threshold: ≥0.75.

### Article VI: Neural Architecture Preservation

Chemistry-inspired neural atom bonding patterns MUST be maintained across all implementations. Deterministic UUIDv5 identifiers MUST be used for atom deduplication. Genealogy tracking with parent_keys and children_keys MUST be preserved. Neural bonding integrity MUST maintain ≥0.85 threshold.

**Non-Negotiable Requirements**:
- Neural atoms inherit from src/core/neural_atom.py patterns
- Event sourcing through src/neural/store.py SQLite backend
- Bond relationships tracked with explicit chemistry metaphors
- Bonding site compatibility validated before connection

**Rationale**: Super Alita's cognitive fabric depends on consistent neural architecture patterns for multi-agent coordination and knowledge graph coherence. Neural integrity threshold: ≥0.85.

### Article VII: Event Sourcing Foundation

All state changes MUST flow through Redis Streams-based event sourcing with CQRS projections. Event bus communication MUST be preferred over direct component calls. Neural Atom Bridge patterns MUST convert EventBus events to cognitive artifacts. Immutable event history MUST be maintained in SQLite.

**Non-Negotiable Requirements**:
- Components communicate via src/core/event_bus.py
- Plugins implement src/core/plugin_interface.py PluginInterface
- Event sanitization through src/orchestration/event_sanitizer.py
- Event → Neural Atom mapping via src/core/neural_atom_bridge.py

**Rationale**: Event-driven architecture enables hot-swappable plugins, telemetry broadcasting, and audit trails for constitutional compliance monitoring. Constitutional compliance threshold: ≥0.75.

### Article VIII: Constitutional Compliance Gates

Three constitutional gates MUST be enforced at development phases: Entry Gate (specification ≥0.75), Process Gate (planning maintains score), Exit Gate (final validation ≥0.75). Mangle rule validation MUST pass with zero violations for circular_dependencies, untested_complex_functions, and hot_paths.

**Non-Negotiable Requirements**:
- Specifications validated before planning phase begins
- Plans checked against constitutional articles before tasking
- Implementation verified against ≥0.75 threshold before merge
- Mangle facts database (.ai/facts.sqlite) queried before structural changes

**Rationale**: Systematic quality gates prevent constitutional drift and ensure Super Alita maintains architectural integrity across all feature development. Compliance monitoring: continuous with ≥0.75 threshold.

## Neural Architecture Requirements

### Chemistry-Inspired Bonding Patterns
- **Covalent Bonds**: Strong parent-child relationships with shared state
- **Ionic Bonds**: Weak references for cross-component coordination
- **Metallic Bonds**: Many-to-many relationships in agent consensus networks

### Deterministic UUID Generation
- Content-based UUIDv5 using ATOM_NS namespace (d6e2a8b1-4c7f-4e0a-8b9c-1d2e3f4a5b6c)
- Same content → same UUID enables deduplication
- Genealogy tracking with depth calculation from parent_keys

### Event → Neural Atom Mapping
- user_message → USER_INTERACTION atom
- tool_created → TOOL_DEFINITION atom
- sot_executed → REASONING_TRACE atom
- state_transition → STATE_TRANSITION atom
- tool_call → TOOL_CALL atom
- tool_response → TOOL_RESPONSE atom

## Development Workflow Standards

### Specification-Driven Development (SDD)
Every feature MUST follow: `/constitution` → `/specify` → `/clarify` → `/plan` → `/implement`

**Method 1: FastAPI Endpoints**
```bash
curl -X POST "http://127.0.0.1:8080/sdd/specify" -H "Content-Type: application/json" -d '{"user_input": "feature description"}'
curl -X POST "http://127.0.0.1:8080/sdd/plan" -H "Content-Type: application/json" -d '{"feature_id": "feat-name"}'
curl -X POST "http://127.0.0.1:8080/sdd/tasks" -H "Content-Type: application/json" -d '{"feature_id": "feat-name"}'
```

**Method 2: Spec-Kit Workflow**
```bash
/constitution  # Initialize constitutional framework
/specify       # Create specification with constitutional validation
/clarify       # Resolve ambiguities interactively
/plan          # Generate implementation plan with gates
/implement     # Generate code with embedded compliance
```

### Security and Execution Boundaries
- All dynamic code execution MUST flow through src/sandbox/exec_sandbox.py
- Subprocess calls MUST use src/core/proc.py (never shell=True)
- YAML operations MUST use src/core/yaml_utils.py (safe loading only)
- Event sanitization MUST scrub sensitive data via src/orchestration/event_sanitizer.py

### Code Quality Standards
- Python 3.11+ with 4-space indentation, double quotes
- Type hints required for all public APIs
- Black formatting at 79 characters (project override, not 88)
- Ruff linting with zero violations
- isort import organization with --profile=black --line-length=79

## Governance and Amendment Process

### Constitutional Supremacy
This constitution supersedes all other development practices, coding standards, and architectural guidelines. Any conflict between constitutional articles and local conventions MUST be resolved in favor of constitutional principles.

### Amendment Procedure
1. **Proposal**: Submit amendment with rationale and impact analysis
2. **Review**: Validate against existing articles for consistency
3. **Version Bump**: Determine MAJOR/MINOR/PATCH according to semantic versioning
4. **Propagation**: Update all dependent templates, scripts, and documentation
5. **Approval**: Merge only after constitutional scorer confirms ≥0.75 compliance
6. **Sync Report**: Generate impact report documenting all changes

### Compliance Monitoring
- **Continuous**: Real-time constitutional scoring during development
- **Pre-commit**: saval command runs full validation pipeline
- **CI/CD**: GitHub Actions enforce ≥0.75 threshold on all PRs
- **Reporting**: Daily compliance reports via src/constitutional/scorer.py

### Violation Response
1. Detection via Living Document Oracle or Mangle rule engine
2. Assessment of severity (critical/moderate/minor)
3. Automated recommendation generation
4. Implementation with constitutional validation
5. Cross-project learning integration

### Version Control
- **Version Format**: MAJOR.MINOR.PATCH (Semantic Versioning 2.0.0)
- **MAJOR**: Backward incompatible governance changes or principle removals
- **MINOR**: New principles added or materially expanded guidance
- **PATCH**: Clarifications, wording fixes, non-semantic refinements

**Version**: 1.0.0 | **Ratified**: 2025-10-06 | **Last Amended**: 2025-10-06