# SDD (Spec-Driven Development) Documentation

> **Master Blueprint:** The complete Constitutional Mastery Architect v5.3 instructions live in [`constitutional_mastery_architect_v5_3.md`](constitutional_mastery_architect_v5_3.md). Use it as the authoritative reference for persona, workflow, and templates.


## Overview

Spec-Driven Development (SDD) is a constitutional development methodology that integrates the six constitutional articles into every phase of software development. The SDD workflow transforms specifications directly into executable implementations through constitutional validation gates.

## Core Workflow

### 1. Specify Phase (`/specify`)
**Purpose**: Define requirements with constitutional compliance validation.

**Constitutional Integration**:
- **Article I (Library-First)**: Research existing solutions before defining new requirements
- **Article II (Test-First)**: Include testability requirements in specifications
- **Article III (Simplicity)**: Define simplicity constraints and acceptance criteria
- **Article IV (Integration-First)**: Plan end-to-end workflows and integration points
- **Article V (Clarity)**: Use unambiguous language and clear acceptance criteria
- **Article VI (Counterfactual)**: Document alternatives considered and decisions made

**Template**: `templates/sdd/spec-template.md`

**Validation**: Constitutional compliance threshold: 0.75

**Output**: Constitutional specification with user stories, acceptance criteria, and decision rationale

### 2. Plan Phase (`/plan`)
**Purpose**: Design implementation approach with constitutional validation.

**Constitutional Integration**:
- **Article I (Library-First)**: Evaluate and justify library choices over custom implementations
- **Article II (Test-First)**: Allocate development time for test design and implementation
- **Article III (Simplicity)**: Break complex features into simple, composable components
- **Article IV (Integration-First)**: Include integration testing phases in timeline
- **Article V (Clarity)**: Define clear architecture and technical decisions
- **Article VI (Counterfactual)**: Document architectural alternatives and trade-offs

**Template**: `templates/sdd/plan-template.md`

**Validation**: Constitutional compliance threshold: 0.75

**Output**: Constitutional implementation plan with architecture, dependencies, and testing strategy

### 3. Tasks Phase (`/tasks`)
**Purpose**: Break down implementation into constitutional task units.

**Constitutional Integration**:
- **Article I (Library-First)**: Prioritize integration over ground-up development tasks
- **Article II (Test-First)**: Include test creation as prerequisites for implementation tasks
- **Article III (Simplicity)**: Validate task complexity metrics (functions <50 lines, <10 cyclomatic complexity)
- **Article IV (Integration-First)**: Schedule integration testing tasks with higher priority
- **Article V (Clarity)**: Define clear task acceptance criteria and definitions of done
- **Article VI (Counterfactual)**: Include decision validation and alternative evaluation tasks

**Template**: `templates/sdd/tasks-template.md`

**Validation**: Constitutional compliance threshold: 0.75

**Output**: Constitutional task breakdown with dependencies, estimates, and validation criteria

## API Endpoints

### SDD Workflow Endpoints
- `POST /sdd/specify` - Execute specification phase with constitutional validation
- `POST /sdd/plan` - Execute planning phase with constitutional validation
- `POST /sdd/tasks` - Execute task breakdown phase with constitutional validation
- `GET /sdd/status` - Get current SDD workflow state and compliance scores

### Integration Endpoints
- `POST /tools/sdd_specify` - Tool-based specification execution
- `POST /tools/sdd_plan` - Tool-based planning execution
- `POST /tools/sdd_tasks` - Tool-based task breakdown execution

## VS Code Integration

### Commands
- `alita.sdd.specify` - Execute SDD specification phase
- `alita.sdd.plan` - Execute SDD planning phase
- `alita.sdd.tasks` - Execute SDD task breakdown phase
- `alita.sdd.viewState` - View current SDD workflow state

### Configuration
SDD workflow configuration is managed in `src/sdd/config.py`:

```python
@dataclass
class SDDConfig:
    """SDD workflow configuration."""

    commands: dict[str, str]  # SDD command definitions
    templates: dict[str, str]  # Template file paths
    validation: dict[str, Any]  # Validation rules and thresholds
    constitutional_gates: list[str]  # Constitutional validation phases
    mangle_integration: dict[str, Any]  # Mangle engine integration
```

## Constitutional Validation

### Validation Framework
Each SDD phase undergoes constitutional validation against all six articles:

1. **Library-First Development**: Validates research of existing solutions
2. **Test-First Development**: Validates test strategy and coverage planning
3. **Simplicity Gate**: Validates complexity constraints and simple design
4. **Integration-First Testing**: Validates end-to-end testing prioritization
5. **Clarity and Unambiguity**: Validates clear requirements and decisions
6. **Counterfactual Justification**: Validates decision rationale and alternatives

### Validation Process
1. **Content Analysis**: Regex-based pattern matching for constitutional indicators
2. **Scoring**: Weighted scoring across constitutional articles (target: 35-42 points)
3. **Threshold Validation**: Minimum 0.75 compliance score required
4. **Feedback**: Detailed compliance report with improvement recommendations
5. **Gate Control**: Prevents progression to next phase until compliance achieved

### Violation Response Protocol
1. **Detection**: Living Document Oracle identifies constitutional violations
2. **Assessment**: Enhanced Consensus evaluates severity and impact
3. **Recommendation**: Auto-Reasoning Stack Generator suggests corrections
4. **Implementation**: APE Engine optimizes corrective prompts
5. **Validation**: Cross-Project Reasoning Miner confirms resolution

## Spec-Code Integrity Mini-Protocols

Article XI enforcement now relies on paired mini-protocols that coordinate specification changes with downstream code updates. These templates live under `templates/sync-protocols/` and are invoked immediately after `/specify` completes and again before merge readiness reviews:

- [`sync_spec.md`](../../templates/sync-protocols/sync_spec.md) — Drives the spec-side loop (Event → Atom/Bond → Energy → TODO → Bandit → Reward) so revised requirements update the knowledge graph, TODO ledger, and telemetry before implementation starts.
- [`sync_code.md`](../../templates/sync-protocols/sync_code.md) — Governs the implementation loop, binding commits, tests, and observability artifacts back to the spec hash produced by the first protocol.

Both protocols emit deterministic telemetry (`STATE_TRANSITION`, `Ability*`, `Task*`) and require Socratic readiness ≥ 0.75 before code can ship. Integrating them into SDD automation prevents drift and keeps PR templates aligned with the latest constitutional guidance.

## Templates

### Specification Template (`templates/sdd/spec-template.md`)
```markdown
# [Feature Name] Specification

## Constitutional Framework Compliance
- [ ] Article I: Library-First Development - Research existing solutions
- [ ] Article II: Test-First Development - Define testability requirements
- [ ] Article III: Simplicity Gate - Keep requirements focused and simple
- [ ] Article IV: Integration-First Testing - Plan end-to-end workflows
- [ ] Article V: Clarity and Unambiguity - Use clear, unambiguous language
- [ ] Article VI: Counterfactual Justification - Document alternatives considered

## User Stories
[User stories in "As a... I want... So that..." format]

## Acceptance Criteria
[Given/When/Then scenarios]

## Decision Rationale
[Alternatives considered and justification for choices]
```

### Plan Template (`templates/sdd/plan-template.md`)
```markdown
# [Feature Name] Implementation Plan

## Constitutional Framework Compliance
- [ ] Article I: Library-First Development - Evaluate library options
- [ ] Article II: Test-First Development - Allocate test design time
- [ ] Article III: Simplicity Gate - Break into simple components
- [ ] Article IV: Integration-First Testing - Include integration phases
- [ ] Article V: Clarity and Unambiguity - Define clear architecture
- [ ] Article VI: Counterfactual Justification - Document architectural alternatives

## Architecture
[High-level design and component structure]

## Dependencies
[Library choices with justification]

## Testing Strategy
[Test-first approach and coverage plans]

## Implementation Phases
[Breakdown with constitutional validation gates]
```

### Tasks Template (`templates/sdd/tasks-template.md`)
```markdown
# [Feature Name] Task Breakdown

## Constitutional Framework Compliance
- [ ] Article I: Library-First Development - Prioritize integration tasks
- [ ] Article II: Test-First Development - Test creation prerequisites
- [ ] Article III: Simplicity Gate - Validate complexity metrics
- [ ] Article IV: Integration-First Testing - High-priority integration tasks
- [ ] Article V: Clarity and Unambiguity - Clear acceptance criteria
- [ ] Article VI: Counterfactual Justification - Decision validation tasks

## Task List
[Numbered tasks with descriptions, dependencies, and estimates]

## Constitutional Validation Tasks
[Specific tasks for constitutional compliance verification]

## Integration Testing Priority
[High-priority integration and end-to-end testing tasks]
```

## Mangle Integration

SDD integrates with the Mangle reasoning engine for:

- **Specification Enhancement**: Semantic analysis of requirements
- **Plan Optimization**: Architecture and dependency optimization
- **Task Sequencing**: Optimal task ordering and dependency resolution
- **Constitutional Validation**: Deep reasoning about constitutional compliance
- **Decision Support**: Alternative evaluation and recommendation

## Files and Structure

### Core Files
- `src/sdd/config.py` - SDD workflow configuration
- `src/sdd/router.py` - SDD API endpoints and workflow execution
- `src/sdd/validators.py` - Constitutional validation utilities (planned)

### Templates
- `templates/sdd/spec-template.md` - Specification phase template (legacy path `specification.md` retired)
- `templates/sdd/plan-template.md` - Planning phase template (legacy path `plan.md` retired)
- `templates/sdd/tasks-template.md` - Task breakdown phase template (legacy path `tasks.md` retired)

### Memory/Documentation
- `memory/sdd/constitutional_sdd_framework.md` - Constitutional integration document
- `docs/sdd/README.md` - This documentation file

### Integration Points
- `src/main.py` - SDD router registration
- `src/orchestration/unified_orchestrator.py` - SDD config integration
- `.github/copilot-instructions.md` - Agent mode SDD instructions

## Usage Examples

### Command Line
```bash
# Execute SDD workflow phases
curl -X POST http://localhost:8080/sdd/specify -d '{"content": "Build user authentication"}'
curl -X POST http://localhost:8080/sdd/plan -d '{"spec_content": "..."}'
curl -X POST http://localhost:8080/sdd/tasks -d '{"plan_content": "..."}'

# Enable canonical telemetry during local runs
export CANONICAL_EVENTS_ENABLED=true
python -m src.main --prompt "Summarize aggregator rollout"
```

### Agent Mode
```
/specify user authentication with OAuth integration
/plan the authentication implementation approach
/tasks break down authentication into implementable units
```

### Integration with Orchestrator
```python
from src.orchestration.unified_orchestrator import UnifiedOrchestrator
from src.sdd.config import SDDConfig

config = UnifiedRunConfig(
    sdd_enabled=True,
    sdd_config=SDDConfig(
        commands={"specify": "sdd_specify", "plan": "sdd_plan", "tasks": "sdd_tasks"},
        constitutional_gates=["Specification_Gate", "Planning_Gate", "Implementation_Gate"]
    )
)

orchestrator = UnifiedOrchestrator(config)
async for event in orchestrator.execute_unified_pipeline("Build user auth"):
    print(f"SDD Event: {event}")
```

## Best Practices

### 1. Constitutional Compliance
- Always validate against all six constitutional articles
- Maintain minimum 0.75 compliance score at each phase
- Document decision rationale and alternatives considered
- Prioritize existing solutions over custom implementations

### 2. Workflow Execution
- Execute phases sequentially: specify → plan → tasks → implement
- Use templates as starting points, customize for specific requirements
- Validate constitutional compliance before progressing to next phase
- Maintain traceability from specification through implementation

### 3. Integration with Development Process
- Integrate SDD workflow into CI/CD pipeline
- Use SDD validation in code review process
- Track constitutional compliance metrics over time
- Document SDD decisions in project wiki/documentation

### 4. Tool Integration
- Use Mangle engine for semantic analysis and optimization
- Integrate with VS Code for seamless developer experience
- Leverage orchestrator for automated workflow execution
- Use constitutional validation for quality gates

## Troubleshooting

### Common Issues
1. **Low Constitutional Compliance Score**: Review specification against constitutional articles, add missing elements
2. **SDD Workflow Errors**: Check template syntax and required fields
3. **Integration Issues**: Verify Mangle engine connectivity and configuration
4. **Validation Failures**: Review validation patterns and threshold settings

### Debug Commands
```bash
# Check SDD configuration
curl http://localhost:8080/sdd/status

# Validate specific content
curl -X POST http://localhost:8080/tools/sdd_validate -d '{"content": "...", "phase": "specify"}'

# View constitutional compliance details
curl http://localhost:8080/sdd/compliance -d '{"artifact_id": "..."}'
```

## Research Agent Add-on
- Use `scripts/cma/research_specify.sh "<goal>"` to bootstrap a research-focused feature branch.
- Run `python scripts/cma/research_plan_hook.py specs/<branch>/spec.md` during `/plan` Phase 0 to consolidate evidence into `research.md`.
- Before `/tasks`, execute `python scripts/cma/research_tasks_hook.py specs/<branch>` to surface verification work.
- In VS Code, enable the "Deep Research" participant from `tools/vscode-deep-research` for Copilot-integrated research runs (writes to `specs/research-agent/latest.md`).

