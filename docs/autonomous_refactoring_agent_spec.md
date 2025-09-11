# Autonomous Refactoring Agent Kit v1.0 Specification

**Constitutional Article**: VIII (Automation of Expertise)  
**Creation Date**: September 8, 2025  
**Status**: Draft  
**Version**: 1.0.0

## Overview

The Autonomous Refactoring Agent (ARA) automates code quality improvements by identifying technical debt hotspots, suggesting constitutional patterns, and applying refactors with human approval. It codifies expertise in architectural pattern application and debt remediation.

## Problem Statement

Super-Alita's codebase accumulates technical debt through rapid iteration. Manual refactoring is time-intensive and inconsistent. The ARA provides automated identification and resolution of:

- Cyclomatic complexity hotspots  
- Duplicate code patterns
- Constitutional pattern violations
- Performance anti-patterns
- Security vulnerability patterns

## User Stories

### Primary Stories

**Story 1: Debt Hotspot Detection**
As a developer, I want the ARA to scan my codebase and identify the highest-impact technical debt areas so I can prioritize refactoring efforts effectively.

**Story 2: Pattern-Based Refactoring**  
As a developer, I want the ARA to suggest constitutional pattern applications (Plugin inheritance, event-driven architecture, sandbox execution) with code examples and diffs.

**Story 3: Automated Application**
As a developer, I want the ARA to apply approved refactors automatically while maintaining all tests and preserving functionality.

### Secondary Stories

**Story 4: Constitutional Compliance**
As a team lead, I want the ARA to ensure all refactors follow constitutional articles and maintain architectural consistency.

**Story 5: Multi-Step Workflows**
As a developer, I want the ARA to handle complex refactors that require multiple coordinated changes across files.

## Functional Requirements

### Core Capabilities

1. **Hotspot Analysis**
   - Static analysis using AST parsing
   - Cyclomatic complexity calculation
   - Duplicate code detection
   - Import dependency analysis
   - Constitutional pattern compliance checking

2. **Pattern Recognition**
   - Plugin interface violations
   - Missing event bus subscriptions
   - Unsafe eval/exec usage
   - Missing sandbox wrapping
   - Test coverage gaps

3. **Refactor Generation**  
   - Constitutional pattern application
   - Code extraction and modularization
   - Type hint addition and improvement
   - Documentation generation
   - Test case generation

4. **Safe Application**
   - Sandbox execution of changes
   - Incremental application with rollback
   - Test suite validation after each change
   - Git commit with descriptive messages

### Inputs

- **Source Directory**: Path to codebase for analysis
- **Debt Report**: Optional JSON report from previous scans  
- **Target Patterns**: Constitutional patterns to apply
- **Approval Mode**: Interactive/batch/auto approval levels

### Outputs

- **Debt Hotspot Report**: JSON/YAML with prioritized refactor opportunities
- **Refactor Suggestions**: Structured diffs and rationale
- **Applied Changes**: Git commits with constitutional compliance tags

## Technical Architecture

### Class Structure

`python
@dataclass
class RefactorOpportunity:
    file_path: str
    issue_type: str  # "complexity", "duplication", "pattern_violation"
    severity: float  # 0.0-1.0
    description: str
    suggested_pattern: str
    estimated_effort: str  # "low", "medium", "high"

@dataclass  
class RefactorPlan:
    opportunities: List[RefactorOpportunity]
    execution_order: List[str]  # file paths in dependency order
    rollback_strategy: str
    test_requirements: List[str]

class CodeAnalyzer:
    """Analyzes codebase for refactoring opportunities"""
    def scan_directory(self, path: Path) -> List[RefactorOpportunity]
    def calculate_complexity(self, file_path: Path) -> float
    def detect_duplicates(self, files: List[Path]) -> List[RefactorOpportunity]
    def check_constitutional_compliance(self, file_path: Path) -> List[RefactorOpportunity]

class PatternApplicator:
    """Applies constitutional patterns to code"""
    def apply_plugin_inheritance(self, target_class: str, file_path: Path) -> str
    def add_event_bus_integration(self, class_def: str) -> str  
    def wrap_with_sandbox(self, unsafe_code: str) -> str
    def generate_type_hints(self, function_def: str) -> str

class RefactorExecutor:
    """Safely executes refactoring plans"""
    def execute_plan(self, plan: RefactorPlan, approval_callback: Callable) -> bool
    def apply_single_refactor(self, opportunity: RefactorOpportunity) -> bool
    def validate_changes(self, modified_files: List[Path]) -> bool
    def rollback_changes(self, commit_hash: str) -> bool

class AutonomousRefactoringAgent:
    """Main orchestrator for automated refactoring"""
    def __init__(self, project_path: Path, approval_mode: str = "interactive")
    def analyze_project(self) -> RefactorPlan
    def suggest_improvements(self, plan: RefactorPlan) -> List[str]
    def execute_refactors(self, plan: RefactorPlan) -> bool
`

### CLI Interface

`ash
# Hotspot analysis
python tools/refactor_hotspots.py --scan /path/to/project --output debt_report.json

# Pattern application
python tools/refactor_hotspots.py --apply-patterns --report debt_report.json --approval interactive

# Specific pattern focus
python tools/refactor_hotspots.py --pattern plugin_inheritance --target src/abilities/

# Batch mode
python tools/refactor_hotspots.py --report debt_report.json --approval batch --max-changes 5
`

### Launcher Integration

`ash
# Via unified launcher
python start.py --mode refactor-agent --report debt_report.json
python start.py --mode refactor-agent --scan-only --output reports/

# Environment variables
REFACTOR_AGENT_ENABLED=true
SANDBOX_MODE=enabled  
APPROVAL_REQUIRED=true
`

## Constitutional Compliance

### Article I (Library-First)
- Each analyzer is a reusable component
- Pattern applicators work independently
- CLI and library interfaces provided

### Article II (Test-First)
- Test suite must fail initially
- All refactor operations preserve test coverage
- New functionality requires new tests

### Article III (Simplicity Gate)
- Minimal complexity with clear justification
- Single responsibility per analyzer
- Explicit error handling

### Article VIII (Automation of Expertise)
- Codifies architectural pattern knowledge
- Automates repetitive refactoring tasks
- Preserves human oversight through approval workflows

## Quality Attributes

### Performance
- Analysis completes within 30 seconds for 1000+ file projects
- Incremental scanning for changed files only
- Parallel processing for independent refactors

### Reliability  
- Zero data loss through git-based rollback
- Test suite validation before commit
- Sandbox execution prevents system corruption

### Usability
- Clear progress indicators and ETA
- Diff previews for all changes
- Interactive approval with rationale display

## Acceptance Criteria

### Given/When/Then Scenarios

**Scenario 1: Hotspot Detection**
- **Given** a Python project with cyclomatic complexity > 10 in 3 files
- **When** I run python tools/refactor_hotspots.py --scan /project/path
- **Then** I receive a JSON report with 3 complexity hotspots ranked by severity

**Scenario 2: Plugin Pattern Application**
- **Given** a class that doesn't inherit from PluginInterface but should
- **When** I run the refactor agent with --pattern plugin_inheritance
- **Then** The class is refactored to inherit from PluginInterface with proper initialization

**Scenario 3: Safe Rollback**
- **Given** a refactor that breaks the test suite
- **When** the agent detects test failures
- **Then** All changes are automatically rolled back and the failure is reported

## Implementation Priority

1. **Phase 1**: CodeAnalyzer with complexity detection
2. **Phase 2**: PatternApplicator for plugin inheritance
3. **Phase 3**: RefactorExecutor with sandbox safety
4. **Phase 4**: CLI interface and launcher integration
5. **Phase 5**: Advanced patterns (event bus, type hints)

## Dependencies

- **AST Parsing**: Python st module
- **Git Operations**: GitPython library  
- **Code Analysis**: mccabe for complexity, adon for metrics
- **Testing**: Integration with existing pytest suite
- **Sandbox**: Existing src/sandbox/exec_sandbox.py

## Success Metrics

- 90%+ reduction in manual refactoring time
- Zero production bugs from automated refactors  
- 95%+ developer approval rating for suggestions
- Constitutional compliance score > 0.9 for all outputs
