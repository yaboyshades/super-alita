# Super Alita - AI Agent Instructions

You are a **Super Alita Constitutional Engineer** working with a production-grade AI orchestration platform built on event-driven neural architecture, constitutional compliance gates, and specification-driven development.

## �️ Core Architecture (The Big Picture)

### Event-Driven Neural System
Super Alita coordinates through an **event bus** (`src/core/event_bus.py`) with Redis/Memurai backing. All components communicate via events, not direct calls.

**Neural Atoms & Bonds** (`src/core/neural_atom.py`, `src/neural/atom.py`):
- Deterministic UUIDv5 IDs from content hashing - same content = same UUID
- Cognitive artifacts with genealogy tracking (parent_keys, children_keys, depth)
- Bridge pattern connects EventBus → Neural Atoms (`src/core/neural_atom_bridge.py`)
- Store in SQLite via `src/neural/store.py` for event sourcing

**Plugin Architecture** (`src/core/plugin_interface.py`):
- All components inherit `PluginInterface` with `handle_event()` and `shutdown()`
- Hot-swappable, register via `plugin_registry` at startup
- Examples: `src/plugins/compose_plugin.py`, `src/plugins/puter_plugin.py`

### Critical Service Boundaries

```
User Request → FastAPI (src/main.py:app)
            → REUG Router (src/orchestration/unified_orchestrator.py)
            → Plugin System (event_bus broadcasts)
            → Sandbox Execution (src/sandbox/exec_sandbox.py)
            → Neural Store (src/neural/store.py)
            → MCP Integration (src/neural/mcp_server.py)
```

**Entry Points:**
- Runtime: `python -m src.main` (console orchestrator)
- API Server: `uvicorn src.main:app --reload --port 8080` (FastAPI app in src/main.py)
- SDD Endpoints: `POST /sdd/specify`, `/sdd/plan`, `/sdd/tasks` (`src/sdd/router.py`)

## ⚡ Essential Developer Workflows

### Specification-Driven Development (SDD) - Default Workflow
**Every feature starts here.** Do not write code without running SDD first.

#### Core Principle: Specifications Don't Serve Code—Code Serves Specifications
SDD inverts traditional development: specifications are executable contracts that generate architecturally sound code with built-in constitutional compliance.

#### Method 1: FastAPI Endpoints (Python API)

```powershell
# 1. Start the dev server
uvicorn src.main:app --reload --port 8080

# 2. Create specification (with constitutional validation)
curl -X POST "http://127.0.0.1:8080/sdd/specify" `
  -H "Content-Type: application/json" `
  -d '{"user_input": "Add streaming endpoints", "context": {"priority": "high"}}'

# 3. Generate implementation plan (with Mangle integration)
curl -X POST "http://127.0.0.1:8080/sdd/plan" `
  -H "Content-Type: application/json" `
  -d '{"feature_id": "feat-streaming-api"}'

# 4. Break down into tasks
curl -X POST "http://127.0.0.1:8080/sdd/tasks" `
  -H "Content-Type: application/json" `
  -d '{"feature_id": "feat-streaming-api"}'

# CLI alternative:
python -m src.sdd.sdd_cli specify "Add streaming endpoints" --context '{"owner":"platform"}'
python -m src.sdd.sdd_cli plan feat-streaming-api
python -m src.sdd.sdd_cli tasks feat-streaming-api
```

#### Method 2: Spec-Kit Workflow (Integrated SDD with GitHub Copilot)

**Full Constitutional Workflow:**
```bash
# Step 1: Initialize constitutional foundation (one-time setup)
/constitution Create principles for constitutional AI orchestration with neural atom bonding, 
multi-agent coordination, event sourcing, and ≥0.75 compliance gates.

# Step 2: Create neural-aware specification
/specify Neural atom coordination system for multi-agent orchestration. Include chemistry-inspired 
bonding patterns, event sourcing through Redis Streams, ≥0.75 constitutional compliance, and 
Mangle rule validation.

# Step 3: Clarify constitutional requirements
/clarify  # Interactive: Answers questions about neural bonding thresholds, agent patterns, etc.

# Step 4: Generate constitutional plan
/plan Python-based system using Redis for event sourcing, SQLite for Mangle facts, FastAPI for 
API layer. Neural atoms as modular classes with bonding managers and constitutional gates.

# Step 5: Validate constitutional compliance
Review generated plan for ≥0.75 constitutional score, neural bonding integrity ≥0.85, 
and zero Mangle rule violations.

# Step 6: Generate implementation
/implement  # Creates code with constitutional gates embedded
```

**PowerShell Integration:**
```powershell
# Initialize Spec-Kit for Super Alita
sask  # Alias for Start-SuperAlitaSpecKit

# Create constitutional specification
cspec "Multi-agent neural coordination with constitutional governance"

# Validate compliance
ctest  # Runs constitutional validator + neural integrity + Mangle
```

#### Constitutional Gates at Each Phase

**Entry Gate (Specification):**
- Constitutional compliance ≥0.75
- All requirements testable
- Neural atom bonding sites defined
- Agent coordination patterns specified

**Process Gate (Planning):**
- Architecture preserves neural patterns
- Multi-agent orchestration patterns validated
- Event sourcing integration confirmed
- Mangle rules pass (circular_dependencies, untested_complex_functions)

**Exit Gate (Implementation):**
- Final constitutional score ≥0.75
- Neural bonding integrity ≥0.85
- Test coverage ≥70%
- All constitutional articles satisfied

**Templates:** `templates/sdd/spec-template.md`, `plan-template.md`, `tasks-template.md`

### Build, Test, Lint Commands

```powershell
# Install dependencies
pip install -r requirements.txt -r requirements-test.txt
# Or: make deps

# Run tests
pytest -q                           # All tests
pytest -q -k "integration"          # Filter by name
pytest -q -m integration_redis      # Filter by marker

# Linting & formatting
ruff check src tests                # Lint
black . -l 79                       # Format (79 chars for Super Alita)
isort . --profile=black --line-length=79  # Sort imports
pre-commit run --all-files          # Run all hooks

# Type checking
mypy --strict src core              # Focus on src/core, src/sandbox first
```

**VS Code Tasks:** Run from Command Palette → "Tasks: Run Task"
- `🔍 Full Quality Pipeline` - Runs lint + typecheck + tests + format
- `🎯 Start Super Alita` - Launches runtime server
- `🌐 Start Dev Server` - Launches FastAPI on port 8080

## 🏛️ Constitutional Compliance (Non-Negotiable)

All code must maintain **≥0.75 compliance** with `specs/071-rules-for-ai/spec.md`:

### Six Articles Framework
1. **Library-First Principle** - Design as reusable libraries with clean APIs
2. **Test-First Imperative** - Acceptance criteria before implementation
3. **Simplicity Gate** - ≤3 projects per feature, justify complexity
4. **Anti-Abstraction Gate** - Use framework features directly, avoid wrappers
5. **Integration-First Testing** - Prefer real services over mocks
6. **Clarity and Unambiguity** - No TBDs, comprehensive edge case documentation

**Before suggesting changes:**
- Reference specific constitutional articles
- Validate against Mangle facts database (`.ai/facts.sqlite`)
- Check for circular dependencies, untested complex functions, hot paths
- Ensure ≥70% test coverage for new code

### Mangle Reasoning Engine Integration
**Mangle** provides code analysis via Datalog-like rules on SQLite facts.

```python
# Query before structural changes
from src.unified_intelligence.code_reasoning.rules import apply_rules

results = apply_rules(db_path=".ai/facts.sqlite", rule="circular_dependencies")
# Check: untested_complex_functions, hot_paths, config_cascade_breaks
```

**Key files:** `src/unified_intelligence/code_reasoning/ingester.py`, `rules.py`, `models.py`

## 🔒 Security & Execution Boundaries (Critical)

### Sandbox Execution - Never Bypass
**All dynamic code** must flow through `src/sandbox/exec_sandbox.py`:
- AST-based restrictions: no imports, no attribute access, no subscripts
- Allowlist-based execution environment (`src/sandbox/registry.py`)
- For more isolation: `src/sandbox/py_venv_runner.py` (temp venv)

```python
from src.sandbox.exec_sandbox import evaluate_expression, execute_statements

result = evaluate_expression("2 + 2")  # Safe
execute_statements("x = 42")           # Safe with restrictions
```

### Subprocess Safety - Use src/core/proc.py
**Never use `shell=True`**. Always sanitize arguments:

```python
from src.core.proc import run

stdout = run(["git", "status"], timeout=5.0, cwd=workspace_root)
# Raises ProcError on non-zero exit, sanitizes args against injection
```

### YAML Safety - Use src/core/yaml_utils.py
**Never use `yaml.load()`** - use safe loaders only (enforced in `yaml_utils.py`).

## 🧬 Neural Atom Patterns (Chemistry-Inspired Modularity)

### Creating Atoms with Proper Metadata

```python
from src.core.neural_atom import NeuralAtom, NeuralAtomMetadata

metadata = NeuralAtomMetadata(
    atom_type="tool_output",
    title="Query Result",
    source="search_plugin",
    confidence=0.92,
    tags=["search", "production"]
)

atom = NeuralAtom(
    key="query_result_abc123",
    value=result_data,
    parent_keys=["parent_atom_id"],  # Genealogy tracking
    birth_event="tool_executed",
    metadata=metadata
)

# Deterministic UUID from content
# Same content → same UUID (enables deduplication)
```

### Event Bus → Neural Atom Bridge

```python
# Bridge automatically converts events to atoms
from src.core.neural_atom_bridge import NeuralAtomBridge

bridge = NeuralAtomBridge(mcp_server)
await bridge.handle_event(event)  # Maps to atom type based on event.event_type
```

**Event mappings:** `user_message`, `tool_created`, `sot_executed`, `state_transition`, `tool_call`, `tool_response`

## 🔧 Code Conventions (Project-Specific)

### Naming & Style
- **Python 3.11+** required
- **4-space indentation**, double quotes, explicit type hints everywhere
- `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants
- **Black formatting:** 79 characters line length (not 88 - project override)

### Module Organization
```
src/
├── core/           # Event bus, neural atoms, plugin interface, execution flow
├── plugins/        # Plugin implementations (inherit PluginInterface)
├── sandbox/        # Secure execution environment
├── sdd/            # Specification-driven development (FastAPI endpoints)
├── orchestration/  # Unified orchestrator, event sanitizer
├── neural/         # Neural atom/bond storage, MCP server
├── abilities/      # Tool/ability implementations
└── main.py         # FastAPI app entry point (4245 lines - see startup logic)
```

**Tests mirror src/ structure:** `tests/` plus top-level `test_*.py`

### Commit Messages
`[module] Short description` (e.g., `[sandbox] Harden exec policy`, `[sdd] Add constitutional gates`)

## 📚 Context Triggers (When to Reference What)

| Topic | Key Files |
|-------|-----------|
| **Authentication/Security** | `src/orchestration/event_sanitizer.py`, `docs/constitution_update_checklist.md` |
| **SDD Pipeline** | `src/sdd/router.py`, `templates/sdd/*.md`, `src/sdd/validators.py` |
| **Neural Architecture** | `src/core/neural_atom.py`, `src/neural/atom.py`, `src/neural/store.py` |
| **Plugin Development** | `src/core/plugin_interface.py`, `src/plugins/compose_plugin.py` |
| **Sandbox Execution** | `src/sandbox/exec_sandbox.py`, `src/sandbox/py_venv_runner.py` |
| **Testing Strategy** | Query Mangle `untested_complex_functions`, see `AGENTS.md` |
| **Event Bus** | `src/core/event_bus.py`, `src/core/events.py` |
| **Orchestration** | `src/orchestration/unified_orchestrator.py`, `src/core/execution_flow.py` |

## 🚀 Quality Standards (Enforced by CI)

- **Constitutional Compliance:** ≥0.75 threshold at all stages
- **Test Coverage:** ≥70% for new code, integration tests preferred
- **Code Quality:** Type hints required, complexity <10, 0 circular dependencies
- **Mangle Validation:** Run `scripts/unified_sdd_mangle.py` before PRs
- **Pre-commit Hooks:** Ruff, Black, isort, mypy - must pass before merge

## � Response Guidelines

1. **Ground in Reality** - Reference actual files (`src/core/proc.py:32`), not hypotheticals
2. **Extend, Don't Invent** - Enhance existing modules over creating new ones
3. **Constitutional First** - Cite articles from `specs/071-rules-for-ai/spec.md`
4. **Copy-Pasteable Code** - PowerShell commands for Windows, full imports, no placeholders
5. **Mangle Before Refactor** - Query facts database for dependency impacts
6. **Security by Default** - Always mention sandbox/proc.py safety when suggesting execution

## 🎯 Quick Reference

**Start Server:** `uvicorn src.main:app --reload --port 8080`
**Run SDD:** `python -m src.sdd.sdd_cli specify "feature description"`
**Test Suite:** `pytest -q` (≥70% coverage required)
**Lint:** `ruff check src tests && black . -l 79`
**Constitutional Check:** `python scripts/unified_sdd_mangle.py --repo . --spec .spec --db .ai/facts.sqlite`

## 🔧 PowerShell Profile Integration (Windows Development)

### Super Alita Management Commands

Add these to your PowerShell profile (`$PROFILE`) for seamless Super Alita workflows:

```powershell
# Spec-Kit Integration for Constitutional SDD
function Start-SuperAlitaSpecKit {
    Write-Host "🎯 Initializing Super-Alita Spec-Kit Integration..." -ForegroundColor Cyan
    
    # Ensure Spec-Kit is available via uvx
    if (!(Get-Command "uvx" -ErrorAction SilentlyContinue)) {
        Write-Warning "uvx not found. Install with: pip install uv"
        return
    }
    
    # Initialize with Super-Alita constitutional templates
    uvx spec-kit init . --force
    
    Write-Host "✅ Spec-Kit ready for constitutional SDD!" -ForegroundColor Green
    Write-Host "Workflow: /constitution → /specify → /clarify → /plan → /implement" -ForegroundColor Cyan
}

function Invoke-ConstitutionalSpecify {
    param(
        [Parameter(Mandatory)]
        [string]$FeatureDescription,
        [switch]$IncludeNeuralAtoms,
        [double]$ConstitutionalThreshold = 0.75
    )
    
    Write-Host "🏛️ Creating constitutional specification..." -ForegroundColor Yellow
    Write-Host "Feature: $FeatureDescription" -ForegroundColor White
    Write-Host "Constitutional Threshold: $ConstitutionalThreshold" -ForegroundColor Cyan
    
    if ($IncludeNeuralAtoms) {
        Write-Host "🧬 Neural atom bonding patterns: ENABLED" -ForegroundColor Green
        Write-Host "Run in Copilot Chat:" -ForegroundColor Yellow
        Write-Host "/specify $FeatureDescription Include chemistry-inspired neural atom bonding with ≥0.85 integrity threshold." -ForegroundColor White
    } else {
        Write-Host "Run in Copilot Chat:" -ForegroundColor Yellow
        Write-Host "/specify $FeatureDescription" -ForegroundColor White
    }
}

function Test-ConstitutionalCompliance {
    param([string]$SpecPath = ".specify/specs")
    
    Write-Host "🏛️ Testing constitutional compliance..." -ForegroundColor Yellow
    
    # Run Python constitutional validation
    if (Test-Path "scripts/constitutional_validator.py") {
        python scripts/constitutional_validator.py --path $SpecPath --threshold 0.75
    }
    
    # Check neural integrity
    if (Test-Path "scripts/neural_integrity_check.py") {
        python scripts/neural_integrity_check.py --path $SpecPath --threshold 0.85
    }
    
    # Validate with Mangle
    python scripts/unified_sdd_mangle.py --repo . --db .ai/facts.sqlite
    
    Write-Host "✅ Constitutional validation complete" -ForegroundColor Green
}

# Spec-Kit aliases
Set-Alias sask Start-SuperAlitaSpecKit
Set-Alias cspec Invoke-ConstitutionalSpecify
Set-Alias ctest Test-ConstitutionalCompliance
```

### Core Super Alita Commands

```powershell
# Start Super Alita environment with all dependencies
function Start-SuperAlitaEnvironment {
    param([switch]$FullStack)
    
    Write-Host "🚀 Initializing Super-Alita..." -ForegroundColor Cyan
    
    # Activate venv
    if (Test-Path ".\.venv\Scripts\Activate.ps1") {
        & .\.venv\Scripts\Activate.ps1
        Write-Host "✅ Virtual environment activated" -ForegroundColor Green
    }
    
    # Start Redis if full stack mode (optional)
    if ($FullStack) {
        Start-Process redis-server -WindowStyle Minimized -ErrorAction SilentlyContinue
        Write-Host "🔴 Redis server started" -ForegroundColor Yellow
    }
    
    Write-Host "🎯 Super-Alita Ready! Use 'sac-help' for commands" -ForegroundColor Green
}

# Super Alita Command wrapper
function Invoke-SuperAlitaCommand {
    param(
        [ValidateSet("validate", "neural-check", "sdd", "mangle", "health")]
        [string]$Command,
        [Parameter(ValueFromRemainingArguments)]
        [string[]]$Arguments
    )
    
    switch ($Command) {
        "validate" {
            python scripts/unified_sdd_mangle.py --repo . --db .ai/facts.sqlite --report .ai/report.json
            if (Test-Path ".ai/report.json") {
                $report = Get-Content ".ai/report.json" | ConvertFrom-Json
                $score = $report.constitutional_score
                $color = if ($score -ge 0.75) {"Green"} else {"Red"}
                Write-Host "🏛️ Constitutional Score: $score" -ForegroundColor $color
            }
        }
        "neural-check" {
            python -c @"
import asyncio
from src.neural.store import MessageStore
store = MessageStore('.ai/neural.db')
atoms = store.list_atoms()
print(f'🧬 Neural Atoms: {len(atoms)} stored')
"@
        }
        "sdd" {
            $feature = if ($Arguments) { $Arguments[0] } else { Read-Host "Feature name" }
            python -m src.sdd.sdd_cli specify "$feature"
        }
        "mangle" {
            $rule = if ($Arguments) { $Arguments[0] } else { "circular_dependencies" }
            python -c "from src.unified_intelligence.code_reasoning.rules import apply_rules; print(apply_rules('.ai/facts.sqlite', '$rule'))"
        }
        "health" {
            Write-Host "🏥 System Health Check" -ForegroundColor Cyan
            pytest -q -k "health" --tb=no
            Invoke-SuperAlitaCommand -Command validate
            Invoke-SuperAlitaCommand -Command neural-check
        }
    }
}

# Convenience aliases
Set-Alias sae Start-SuperAlitaEnvironment
Set-Alias sac Invoke-SuperAlitaCommand

function Show-SuperAlitaHelp {
    Write-Host "🎯 Super-Alita Commands:" -ForegroundColor Cyan
    Write-Host "  sae              - Start environment" -ForegroundColor Yellow
    Write-Host "  sac validate     - Constitutional validation" -ForegroundColor Yellow
    Write-Host "  sac neural-check - Neural atom count" -ForegroundColor Yellow
    Write-Host "  sac sdd [name]   - SDD workflow" -ForegroundColor Yellow
    Write-Host "  sac mangle [rule]- Mangle analysis" -ForegroundColor Yellow
    Write-Host "  sac health       - Full health check" -ForegroundColor Yellow
}
Set-Alias sac-help Show-SuperAlitaHelp
```

**Usage:**
```powershell
# Start environment
sae

# Run constitutional validation
sac validate

# Check neural atom storage
sac neural-check

# Run SDD workflow
sac sdd "Add streaming endpoints"

# Run Mangle analysis
sac mangle circular_dependencies

# Full system health check
sac health
```

## 🔌 Enhanced MCP Integration

### Current MCP Server Architecture

Super Alita already has MCP integration through:
- `src/neural/mcp_server.py` - Core MCP server for neural operations
- `src/vscode_integration/agent_mcp_server.py` - VS Code agent integration
- Event bus → Neural Atom bridge pattern (`src/core/neural_atom_bridge.py`)

### Available MCP Tools (via agent_mcp_server.py)

When configured in `.vscode/mcp.json`, Copilot can access:
- **Neural Atom Operations**: Create, query, bond atoms
- **Event Sourcing**: Query event history from SQLite store
- **Constitutional Validation**: Check compliance scores
- **SDD Workflows**: Trigger specify/plan/tasks phases

### Configuration Pattern

```json
{
  "mcpServers": {
    "super-alita-neural": {
      "command": "python",
      "args": ["src/neural/mcp_server.py"],
      "env": {
        "NEURAL_STORE_PATH": ".ai/neural.db",
        "CONSTITUTIONAL_THRESHOLD": "0.75"
      }
    },
    "super-alita-agent": {
      "command": "python",
      "args": ["src/vscode_integration/agent_mcp_server.py"],
      "env": {
        "EVENT_BUS_REDIS": "redis://localhost:6379"
      }
    }
  }
}
```

## 🏥 System Health & Validation Workflows

### Pre-Commit Validation Workflow

```powershell
# Run before any commit
function Invoke-PreCommitValidation {
    Write-Host "🔍 Running pre-commit validation..." -ForegroundColor Cyan
    
    # 1. Lint and format
    ruff check src tests --fix
    black . -l 79
    isort . --profile=black --line-length=79
    
    # 2. Type check
    mypy --strict src/core src/sandbox
    
    # 3. Constitutional validation
    sac validate
    
    # 4. Test suite
    pytest -q
    
    Write-Host "✅ Pre-commit validation complete!" -ForegroundColor Green
}
Set-Alias saval Invoke-PreCommitValidation
```

### Feature Development Workflow

```powershell
# Complete feature development workflow
function Start-FeatureDevelopment {
    param([Parameter(Mandatory)][string]$FeatureName)
    
    Write-Host "🚀 Starting feature development: $FeatureName" -ForegroundColor Cyan
    
    # 1. SDD Specification
    Write-Host "📝 Phase 1: Specification" -ForegroundColor Yellow
    python -m src.sdd.sdd_cli specify "$FeatureName"
    
    # 2. Generate plan
    Write-Host "📋 Phase 2: Planning" -ForegroundColor Yellow
    $featureId = "feat-$(($FeatureName -replace '\s','-').ToLower())"
    python -m src.sdd.sdd_cli plan $featureId
    
    # 3. Generate tasks
    Write-Host "✅ Phase 3: Tasks" -ForegroundColor Yellow
    python -m src.sdd.sdd_cli tasks $featureId
    
    # 4. Constitutional check
    Write-Host "🏛️ Phase 4: Constitutional Validation" -ForegroundColor Yellow
    sac validate
    
    Write-Host "🎯 Feature '$FeatureName' ready for implementation!" -ForegroundColor Green
}
Set-Alias sadev Start-FeatureDevelopment
```

### Continuous Quality Monitoring

```powershell
# Monitor code quality metrics
function Show-QualityMetrics {
    Write-Host "📊 Super-Alita Quality Metrics" -ForegroundColor Cyan
    
    # Constitutional compliance
    $report = Get-Content ".ai/report.json" -ErrorAction SilentlyContinue | ConvertFrom-Json
    if ($report) {
        $constScore = $report.constitutional_score
        $constColor = if ($constScore -ge 0.75) {"Green"} else {"Red"}
        Write-Host "🏛️  Constitutional: $constScore" -ForegroundColor $constColor
    }
    
    # Test coverage (parse pytest output)
    $coverage = pytest --cov=src --cov-report=term-missing -q 2>&1 | Select-String "TOTAL.*(\d+)%"
    if ($coverage) {
        Write-Host "🧪 Test Coverage: $($coverage.Matches.Groups[1].Value)%" -ForegroundColor Green
    }
    
    # Neural atom count
    Write-Host "🧬 Neural Atoms: $(python -c 'from src.neural.store import MessageStore; print(len(MessageStore(\".ai/neural.db\").list_atoms()))')" -ForegroundColor Cyan
    
    # Mangle analysis summary
    Write-Host "🔍 Mangle Analysis:" -ForegroundColor Cyan
    @("circular_dependencies", "untested_complex_functions", "hot_paths") | ForEach-Object {
        $results = python -c "from src.unified_intelligence.code_reasoning.rules import apply_rules; print(len(apply_rules('.ai/facts.sqlite', '$_')))" 2>$null
        if ($results) {
            Write-Host "   - $_: $results findings" -ForegroundColor Yellow
        }
    }
}
Set-Alias saqm Show-QualityMetrics
```

## 📋 VS Code Task Integration Tips

Leverage existing VS Code tasks (see workspace task list above) with Copilot:
- Ask: "Run the Full Quality Pipeline task" → Copilot knows about `🔍 Full Quality Pipeline`
- Ask: "Start the dev server" → Copilot will reference `🌐 Start Dev Server`
- Ask: "Run SDD validation" → Copilot knows about `⚖️ Constitutional Validate`

## 🎓 Teaching Copilot Your Patterns

When working with Copilot on Super Alita code:

1. **Always mention constitutional compliance**: "Ensure ≥0.75 constitutional score"
2. **Reference neural atoms**: "Create atoms with deterministic UUIDs"
3. **Use sandbox for dynamic code**: "Execute through `src/sandbox/exec_sandbox.py`"
4. **Cite SDD workflow**: "Follow SDD: specify → plan → tasks"
5. **Query Mangle first**: "Check Mangle for circular dependencies before refactoring"

---

You're working with a cutting-edge constitutional AI orchestration platform. Every response should demonstrate understanding of event-sourcing, neural atom genealogy, sandbox security, and the SDD workflow that makes Super-Alita production-ready.
