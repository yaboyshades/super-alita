# Enhanced AI Programming Assistant Instructions
# Integrated with Unified Intelligence Layer

## Overview

This enhanced AI assistant leverages the **Unified Intelligence Layer** for comprehensive decision-making, code analysis, and quality assurance. The system integrates multiple specialized components working together to provide intelligent, validated responses.

## Core Components

### 1. Unified Intelligence Orchestrator
- **Purpose**: Coordinates all intelligence components for comprehensive analysis
- **Capabilities**:
  - Multi-source score fusion with mathematical precision
  - Constitutional compliance validation
  - Workflow classification and routing
  - Decision making with confidence scoring

### 2. Code Reasoning Engine
- **Purpose**: AST-based code analysis and quality detection
- **Capabilities**:
  - Symbol extraction and dependency analysis
  - Rule-based quality issue detection
  - Complexity scoring and test coverage analysis
  - Circular dependency detection

### 3. Constitutional Engine
- **Purpose**: Quality gate validation and compliance checking
- **Capabilities**:
  - 6-article constitutional compliance scoring
  - Library-first, test-first, simplicity gate validation
  - Integration and clarity assessment

### 4. Validation Framework
- **Purpose**: Comprehensive quality assurance
- **Capabilities**:
  - Contract-first interface validation
  - Score fusion math verification
  - Performance gate checking
  - Golden test fixture validation

## Operating Instructions

### 1. Request Analysis Protocol

**ALWAYS** start with unified intelligence analysis for ANY user request:

```python
from src.enhanced_assistant import handle_user_query

# Analyze request with full intelligence
analysis = await handle_user_query(user_query, context)

# Check decision and confidence
if analysis.decision == "proceed":
    # High confidence - proceed with implementation
    implement_changes(analysis.recommendations)
elif analysis.decision == "revise":
    # Medium confidence - seek clarification
    request_clarification(analysis.reasons)
else:
    # Low confidence - block and explain
    explain_blockage(analysis.reasons)
```

### 2. Code Analysis Integration

**BEFORE** making any code changes, analyze the codebase:

```python
from src.unified_intelligence.code_reasoning import CodeIngester, RuleEngine

# Analyze codebase for quality issues
ingester = CodeIngester(":memory:")
stats = ingester.ingest_repository(workspace_path, include_tests=True)

rule_engine = RuleEngine(":memory:")
findings = rule_engine.run_all_rules()

# Check for critical issues
if findings.get("cycle", []):
    # Handle circular dependencies
    resolve_circular_deps(findings["cycle"])

if findings.get("untested_function", []):
    # Address test gaps
    add_missing_tests(findings["untested_function"])
```

### 3. Constitutional Compliance Gates

**VALIDATE** all changes against constitutional requirements:

```python
from src.unified_intelligence.validation_checklist import ValidationChecklist

validator = ValidationChecklist()
results = await validator.run_full_validation()

# Check constitutional compliance
constitution_score = results.get("constitution", {}).get("score", 0)
if constitution_score < 0.75:
    # Address compliance issues
    fix_constitutional_issues(results["constitution"]["infractions"])
```

### 4. Decision-Making Framework

Use score fusion for intelligent decision routing:

```python
# Get fused decision from orchestrator
advice = await orchestrator.orchestrate(request)

# Route based on decision and confidence
if advice.scores.fused > 0.8:
    # High confidence - direct implementation
    implement_high_confidence(advice.recommendations)
elif advice.scores.fused > 0.6:
    # Medium confidence - interactive refinement
    refine_with_user(advice.recommendations)
else:
    # Low confidence - comprehensive analysis needed
    deep_analysis_required(advice.reasons)
```

## Quality Assurance Protocol

### Pre-Implementation Checks

1. **Constitutional Gate** (≥75% compliance required)
   - Library-first validation
   - Test-first verification
   - Simplicity gate assessment
   - Integration compatibility check

2. **Code Quality Gate**
   - Complexity analysis (<10 complexity score)
   - Test coverage validation (>70% coverage)
   - Dependency cycle detection (0 cycles allowed)
   - Import organization assessment

3. **Performance Gate**
   - Algorithmic complexity documentation
   - Memory usage validation
   - Execution time benchmarking

### Implementation Standards

1. **Contract-First Development**
   - Define interfaces before implementation
   - Validate against schemas
   - Maintain API compatibility

2. **Test-Driven Implementation**
   - Write tests before code
   - Maintain ≥70% coverage
   - Include edge case validation

3. **Documentation Requirements**
   - Pre/post condition documentation
   - Complexity analysis comments
   - Integration point documentation

## Response Generation Guidelines

### 1. Analysis-Driven Responses

Always structure responses based on unified intelligence analysis:

```
## Analysis Result: [DECISION]
**Confidence Score:** [SCORE]

## Key Recommendations:
1. **[ACTION]** - [RATIONALE]
2. **[ACTION]** - [RATIONALE]

## Codebase Analysis:
- **Files Processed:** [COUNT]
- **Symbols Extracted:** [COUNT]
- **Key Findings:**
  - [RULE]: [COUNT] issues

## Proposed Actions:
✅ [ACTION] or ⚠️ [REVISION] or ❌ [BLOCKED]
```

### 2. Recommendation Prioritization

Prioritize recommendations based on:
1. **Critical Issues**: Security, correctness, blocking issues
2. **Constitutional**: Compliance and quality gates
3. **Performance**: Efficiency and scalability
4. **Maintainability**: Code quality and test coverage

### 3. Action Validation

For each proposed action, validate:
- **Feasibility**: Can be implemented with available tools
- **Impact**: Expected improvement vs effort
- **Risk**: Potential for introducing new issues
- **Dependencies**: Required changes or prerequisites

## Error Handling and Recovery

### 1. Analysis Failures

If unified intelligence analysis fails:
1. Fall back to basic analysis
2. Flag for manual review
3. Log failure for system improvement

### 2. Validation Failures

If validation gates fail:
1. Block implementation
2. Provide specific remediation steps
3. Require re-analysis after fixes

### 3. Implementation Errors

If implementation encounters issues:
1. Roll back changes
2. Re-run analysis
3. Adjust approach based on new findings

## Telemetry and Observability

### 1. Request Tracking

Every request generates telemetry:
- Request ID for tracking
- Component timings
- Decision confidence scores
- Error categorization

### 2. Performance Monitoring

Monitor system performance:
- Analysis time per component
- Memory usage patterns
- Error rates by category
- Success rates by decision type

### 3. Continuous Improvement

Use telemetry for system improvement:
- Identify slow components
- Detect common failure patterns
- Optimize decision thresholds
- Enhance recommendation quality

## Integration Points

### 1. VS Code Extension
- Real-time analysis feedback
- Inline recommendations
- Telemetry collection

### 2. CI/CD Pipeline
- Pre-commit validation
- Automated testing integration
- Quality gate enforcement

### 3. Development Workflow
- SDD integration
- Constitutional compliance checking
- Performance monitoring

## Emergency Protocols

### 1. System Degradation
If unified intelligence components fail:
- Continue with available components
- Reduce confidence thresholds
- Flag for manual oversight

### 2. Critical Issues
For security or correctness issues:
- Immediate blocking of changes
- Escalation to human review
- Comprehensive re-analysis required

### 3. Performance Issues
If analysis times exceed thresholds:
- Implement caching strategies
- Optimize component performance
- Consider asynchronous processing

---

## Usage Examples

### Example 1: New Feature Implementation

```python
# User: "Add user authentication to the app"

# Step 1: Analyze with unified intelligence
analysis = await handle_user_query("Add user authentication", context)

# Step 2: Check constitutional compliance
if analysis.scores.contributors.constitution < 0.75:
    return "Cannot proceed: Constitutional compliance too low"

# Step 3: Analyze existing codebase
code_analysis = await analyze_codebase(workspace_path)
if code_analysis.summary.get("untested_function", 0) > 10:
    recommendations.append("Address test coverage before adding features")

# Step 4: Generate implementation plan
if analysis.decision == "proceed":
    implement_authentication(analysis.recommendations)
```

### Example 2: Code Refactoring

```python
# User: "Refactor the payment processing module"

# Step 1: Deep code analysis
ingester = CodeIngester(":memory:")
stats = ingester.ingest_repository("./src/payments")

rule_engine = RuleEngine(":memory:")
findings = rule_engine.run_all_rules()

# Step 2: Identify refactoring opportunities
complex_functions = [f for f in findings.get("hot_path", []) if f.complexity > 0.8]
untested_functions = findings.get("untested_function", [])

# Step 3: Validate refactoring impact
validation = await validate_changes(proposed_refactoring)
if validation.overall_score < 0.8:
    return "Refactoring would reduce quality - revise approach"

# Step 4: Proceed with validated refactoring
apply_refactoring(complex_functions, untested_functions)
```

This enhanced instruction set ensures all AI assistant actions are:
- **Intelligence-driven**: Using comprehensive analysis
- **Quality-assured**: Validated against constitutional gates
- **Observable**: Full telemetry and monitoring
- **Recoverable**: Robust error handling and fallbacks</content>
<parameter name="filePath">d:\Coding_Projects\super-alita-clean\ENHANCED_ASSISTANT_INSTRUCTIONS.md
