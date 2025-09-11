# Manual Testing Procedures

**Document**: manual-testing.md
**Constitutional Article**: IV - Integration-First Testing
**Last Updated**: September 10, 2025

## Constitutional Manual Testing Philosophy

This document follows Article IV of the Super-Alita Constitutional Framework: "Validate system interactions in realistic environments."

Manual testing validates user experiences and constitutional compliance that automated tests cannot fully cover:

1. **Human-AI Interaction Quality**: Constitutional prompt optimization effectiveness
2. **Cross-System Integration**: VS Code + CLI + Core Engine coordination
3. **Constitutional User Experience**: How well constitutional principles enhance developer workflow
4. **Real-World Scenarios**: Actual development projects using SDD methodology

## Test Environment Setup

### Prerequisites Checklist
- [ ] Python 3.11+ installed and configured
- [ ] VS Code with SDD extension installed
- [ ] AI API keys configured (OpenAI, Claude, Gemini)
- [ ] Git repository access for testing
- [ ] Constitutional framework loaded and active
- [ ] Test project directory: `~/sdd-manual-testing/`

### Configuration Validation
```bash
# Verify constitutional framework
sdd validate --type constitutional-framework ~/.sdd/constitution.md

# Check API connectivity
sdd test-apis --all

# Verify VS Code extension
code --list-extensions | grep alita-sdd

# Constitutional environment check
echo $SDD_CONSTITUTIONAL_MODE  # Should be 'true'
```

## Epic 1: Specification Creation & Management

### Manual Test 1.1: Create Feature Specification via VS Code

**Objective**: Validate `/specify` command through VS Code with constitutional optimization

**Test Steps**:

1. **Setup**
   - Open VS Code in test project directory
   - Open command palette (Ctrl+Shift+P)
   - Verify "Alita SDD: Specify" command is available

2. **Execute Specify Command**
   - Run command: `Alita SDD: Specify`
   - Input: "Build a photo album organizer that allows users to create albums grouped by date and reorganize photos by drag and drop"
   - Observe constitutional APE engine optimization process

3. **Validate Constitutional Processing**
   - **Expected**: Progress indicator shows "Constitutional optimization in progress..."
   - **Expected**: APE engine generates multiple prompt variations
   - **Expected**: Constitutional scoring appears in real-time
   - **Verify**: Score ≥0.75 before proceeding

4. **Review Generated Specification**
   - **Expected**: New file created: `specs/002-photo-album-organizer/spec.md`
   - **Expected**: Constitutional compliance section included
   - **Expected**: User stories follow constitutional format
   - **Expected**: [NEEDS CLARIFICATION] markers for ambiguities

5. **Constitutional Compliance Validation**
   ```markdown
   # Expected sections in generated spec:
   - Constitutional Compliance Review
   - Library-First Research Requirements
   - Test-First Development Plan
   - Simplicity Constraints
   - Integration Requirements
   - Alternative Approaches Considered
   ```

**Pass Criteria**:
- [ ] Command executes without errors
- [ ] Constitutional score ≥0.75
- [ ] All six articles addressed in specification
- [ ] File structure follows SDD conventions
- [ ] APE optimization provides clear improvements

**Manual Verification Notes**:
```
Tester: ________________
Date: __________________
Constitutional Score: ___
Issues Found: __________
_______________________
```

### Manual Test 1.2: Iterative Specification Refinement

**Objective**: Test constitutional feedback loop and clarification handling

**Test Steps**:

1. **Create Ambiguous Specification**
   - Input: "Build a system for managing stuff"
   - Observe constitutional violation detection

2. **Review Constitutional Feedback**
   - **Expected**: Multiple [NEEDS CLARIFICATION] markers
   - **Expected**: Constitutional score <0.50
   - **Expected**: Specific improvement suggestions

3. **Refine Based on Constitutional Guidance**
   - Follow suggestions to clarify requirements
   - Add library research requirements (Article I)
   - Include test-first specifications (Article II)
   - Define simplicity constraints (Article III)

4. **Validate Improvement**
   - Re-run constitutional validation
   - **Expected**: Score improvement >0.25
   - **Expected**: Fewer clarification markers

**Pass Criteria**:
- [ ] Constitutional violations properly detected
- [ ] Improvement suggestions are actionable
- [ ] Iterative refinement increases compliance score
- [ ] Final specification meets constitutional threshold

## Epic 2: Implementation Planning & Code Generation

### Manual Test 2.1: Generate Implementation Plan

**Objective**: Validate `/plan` command with constitutional gate enforcement

**Test Steps**:

1. **Execute Plan Command**
   - Use specification from Test 1.1
   - Command: `Alita SDD: Plan`
   - Technology preference: "Python, FastAPI, SQLite"

2. **Observe Constitutional Gate Processing**
   - **Expected**: Pre-implementation gates displayed
   - **Expected**: Simplicity Gate validation
   - **Expected**: Anti-Abstraction Gate check
   - **Expected**: Integration-First Gate verification

3. **Review Generated Plan**
   - Verify constitutional phases structure
   - Check technology justifications (Article VI)
   - Validate test-first workflows (Article II)

4. **Constitutional Gate Validation**
   ```
   Phase -1 Gates Checklist:
   - [ ] ≤3 projects constraint met
   - [ ] Direct framework usage (no wrappers)
   - [ ] Real environment testing planned
   - [ ] CLI interfaces mandatory
   ```

**Pass Criteria**:
- [ ] All constitutional gates pass or are justified
- [ ] Technology choices include library-first analysis
- [ ] Implementation phases follow test-first approach
- [ ] Plan maps directly to specification requirements

### Manual Test 2.2: Code Generation with Constitutional Constraints

**Objective**: Test automated code generation with constitutional compliance

**Test Steps**:

1. **Generate Library Structure**
   - Command: `sdd generate --component photo-album-core`
   - Observe constitutional template application

2. **Validate Generated Code Structure**
   ```
   Expected structure:
   photo-album-core/
   ├── cli.py              # CLI interface (constitutional requirement)
   ├── core.py             # Core functionality
   ├── tests/              # Test-first approach
   │   ├── test_cli.py
   │   └── test_core.py
   └── README.md           # Usage documentation
   ```

3. **Check Constitutional Compliance**
   - Function length ≤50 lines
   - Cyclomatic complexity ≤10
   - CLI interface mandatory
   - Tests generated before implementation

4. **Manual Code Review**
   - Verify library-first imports
   - Check for unnecessary abstractions
   - Validate test coverage requirements

**Pass Criteria**:
- [ ] Generated code passes constitutional validation
- [ ] CLI interface implemented
- [ ] Tests exist before implementation code
- [ ] No constitutional violations in generated artifacts

## Epic 3: End-to-End Workflow Integration

### Manual Test 3.1: Complete SDD Workflow

**Objective**: Validate entire specification-to-code workflow with constitutional compliance

**Test Steps**:

1. **Full Workflow Execution**
   - Start with natural language requirement
   - Execute: specify → plan → tasks → generate
   - Monitor constitutional compliance throughout

2. **Cross-Component Integration**
   - VS Code extension ↔ Core Engine communication
   - CLI tools ↔ API integration
   - Constitutional validation ↔ All components

3. **Real Development Scenario**
   - Use SDD for actual feature development
   - Test with real Git repository
   - Integrate with existing codebase

4. **Constitutional Consistency Check**
   - Verify same constitutional principles across all tools
   - Check compliance scoring consistency
   - Validate cross-component quality gates

**Pass Criteria**:
- [ ] End-to-end workflow completes successfully
- [ ] Constitutional compliance maintained throughout
- [ ] All generated artifacts are usable
- [ ] Integration points work seamlessly

### Manual Test 3.2: Real-World Project Integration

**Objective**: Test SDD integration with existing development projects

**Test Steps**:

1. **Existing Project Setup**
   - Clone existing Python project
   - Initialize SDD in project root: `sdd init`
   - Configure constitutional framework

2. **Add New Feature Using SDD**
   - Create specification for new feature
   - Generate implementation plan
   - Validate constitutional compatibility

3. **Integration Validation**
   - Ensure SDD artifacts integrate with existing code
   - Check constitutional compliance of existing code
   - Validate test integration and coverage

4. **Team Workflow Testing**
   - Test specification review process
   - Validate constitutional scoring consistency
   - Check collaborative editing scenarios

**Pass Criteria**:
- [ ] SDD integrates with existing projects
- [ ] Constitutional framework adapts to project constraints
- [ ] Generated code follows project conventions
- [ ] Team collaboration workflow functions properly

## Constitutional Validation Manual Tests

### Manual Test 4.1: Constitutional Scoring Accuracy

**Objective**: Validate constitutional scoring against manual expert review

**Test Steps**:

1. **Create Test Artifacts**
   - High-compliance specification (expected score: 0.90+)
   - Medium-compliance plan (expected score: 0.75-0.85)
   - Low-compliance code (expected score: <0.65)

2. **Manual Constitutional Review**
   - Expert review each artifact against six articles
   - Document expected scores and violations
   - Identify specific constitutional issues

3. **Automated Scoring Comparison**
   - Run constitutional validation on each artifact
   - Compare automated scores with manual review
   - Analyze discrepancies and false positives/negatives

4. **Calibration Testing**
   - Adjust scoring thresholds if needed
   - Test edge cases and boundary conditions
   - Validate improvement suggestions

**Pass Criteria**:
- [ ] Automated scores within ±0.10 of expert review
- [ ] Constitutional violations correctly identified
- [ ] Improvement suggestions are actionable
- [ ] No false positives for compliant artifacts

### Manual Test 4.2: Constitutional Enforcement Effectiveness

**Objective**: Test how well constitutional framework improves development quality

**Test Steps**:

1. **Baseline Measurement**
   - Create specification without constitutional framework
   - Generate plan and code using traditional methods
   - Document quality metrics and issues

2. **Constitutional Enhancement**
   - Enable constitutional framework
   - Recreate same specification with constitutional guidance
   - Generate plan and code with constitutional constraints

3. **Quality Comparison**
   ```
   Metrics to compare:
   - Specification clarity and completeness
   - Technology choice justification
   - Test coverage and quality
   - Code complexity and maintainability
   - Documentation currency
   ```

4. **User Experience Assessment**
   - Time to complete workflow
   - Cognitive load and ease of use
   - Quality of generated artifacts
   - Learning curve for new users

**Pass Criteria**:
- [ ] Constitutional framework improves artifact quality
- [ ] Development time remains reasonable
- [ ] User experience is positive
- [ ] Quality improvements are measurable

## CLI Tool Manual Testing

### Manual Test 5.1: CLI Constitutional Commands

**Objective**: Validate CLI tool constitutional integration and usability

**Test Steps**:

1. **Basic CLI Functionality**
   ```bash
   # Test all constitutional commands
   sdd specify "Build a simple calculator" --constitutional
   sdd plan calculator-spec.md --constitutional-gates
   sdd tasks calculator-plan.md --constitutional
   sdd validate calculator-spec.md --strict
   ```

2. **Constitutional Reporting**
   ```bash
   # Test constitutional reporting features
   sdd constitutional-report --project-wide
   sdd constitutional-trends --time-range 30d
   sdd constitutional-violations --severity high
   ```

3. **Cross-Platform Testing**
   - Test on Windows, macOS, Linux
   - Verify constitutional scoring consistency
   - Check file path handling and Git integration

4. **Error Handling and Recovery**
   - Test with invalid inputs
   - Verify constitutional violation handling
   - Check graceful degradation scenarios

**Pass Criteria**:
- [ ] All CLI commands execute correctly
- [ ] Constitutional scoring is consistent
- [ ] Error messages are helpful and actionable
- [ ] Cross-platform compatibility maintained

## Performance and Scale Manual Testing

### Manual Test 6.1: Large Project Constitutional Validation

**Objective**: Test constitutional framework performance with large codebases

**Test Steps**:

1. **Large Repository Setup**
   - Clone large open-source project (>10k files)
   - Initialize SDD constitutional framework
   - Configure for incremental validation

2. **Full Project Validation**
   ```bash
   # Time these operations:
   time sdd validate --type all --recursive ./
   time sdd constitutional-report --detailed
   time sdd constitutional-violations --all
   ```

3. **Performance Monitoring**
   - Monitor memory usage during validation
   - Check CPU utilization patterns
   - Measure constitutional scoring performance

4. **Incremental Validation Testing**
   - Make small changes to files
   - Test incremental constitutional validation
   - Verify change impact analysis

**Pass Criteria**:
- [ ] Validation completes within reasonable time (≤5 minutes for large projects)
- [ ] Memory usage remains acceptable (≤2GB)
- [ ] Incremental validation is faster than full validation
- [ ] Constitutional scoring scales linearly

## Documentation and Training Manual Tests

### Manual Test 7.1: Constitutional Framework Documentation

**Objective**: Validate constitutional framework documentation completeness and usability

**Test Steps**:

1. **New User Onboarding**
   - Test documentation with new user
   - Follow setup instructions step-by-step
   - Document confusion points and gaps

2. **Constitutional Principles Explanation**
   - Review explanation of each constitutional article
   - Test examples and counterexamples
   - Validate practical application guidance

3. **Troubleshooting Documentation**
   - Test common issue resolution
   - Verify constitutional violation explanations
   - Check improvement suggestion documentation

4. **Integration Documentation**
   - Test VS Code extension setup instructions
   - Validate CLI tool configuration guide
   - Check API integration documentation

**Pass Criteria**:
- [ ] New users can successfully set up SDD
- [ ] Constitutional principles are clearly explained
- [ ] Troubleshooting documentation resolves common issues
- [ ] Integration guides are complete and accurate

## Test Execution Tracking

### Test Session Template

```
=== SDD Constitutional Manual Testing Session ===

Tester Information:
- Name: _________________________
- Role: _________________________
- Experience Level: ______________
- Date: _________________________

Environment:
- OS: ___________________________
- VS Code Version: ______________
- Python Version: _______________
- SDD Version: __________________

Test Results Summary:
- Tests Executed: _______________
- Tests Passed: _________________
- Tests Failed: _________________
- Constitutional Score Range: ____

Critical Issues Found:
1. ____________________________
2. ____________________________
3. ____________________________

Constitutional Framework Feedback:
- Effectiveness Rating (1-10): ___
- Usability Rating (1-10): ______
- Documentation Rating (1-10): ___

Recommendations:
_________________________________
_________________________________
_________________________________

Tester Signature: ________________
```

### Test Completion Checklist

#### Pre-Release Manual Testing
- [ ] All Epic tests completed and passed
- [ ] Constitutional validation accuracy verified
- [ ] CLI tool functionality validated
- [ ] VS Code extension integration tested
- [ ] Performance and scale testing completed
- [ ] Documentation accuracy verified
- [ ] Cross-platform compatibility confirmed
- [ ] Real-world project integration validated

#### Post-Release Monitoring
- [ ] User feedback collected and analyzed
- [ ] Constitutional framework effectiveness measured
- [ ] Performance metrics monitored
- [ ] Quality improvements documented
- [ ] Training materials updated based on findings

---

## Constitutional Compliance Review

### Article IV: Integration-First Testing ✅
- All manual tests use realistic environments
- Real AI APIs and actual development workflows
- Cross-system integration validation
- Production-like scenarios tested

### Supporting Articles ✅
- **Article I**: Manual tests validate library-first approach
- **Article II**: Test-first development workflows verified
- **Article III**: Simplicity of user experience tested
- **Article V**: Clear test procedures and criteria
- **Article VI**: Alternative testing approaches documented

**Manual Testing Constitutional Score**: 0.94 ✅

---

*These manual testing procedures ensure constitutional compliance and real-world effectiveness of the SDD framework through comprehensive user experience validation.*
