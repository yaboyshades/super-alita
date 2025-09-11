# Contract Tests Specification

**Document**: 06-contract-tests.md
**Constitutional Article**: II - Test-First Development & IV - Integration-First Testing
**Last Updated**: September 10, 2025

## Constitutional Testing Approach

This document follows Articles II and IV of the Super-Alita Constitutional Framework:
- **Article II**: Write tests before implementation, ensure comprehensive coverage
- **Article IV**: Validate system interactions in realistic environments

## Testing Philosophy

### Test-First Development (Article II)
All contract tests MUST be written and validated BEFORE implementation begins. This ensures:

1. **Clear Interface Definition**: Tests define expected behavior precisely
2. **Constitutional Compliance**: Tests validate constitutional requirements
3. **Regression Prevention**: Changes cannot break existing contracts
4. **Documentation**: Tests serve as executable specifications

### Integration-First Testing (Article IV)
All tests MUST use realistic environments:

1. **Real AI APIs**: Test with actual Claude, OpenAI, Gemini endpoints
2. **Actual File Systems**: Test with real Git repositories and file operations
3. **Live Dependencies**: Test with real FastAPI servers, VS Code instances
4. **Constitutional Validation**: Test against actual constitutional framework

## SDD Core Engine Contract Tests

### 1. Specification Processing Contracts

#### Test: POST /specify - Valid Requirements
```python
@pytest.mark.integration
@pytest.mark.constitutional
async def test_specify_valid_requirements():
    """Contract test for specification generation with valid input."""

    # Arrange - Constitutional test data
    user_input = "Build a real-time chat system with message persistence"
    request_payload = {
        "user_input": user_input,
        "context": {
            "constitutional_mode": True
        }
    }

    # Act - Call actual API
    response = await client.post("/api/v1/specify", json=request_payload)

    # Assert - Constitutional compliance
    assert response.status_code == 200
    data = response.json()

    # Verify specification structure
    assert "specification" in data
    spec = data["specification"]
    assert "id" in spec
    assert "title" in spec
    assert "user_stories" in spec
    assert len(spec["user_stories"]) > 0

    # Verify constitutional compliance (Article II requirement)
    assert "constitutional_compliance" in spec
    compliance = spec["constitutional_compliance"]
    assert "score" in compliance
    assert compliance["score"] >= 0.75  # Constitutional threshold

    # Verify all six articles are scored
    article_scores = compliance["article_scores"]
    required_articles = [
        "library_first", "test_first", "simplicity_gate",
        "integration_first", "clarity_unambiguity", "counterfactual_justification"
    ]
    for article in required_articles:
        assert article in article_scores
        assert 0.0 <= article_scores[article] <= 1.0

    # Verify user stories follow constitutional format
    for story in spec["user_stories"]:
        assert "as_a" in story
        assert "i_want" in story
        assert "so_that" in story
        assert "acceptance_criteria" in story
        assert len(story["acceptance_criteria"]) > 0
```

#### Test: POST /specify - Constitutional Violations
```python
@pytest.mark.integration
@pytest.mark.constitutional
async def test_specify_constitutional_violations():
    """Contract test for handling constitutional violations."""

    # Arrange - Input that violates constitutional principles
    user_input = "Create a complex, over-engineered system with multiple abstraction layers"
    request_payload = {
        "user_input": user_input,
        "context": {"constitutional_mode": True}
    }

    # Act
    response = await client.post("/api/v1/specify", json=request_payload)

    # Assert - Should still succeed but flag violations
    assert response.status_code == 200
    data = response.json()

    # Should have constitutional violations
    compliance = data["specification"]["constitutional_compliance"]
    assert compliance["score"] < 0.75  # Below threshold
    assert len(compliance["violations"]) > 0

    # Violations should reference Article III (Simplicity Gate)
    violation_articles = [v["article"] for v in compliance["violations"]]
    assert "simplicity_gate" in violation_articles
```

#### Test: POST /specify - Missing Required Data
```python
@pytest.mark.integration
async def test_specify_missing_user_input():
    """Contract test for missing required parameters."""

    # Arrange - Invalid payload
    request_payload = {"context": {"constitutional_mode": True}}

    # Act
    response = await client.post("/api/v1/specify", json=request_payload)

    # Assert - Should return 400 Bad Request
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "user_input" in data["error"]["message"].lower()
```

### 2. Implementation Planning Contracts

#### Test: POST /plan - Valid Specification
```python
@pytest.mark.integration
@pytest.mark.constitutional
async def test_plan_valid_specification():
    """Contract test for implementation plan generation."""

    # Arrange - Create specification first (dependency)
    spec_response = await client.post("/api/v1/specify", json={
        "user_input": "Build a CLI tool for file processing",
        "context": {"constitutional_mode": True}
    })
    spec_id = spec_response.json()["specification"]["id"]

    # Act - Generate plan
    plan_request = {
        "specification_id": spec_id,
        "technology_preferences": {
            "languages": ["python"],
            "frameworks": ["click"],
            "constraints": ["cli_interface_mandatory"]
        },
        "constitutional_mode": True
    }
    response = await client.post("/api/v1/plan", json=plan_request)

    # Assert - Constitutional plan structure
    assert response.status_code == 200
    data = response.json()

    plan = data["implementation_plan"]
    assert "id" in plan
    assert "specification_id" in plan
    assert plan["specification_id"] == spec_id

    # Verify phases structure
    assert "phases" in plan
    assert len(plan["phases"]) > 0

    for phase in plan["phases"]:
        assert "id" in phase
        assert "name" in phase
        assert "duration_weeks" in phase
        assert "constitutional_gates" in phase

        # Verify constitutional gates (Article IV requirement)
        for gate in phase["constitutional_gates"]:
            assert "article" in gate
            assert "criteria" in gate
            assert "validation_method" in gate

    # Verify technology stack includes constitutional justifications
    tech_stack = plan["technology_stack"]
    assert "justifications" in tech_stack
    assert "library_choice" in tech_stack["justifications"]

    # Verify CLI interface mandate (Article from spec-kit constitution)
    libraries = tech_stack["libraries"]
    assert any("click" in lib.lower() or "cli" in lib.lower() for lib in libraries)
```

#### Test: POST /plan - Constitutional Gate Validation
```python
@pytest.mark.integration
@pytest.mark.constitutional
async def test_plan_constitutional_gates():
    """Contract test for constitutional gate enforcement."""

    # Arrange - Specification that should trigger simplicity constraints
    spec_response = await client.post("/api/v1/specify", json={
        "user_input": "Build a microservices architecture with multiple databases",
        "context": {"constitutional_mode": True}
    })
    spec_id = spec_response.json()["specification"]["id"]

    # Act
    response = await client.post("/api/v1/plan", json={
        "specification_id": spec_id,
        "constitutional_mode": True
    })

    # Assert - Should flag complexity violations
    assert response.status_code == 200
    plan = response.json()["implementation_plan"]

    # Should have constitutional compliance warnings
    compliance = plan["constitutional_compliance"]
    gate_validations = compliance["gate_validations"]

    # Find simplicity gate validation
    simplicity_gate = next(
        (g for g in gate_validations if "simplicity" in g["gate"].lower()),
        None
    )
    assert simplicity_gate is not None
    # Should fail or have low score due to complexity
    assert simplicity_gate["score"] < 0.8
```

### 3. Task Breakdown Contracts

#### Test: POST /tasks - Valid Implementation Plan
```python
@pytest.mark.integration
@pytest.mark.constitutional
async def test_tasks_valid_plan():
    """Contract test for task breakdown generation."""

    # Arrange - Create spec and plan (dependencies)
    spec_response = await client.post("/api/v1/specify", json={
        "user_input": "Build a simple web API",
        "context": {"constitutional_mode": True}
    })
    spec_id = spec_response.json()["specification"]["id"]

    plan_response = await client.post("/api/v1/plan", json={
        "specification_id": spec_id,
        "constitutional_mode": True
    })
    plan_id = plan_response.json()["implementation_plan"]["id"]

    # Act
    response = await client.post("/api/v1/tasks", json={
        "plan_id": plan_id,
        "constitutional_mode": True
    })

    # Assert
    assert response.status_code == 200
    data = response.json()

    breakdown = data["task_breakdown"]
    assert "plan_id" in breakdown
    assert breakdown["plan_id"] == plan_id

    # Verify tasks structure
    assert "tasks" in breakdown
    tasks = breakdown["tasks"]
    assert len(tasks) > 0

    for task in tasks:
        assert "id" in task
        assert "title" in task
        assert "description" in task
        assert "constitutional_requirements" in task

        # Verify constitutional requirements (Article II compliance)
        const_reqs = task["constitutional_requirements"]
        assert len(const_reqs) > 0

        for req in const_reqs:
            assert "article" in req
            assert "requirement" in req
            assert "validation_method" in req

    # Verify constitutional validation
    validation = breakdown["constitutional_validation"]
    assert "overall_compliance" in validation
    assert validation["overall_compliance"] >= 0.75
```

## Constitutional Validation Contract Tests

### 1. Artifact Validation Contracts

#### Test: POST /constitutional/validate - Code Validation
```python
@pytest.mark.integration
@pytest.mark.constitutional
async def test_constitutional_validate_code():
    """Contract test for code constitutional validation."""

    # Arrange - Code that violates constitutional principles
    code_artifact = {
        "type": "code",
        "content": '''
def very_long_function_that_violates_simplicity_gate():
    # This function is intentionally long to test Article III
    result = []
    for i in range(100):
        for j in range(100):
            for k in range(100):
                if i % 2 == 0:
                    if j % 3 == 0:
                        if k % 5 == 0:
                            result.append(i + j + k)
                        else:
                            result.append(i * j * k)
                    else:
                        result.append(i - j - k)
                else:
                    result.append(i + j - k)
    return result
        ''',
        "metadata": {"language": "python"}
    }

    request_payload = {
        "artifact": code_artifact,
        "validation_options": {"strict_mode": True}
    }

    # Act
    response = await client.post("/api/v1/constitutional/validate", json=request_payload)

    # Assert
    assert response.status_code == 200
    data = response.json()

    validation = data["validation_result"]
    assert "overall_score" in validation
    assert "passed" in validation
    assert validation["passed"] is False  # Should fail due to complexity

    # Should flag Article III violations
    simplicity_score = validation["article_scores"]["simplicity_gate"]
    assert simplicity_score["score"] < 0.5  # Low score for complex code
    assert "violations" in simplicity_score
    assert len(simplicity_score["violations"]) > 0
```

#### Test: POST /constitutional/validate - Specification Validation
```python
@pytest.mark.integration
@pytest.mark.constitutional
async def test_constitutional_validate_specification():
    """Contract test for specification constitutional validation."""

    # Arrange - Well-formed specification
    spec_artifact = {
        "type": "specification",
        "content": '''
# Feature: Simple Calculator
As a user, I want to perform basic arithmetic operations
So that I can solve simple math problems quickly.

## Library Research
This feature will use the built-in Python math operators (+, -, *, /).
No external libraries required, following Article I principles.

## Test Requirements
- Unit tests for each arithmetic operation
- Integration tests for calculator interface
- 80% minimum code coverage (Article II)

## Simplicity Constraints
- Maximum 50 lines per function (Article III)
- Single responsibility per function
- No complex abstractions

## Acceptance Criteria
- [x] Addition function works correctly
- [x] Subtraction function works correctly
- [x] Multiplication function works correctly
- [x] Division function works correctly with zero handling
        '''
    }

    # Act
    response = await client.post("/api/v1/constitutional/validate", json={
        "artifact": spec_artifact
    })

    # Assert
    assert response.status_code == 200
    validation = response.json()["validation_result"]

    # Should pass constitutional validation
    assert validation["overall_score"] >= 0.75
    assert validation["passed"] is True

    # All articles should have reasonable scores
    for article, score_data in validation["article_scores"].items():
        assert score_data["score"] >= 0.7
```

## APE Engine Contract Tests

### 1. Prompt Optimization Contracts

#### Test: POST /ape/optimize - Constitutional Optimization
```python
@pytest.mark.integration
@pytest.mark.constitutional
async def test_ape_optimize_constitutional():
    """Contract test for constitutional prompt optimization."""

    # Arrange - Basic prompt that needs constitutional enhancement
    request_payload = {
        "base_prompt": "Build a web application",
        "optimization_target": "constitutional_compliance",
        "context": {
            "domain": "web_development",
            "constraints": ["library_first", "test_first"]
        }
    }

    # Act
    response = await client.post("/api/v1/ape/optimize", json=request_payload)

    # Assert
    assert response.status_code == 200
    data = response.json()

    optimized = data["optimized_prompt"]
    assert "content" in optimized
    assert "constitutional_score" in optimized
    assert optimized["constitutional_score"] > 0.75

    # Optimized prompt should mention constitutional principles
    content = optimized["content"].lower()
    assert any(term in content for term in [
        "library", "test", "simple", "integrate", "clear", "justify"
    ])

    # Should have constitutional variations
    assert "variations" in optimized
    variations = optimized["variations"]
    assert len(variations) >= 3  # Multiple constitutional approaches

    for variation in variations:
        assert "focus" in variation
        assert "content" in variation
        assert "score" in variation
```

## VS Code Extension Contract Tests

### 1. Command Interface Contracts

#### Test: alita.sdd.specify Command
```typescript
import * as vscode from 'vscode';
import { expect } from 'chai';

describe('SDD Commands Contract Tests', () => {
    let extension: vscode.Extension<any>;

    before(async () => {
        // Load extension in test environment
        extension = vscode.extensions.getExtension('alita.sdd-extension')!;
        await extension.activate();
    });

    it('should execute specify command with constitutional validation', async () => {
        // Arrange
        const userInput = 'Build a file processor tool';
        const args = {
            userInput,
            constitutionalMode: true
        };

        // Act
        const result = await vscode.commands.executeCommand(
            'alita.sdd.specify',
            args
        ) as SpecifyResult;

        // Assert - Constitutional contract
        expect(result).to.have.property('specificationId');
        expect(result).to.have.property('filePath');
        expect(result).to.have.property('constitutionalScore');
        expect(result.constitutionalScore).to.be.at.least(0.75);

        // Verify file was created
        const fileExists = await vscode.workspace.fs.stat(
            vscode.Uri.file(result.filePath)
        );
        expect(fileExists).to.exist;
    });

    it('should validate constitutional violations in real-time', async () => {
        // Arrange - Create document with violations
        const document = await vscode.workspace.openTextDocument({
            content: 'Build a complex system with many abstraction layers',
            language: 'markdown'
        });

        // Act
        const result = await vscode.commands.executeCommand(
            'alita.constitutional.validate',
            { documentPath: document.uri.fsPath }
        ) as ValidationResult;

        // Assert
        expect(result.overallScore).to.be.below(0.75);
        expect(result.passed).to.be.false;
        expect(result.violations).to.have.length.greaterThan(0);
    });
});
```

## CLI Tool Contract Tests

### 1. Command-Line Interface Contracts

#### Test: sdd specify CLI
```python
@pytest.mark.integration
@pytest.mark.cli
def test_cli_specify_command():
    """Contract test for CLI specify command."""

    import subprocess
    import json
    import tempfile
    import os

    # Arrange
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        output_file = f.name

    try:
        # Act - Execute CLI command
        result = subprocess.run([
            'sdd', 'specify',
            '--output-file', output_file,
            '--output-format', 'json',
            'Build a simple web API with authentication'
        ], capture_output=True, text=True, timeout=60)

        # Assert - Command execution
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Verify output file exists and has content
        assert os.path.exists(output_file)

        with open(output_file, 'r') as f:
            content = f.read()

        # Should be valid JSON (as requested)
        data = json.loads(content)

        # Verify constitutional structure
        assert 'constitutional_compliance' in data
        assert data['constitutional_compliance']['score'] >= 0.75

    finally:
        # Cleanup
        if os.path.exists(output_file):
            os.unlink(output_file)
```

#### Test: sdd validate CLI
```python
@pytest.mark.integration
@pytest.mark.cli
def test_cli_validate_command():
    """Contract test for CLI validate command."""

    import subprocess
    import tempfile
    import os

    # Arrange - Create test specification file
    spec_content = '''
# Test Specification
Build a calculator with basic operations.
Uses built-in Python operators (Article I compliance).
Includes comprehensive tests (Article II compliance).
    '''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(spec_content)
        spec_file = f.name

    try:
        # Act
        result = subprocess.run([
            'sdd', 'validate',
            '--type', 'specification',
            '--target-score', '0.75',
            spec_file
        ], capture_output=True, text=True, timeout=30)

        # Assert
        assert result.returncode == 0

        # Output should contain constitutional scores
        output = result.stdout
        assert 'constitutional' in output.lower()
        assert 'score' in output.lower()

    finally:
        os.unlink(spec_file)
```

## Test Environment Setup

### 1. Integration Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
import httpx
from fastapi.testclient import TestClient
from src.main import create_app

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def app():
    """Create test application instance."""
    return create_app(testing=True, constitutional_mode=True)

@pytest.fixture(scope="session")
async def client(app):
    """Create test client for API calls."""
    async with httpx.AsyncClient(
        app=app,
        base_url="http://test",
        timeout=60.0  # Allow time for AI API calls
    ) as client:
        yield client

@pytest.fixture
def constitutional_config():
    """Constitutional framework test configuration."""
    return {
        "enforcement_level": "strict",
        "compliance_threshold": 0.75,
        "enable_all_articles": True,
        "real_ai_apis": True  # Use actual AI APIs for integration tests
    }
```

### 2. Mock vs Real Services

#### Real Services (Article IV: Integration-First)
- **AI APIs**: Use actual OpenAI, Claude, Gemini endpoints with test API keys
- **File System**: Use real temporary directories and Git repositories
- **VS Code**: Use actual VS Code instances in headless mode for extension tests

#### Test Data Management
```python
# tests/test_data.py
CONSTITUTIONAL_TEST_CASES = {
    "high_compliance": {
        "input": "Build a simple file reader using Python's built-in functions",
        "expected_score": 0.90,
        "expected_articles": ["library_first", "simplicity_gate"]
    },
    "low_compliance": {
        "input": "Create a complex microservices architecture with multiple abstraction layers",
        "expected_score": 0.40,
        "violations": ["simplicity_gate", "anti_abstraction"]
    },
    "missing_tests": {
        "input": "Build an API without mentioning testing requirements",
        "expected_score": 0.60,
        "violations": ["test_first"]
    }
}
```

## Constitutional Test Metrics

### 1. Coverage Requirements
- **API Contract Coverage**: 100% of all endpoints tested
- **Constitutional Article Coverage**: 100% of all six articles validated
- **Error Scenario Coverage**: 90% of error conditions tested
- **Integration Path Coverage**: 80% of end-to-end workflows tested

### 2. Performance Benchmarks
- **API Response Time**: <2 seconds for validation endpoints (95th percentile)
- **Constitutional Scoring**: <5 seconds for typical specifications
- **CLI Command Execution**: <30 seconds for standard operations
- **VS Code Extension**: <1 second for command registration

### 3. Quality Gates
- **Test Pass Rate**: 100% for contract tests
- **Constitutional Compliance**: ≥0.90 for all test artifacts
- **Real Integration Success**: ≥95% success rate with actual AI APIs
- **Documentation Coverage**: 100% of contracts documented and tested

---

## Constitutional Compliance Review

### Article II: Test-First Development ✅
- All contracts defined before implementation
- Comprehensive test coverage for all scenarios
- Constitutional compliance testing mandatory

### Article IV: Integration-First Testing ✅
- Real AI API integration testing
- Actual file system and Git operations
- Live dependency validation

### Supporting Articles ✅
- **Article I**: Tests use established testing frameworks
- **Article III**: Simple, focused test design
- **Article V**: Clear, unambiguous test specifications
- **Article VI**: Alternative testing approaches documented

**Contract Test Constitutional Score**: 0.96 ✅

---

*These contract tests ensure constitutional compliance and integration-first validation for all SDD framework components.*
