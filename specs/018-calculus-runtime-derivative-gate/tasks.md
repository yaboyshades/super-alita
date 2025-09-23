# Tasks: Calculus Runtime Derivative Gate

**Input**: Design documents from `/specs/018-calculus-runtime-derivative-gate/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/, quickstart.md

## Summary

Implement an automated calculus-based runtime analysis gate that samples target functions, fits smooth runtime curves, derives first/second derivatives, and fails CI/MCP workflows when derivative thresholds spike. Built as a Python library with CLI and MCP integration.

## Technical Context

**Language**: Python 3.11
**Dependencies**: numpy, scipy, matplotlib, rich, pytest, hypothesis
**Structure**: Single project with `src/calculus_gate/` library + CLI + MCP integration
**Testing**: pytest with property-based tests (Hypothesis), TDD enforced

## Phase 3.1: Setup & Dependencies

- [ ] T001 Create calculus gate project structure (`src/calculus_gate/`, `artifacts/calculus_gate/`, test directories)
- [ ] T002 Install and configure numerical dependencies (numpy, scipy, matplotlib, rich, pytest, hypothesis)
- [ ] T003 [P] Configure linting and type checking for mathematical code standards
- [ ] T004 [P] Set up property-based testing framework with Hypothesis for mathematical validation

## Phase 3.2: Contract Tests (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [ ] T005 [P] Contract test for certificate JSON schema validation in `tests/contract/test_certificate_schema.py`
- [ ] T006 [P] Contract test for MCP response schema compliance in `tests/contract/test_mcp_response_schema.py`
- [ ] T007 [P] Contract test for CLI output format specification in `tests/contract/test_cli_output_format.py`
- [ ] T008 [P] Property-based test for derivative mathematical correctness in `tests/property/test_derivative_properties.py`
- [ ] T009 [P] Property-based test for monotonicity and convexity validation in `tests/property/test_mathematical_properties.py`
- [ ] T010 [P] Integration test for complete analysis workflow in `tests/integration/test_calculus_gate_workflow.py`
- [ ] T011 [P] Integration test for CLI command execution and artifacts in `tests/integration/test_cli_integration.py`
- [ ] T012 [P] Integration test for MCP endpoint analysis requests in `tests/integration/test_mcp_integration.py`

## Phase 3.3: Core Data Models (ONLY after tests are failing)

- [ ] T013 [P] TargetFunction data model in `src/calculus_gate/models.py`
- [ ] T014 [P] RuntimeSampleSet data model in `src/calculus_gate/models.py`
- [ ] T015 [P] DerivativeCertificate data model in `src/calculus_gate/models.py`
- [ ] T016 [P] AlertEvent data model in `src/calculus_gate/models.py`
- [ ] T017 [P] JSON serialization/deserialization for all models in `src/calculus_gate/serialization.py`

## Phase 3.4: Core Algorithm Implementation

- [ ] T018 RuntimeProfiler class for sampling in `src/calculus_gate/sampling.py`
- [ ] T019 Exponential input size generation with validation in `src/calculus_gate/sampling.py`
- [ ] T020 Memory tracking and warmup run support in `src/calculus_gate/sampling.py`
- [ ] T021 CalculusAnalyzer class for derivative computation in `src/calculus_gate/fitting.py`
- [ ] T022 CubicSpline fitting with input validation in `src/calculus_gate/fitting.py`
- [ ] T023 Finite difference derivative calculation in `src/calculus_gate/fitting.py`
- [ ] T024 Savitzky-Golay fallback for noisy data in `src/calculus_gate/fitting.py`
- [ ] T025 Lipschitz constant computation in `src/calculus_gate/fitting.py`

## Phase 3.5: Certificate & Analysis Engine

- [ ] T026 PerformanceCertificate generation logic in `src/calculus_gate/certificate.py`
- [ ] T027 Bootstrap confidence interval calculation in `src/calculus_gate/certificate.py`
- [ ] T028 Certificate grading system (A/B/F) in `src/calculus_gate/certificate.py`
- [ ] T029 Threshold violation detection in `src/calculus_gate/certificate.py`
- [ ] T030 Historical baseline comparison in `src/calculus_gate/certificate.py`
- [ ] T031 Main analysis orchestration function in `src/calculus_gate/__init__.py`

## Phase 3.6: Reporting & Output

- [ ] T032 [P] Rich console reporter for human-readable output in `src/calculus_gate/reporters/console.py`
- [ ] T033 [P] JSON artifact reporter for CI integration in `src/calculus_gate/reporters/json_reporter.py`
- [ ] T034 [P] Alert event emission for CI/MCP workflows in `src/calculus_gate/reporters/alert_reporter.py`
- [ ] T035 [P] Optional plot generation using matplotlib in `src/calculus_gate/reporters/plot_reporter.py`

## Phase 3.7: CLI Interface

- [ ] T036 CLI argument parsing and validation in `src/calculus_gate/cli.py`
- [ ] T037 Configuration management for thresholds and sampling in `src/calculus_gate/cli.py`
- [ ] T038 Developer-friendly error messages and help text in `src/calculus_gate/cli.py`
- [ ] T039 Integration with unified quality gates system in `src/calculus_gate/cli.py`
- [ ] T040 CLI main entry point and workflow orchestration in `src/calculus_gate/cli.py`

## Phase 3.8: MCP Integration

- [ ] T041 [P] MCP endpoint for real-time calculus analysis in `src/mcp/calculus_server.py`
- [ ] T042 [P] MCP response formatting and error handling in `src/mcp/calculus_server.py`
- [ ] T043 [P] MCP streaming analysis capabilities in `src/mcp/calculus_server.py`
- [ ] T044 [P] Integration with existing MCP server infrastructure in `src/mcp/calculus_server.py`

## Phase 3.9: Error Handling & Edge Cases

- [ ] T045 Edge case handling for insufficient data points in `src/calculus_gate/exceptions.py`
- [ ] T046 Graceful degradation for curve fitting failures in `src/calculus_gate/exceptions.py`
- [ ] T047 Timeout and resource limit enforcement in `src/calculus_gate/exceptions.py`
- [ ] T048 Input validation and sanitization throughout system in `src/calculus_gate/validation.py`

## Phase 3.10: Polish & Optimization

- [ ] T049 [P] Unit tests for edge cases and error conditions in `tests/unit/test_edge_cases.py`
- [ ] T050 [P] Performance benchmarks and resource usage tests in `tests/performance/test_benchmarks.py`
- [ ] T051 [P] Mathematical accuracy validation tests in `tests/validation/test_mathematical_accuracy.py`
- [ ] T052 [P] Update project README with calculus gate documentation in `README.md`
- [ ] T053 [P] Create developer guide for extending the system in `docs/developer_guide.md`
- [ ] T054 [P] Add troubleshooting guide for common issues in `docs/troubleshooting.md`
- [ ] T055 Remove code duplication and optimize performance bottlenecks
- [ ] T056 Final integration validation using quickstart scenarios

## Dependencies

### Critical Path
```
Setup (T001-T004) → Contract Tests (T005-T012) → Data Models (T013-T017) →
Core Algorithms (T018-T025) → Certificate Engine (T026-T031) →
Reporting (T032-T035) → CLI (T036-T040) → Final Validation (T056)
```

### Parallel Groups
- **Group A**: T005-T012 (Contract & Property Tests - different files)
- **Group B**: T013-T016 (Data Models - can be implemented in parallel)
- **Group C**: T032-T035 (Reporters - independent implementations)
- **Group D**: T041-T044 (MCP Integration - separate from CLI)
- **Group E**: T049-T054 (Polish tasks - independent documentation and testing)

### Blocking Dependencies
- T018-T025 require T013-T017 (algorithms need data models)
- T026-T031 require T018-T025 (certificate engine needs algorithms)
- T036-T040 require T026-T031 (CLI needs complete analysis engine)
- T049-T056 require all implementation tasks complete

## Parallel Execution Examples

### Phase 3.2 (Contract Tests)
```bash
# All contract tests can run simultaneously - different files
pytest tests/contract/test_certificate_schema.py &
pytest tests/contract/test_mcp_response_schema.py &
pytest tests/contract/test_cli_output_format.py &
pytest tests/property/test_derivative_properties.py &
pytest tests/property/test_mathematical_properties.py &
wait
```

### Phase 3.3 (Data Models)
```bash
# Data models can be implemented in parallel within same file
# Implementation approach: Create separate functions/classes first, then combine
```

### Phase 3.6 (Reporters)
```bash
# Each reporter can be implemented independently
Task: "Rich console reporter in src/calculus_gate/reporters/console.py"
Task: "JSON artifact reporter in src/calculus_gate/reporters/json_reporter.py"
Task: "Alert event emission in src/calculus_gate/reporters/alert_reporter.py"
Task: "Plot generation in src/calculus_gate/reporters/plot_reporter.py"
```

## Validation Checklist

### Contract Coverage
- [x] Certificate JSON schema test (T005)
- [x] MCP response schema test (T006)
- [x] CLI output format test (T007)

### Entity Coverage
- [x] TargetFunction model (T013)
- [x] RuntimeSampleSet model (T014)
- [x] DerivativeCertificate model (T015)
- [x] AlertEvent model (T016)

### Integration Coverage
- [x] Complete workflow test (T010)
- [x] CLI integration test (T011)
- [x] MCP integration test (T012)

### TDD Compliance
- [x] All tests written before implementation
- [x] Tests must fail initially with meaningful errors
- [x] Implementation only proceeds after failing tests exist

## Constitutional Compliance

### Article I (Library-First)
- Uses existing SciPy/NumPy libraries for mathematical computation
- Integrates with existing MCP and unified gates infrastructure
- Reuses established testing patterns (pytest, Hypothesis)

### Article II (Test-First)
- Comprehensive contract tests before any implementation (T005-T012)
- Property-based tests for mathematical correctness
- Integration tests covering all user scenarios

### Article III (Simplicity)
- Single focused library for calculus-based analysis
- Clear separation of concerns (sampling, fitting, reporting)
- Minimal abstractions with mathematical precision

### Article XXI (Calculus Gate)
- Implements mathematical quality gates as constitutional requirement
- Provides runtime derivative analysis with formal bounds
- Ensures performance stability through mathematical validation

## Notes

- **[P] tasks**: Different files, no dependencies, can run in parallel
- **TDD enforcement**: All T005-T012 must fail before starting T013+
- **Mathematical precision**: Property-based tests validate algorithm correctness
- **Performance**: Target <60s local analysis, <5min CI analysis per function
- **Integration**: Works with existing quality gates (mutation, CFG, SyGuS)
- **Artifacts**: JSON certificates saved to `artifacts/calculus_gate/` for CI/MCP consumption
