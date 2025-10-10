#!/usr/bin/env python3
"""
Comprehensive Validation Checklist for Unified Intelligence Layer

This module provides a complete validation framework for the Unified Intelligence Layer,
ensuring all constitutional, contract, and performance gates are validated according
to the architectural critique requirements.

Validation Categories:
1. Contract-First Interfaces: Schema compliance and type safety
2. Score Fusion Math: Mathematical correctness of fusion algorithms
3. Explicit Failure Semantics: Proper error handling and propagation
4. Canonical Orchestration: Workflow consistency and reliability
5. Golden Test Fixtures: Test data integrity and coverage
6. Telemetry Schema: Event structure and metadata compliance
7. Constitutional Compliance: Adherence to SDD and quality gates
8. Performance Gates: Runtime efficiency and resource usage
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(__file__))

from pydantic import ValidationError


@dataclass
class ValidationResult:
    """Result of a validation check."""

    category: str
    check_name: str
    passed: bool
    details: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    remediation: str | None = None
    metrics: dict[str, Any] | None = None


class ValidationChecklist:
    """Comprehensive validation checklist for Unified Intelligence Layer."""

    def __init__(self):
        self.results: list[ValidationResult] = []
        self.start_time = time.time()

    async def run_full_validation(self) -> dict[str, Any]:
        """Run complete validation suite."""
        print(
            "🔍 Starting comprehensive validation of Unified Intelligence Layer..."
        )

        # Contract-First Interfaces
        await self._validate_contract_interfaces()

        # Score Fusion Math
        await self._validate_score_fusion_math()

        # Explicit Failure Semantics
        await self._validate_failure_semantics()

        # Canonical Orchestration
        await self._validate_canonical_orchestration()

        # Golden Test Fixtures
        await self._validate_golden_fixtures()

        # Telemetry Schema
        await self._validate_telemetry_schema()

        # Constitutional Compliance
        await self._validate_constitutional_compliance()

        # Performance Gates
        await self._validate_performance_gates()

        # Code Reasoning Validation
        await self._validate_code_reasoning()

        return self._generate_report()

    async def _validate_contract_interfaces(self):
        """Validate contract-first interfaces and schema compliance."""
        print("📋 Validating contract-first interfaces...")

        try:
            from contracts import (
                ConstitutionResult,
                CopilotEnhancement,
                FusionConfig,
                MangleResult,
                TelemetryHeaders,
                UnifiedAdvice,
                WorkflowResult,
            )

            # Test schema validation
            test_cases = [
                (
                    MangleResult,
                    {
                        "confidence": 0.85,
                        "patterns": ["test"],
                        "reasoning": "test",
                    },
                ),
                (
                    ConstitutionResult,
                    {
                        "compliance_score": 0.92,
                        "violations": [],
                        "recommendations": [],
                    },
                ),
                (
                    WorkflowResult,
                    {
                        "detected_patterns": ["planning"],
                        "confidence": 0.78,
                        "metadata": {},
                    },
                ),
                (
                    CopilotEnhancement,
                    {
                        "enhancements": [],
                        "confidence": 0.65,
                        "reasoning": "test",
                    },
                ),
                (
                    UnifiedAdvice,
                    {
                        "recommendations": [],
                        "confidence": 0.71,
                        "reasoning": "test",
                        "source": "fusion",
                    },
                ),
                (
                    FusionConfig,
                    {
                        "weights": {
                            "constitutional": 0.4,
                            "mangle": 0.3,
                            "workflow": 0.2,
                            "copilot": 0.1,
                        },
                        "threshold": 0.6,
                    },
                ),
                (
                    TelemetryHeaders,
                    {
                        "request_id": "test-123",
                        "session_id": "session-456",
                        "component": "orchestrator",
                    },
                ),
            ]

            for model_cls, test_data in test_cases:
                try:
                    model_cls(**test_data)
                    self.results.append(
                        ValidationResult(
                            category="Contract-First Interfaces",
                            check_name=f"{model_cls.__name__} Schema Validation",
                            passed=True,
                            details=f"Successfully validated {model_cls.__name__} schema",
                            severity="high",
                        )
                    )
                except ValidationError as e:
                    self.results.append(
                        ValidationResult(
                            category="Contract-First Interfaces",
                            check_name=f"{model_cls.__name__} Schema Validation",
                            passed=False,
                            details=f"Schema validation failed: {str(e)}",
                            severity="critical",
                            remediation="Fix Pydantic model definition or test data",
                        )
                    )

        except ImportError as e:
            self.results.append(
                ValidationResult(
                    category="Contract-First Interfaces",
                    check_name="Contract Imports",
                    passed=False,
                    details=f"Failed to import contract models: {str(e)}",
                    severity="critical",
                    remediation="Ensure contracts.yaml and model definitions are properly implemented",
                )
            )

    async def _validate_score_fusion_math(self):
        """Validate mathematical correctness of score fusion algorithms."""
        print("🧮 Validating score fusion mathematics...")

        try:
            from orchestrator import HardenedOrchestrator

            # Test fusion algorithm with known inputs
            test_weights = {
                "constitutional": 0.4,
                "mangle": 0.3,
                "workflow": 0.2,
                "copilot": 0.1,
            }
            test_scores = {
                "constitutional": 0.8,
                "mangle": 0.9,
                "workflow": 0.6,
                "copilot": 0.7,
            }

            orchestrator = HardenedOrchestrator()
            fused_score = orchestrator._calculate_weights(
                test_weights, test_scores
            )

            # Verify mathematical properties
            expected_fused = sum(
                w * s
                for w, s in zip(
                    test_weights.values(), test_scores.values(), strict=False
                )
            )

            if abs(fused_score - expected_fused) < 1e-6:
                self.results.append(
                    ValidationResult(
                        category="Score Fusion Math",
                        check_name="Weighted Average Calculation",
                        passed=True,
                        details=f"Fusion math correct: {fused_score:.6f} ≈ {expected_fused:.6f}",
                        severity="high",
                        metrics={
                            "calculated": fused_score,
                            "expected": expected_fused,
                        },
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        category="Score Fusion Math",
                        check_name="Weighted Average Calculation",
                        passed=False,
                        details=f"Fusion math incorrect: {fused_score:.6f} ≠ {expected_fused:.6f}",
                        severity="critical",
                        remediation="Fix fusion algorithm implementation",
                    )
                )

            # Test edge cases
            edge_cases = [
                ({"a": 1.0}, {"a": 0.5}, 0.5),
                ({"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 0.0}, 0.0),
                ({"a": 0.5, "b": 0.5}, {"a": 0.0, "b": 1.0}, 0.5),
            ]

            for weights, scores, expected in edge_cases:
                result = orchestrator._calculate_weights(weights, scores)
                if abs(result - expected) < 1e-6:
                    self.results.append(
                        ValidationResult(
                            category="Score Fusion Math",
                            check_name="Edge Case Handling",
                            passed=True,
                            details=f"Edge case correct: weights={weights}, scores={scores}, result={result}",
                            severity="medium",
                        )
                    )
                else:
                    self.results.append(
                        ValidationResult(
                            category="Score Fusion Math",
                            check_name="Edge Case Handling",
                            passed=False,
                            details=f"Edge case failed: expected {expected}, got {result}",
                            severity="high",
                            remediation="Fix edge case handling in fusion algorithm",
                        )
                    )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    category="Score Fusion Math",
                    check_name="Fusion Algorithm Validation",
                    passed=False,
                    details=f"Failed to validate fusion math: {str(e)}",
                    severity="critical",
                    remediation="Implement and test fusion algorithm",
                )
            )

    async def _validate_failure_semantics(self):
        """Validate explicit failure semantics and error handling."""
        print("❌ Validating failure semantics...")

        try:
            from .orchestrator import HardenedOrchestrator

            orchestrator = HardenedOrchestrator()

            # Test failure scenarios
            failure_scenarios = [
                {"error": "component_timeout", "component": "constitutional"},
                {"error": "invalid_input", "component": "mangle"},
                {"error": "network_failure", "component": "workflow"},
            ]

            for scenario in failure_scenarios:
                try:
                    # This should handle failures gracefully
                    result = await orchestrator._handle_component_failure(
                        scenario
                    )
                    if result and "error_handled" in result:
                        self.results.append(
                            ValidationResult(
                                category="Explicit Failure Semantics",
                                check_name=f"Failure Handling: {scenario['error']}",
                                passed=True,
                                details=f"Properly handled {scenario['error']} failure",
                                severity="high",
                            )
                        )
                    else:
                        self.results.append(
                            ValidationResult(
                                category="Explicit Failure Semantics",
                                check_name=f"Failure Handling: {scenario['error']}",
                                passed=False,
                                details=f"Failed to handle {scenario['error']} properly",
                                severity="high",
                                remediation="Implement proper failure handling",
                            )
                        )
                except Exception as e:
                    self.results.append(
                        ValidationResult(
                            category="Explicit Failure Semantics",
                            check_name=f"Failure Handling: {scenario['error']}",
                            passed=False,
                            details=f"Exception during failure handling: {str(e)}",
                            severity="critical",
                            remediation="Fix exception handling in failure scenarios",
                        )
                    )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    category="Explicit Failure Semantics",
                    check_name="Failure Semantics Framework",
                    passed=False,
                    details=f"Failed to validate failure semantics: {str(e)}",
                    severity="critical",
                    remediation="Implement comprehensive failure handling",
                )
            )

    async def _validate_canonical_orchestration(self):
        """Validate canonical orchestration workflow consistency."""
        print("🎯 Validating canonical orchestration...")

        try:
            from .golden_fixtures import load_golden_fixtures
            from .orchestrator import HardenedOrchestrator

            orchestrator = HardenedOrchestrator()
            fixtures = load_golden_fixtures()

            # Test canonical workflow execution
            for fixture_name, fixture_data in fixtures.items():
                try:
                    result = await orchestrator.orchestrate_request(
                        fixture_data["input"]
                    )

                    # Verify canonical structure
                    required_fields = [
                        "recommendations",
                        "confidence",
                        "reasoning",
                        "source",
                    ]
                    if all(field in result for field in required_fields):
                        self.results.append(
                            ValidationResult(
                                category="Canonical Orchestration",
                                check_name=f"Workflow Structure: {fixture_name}",
                                passed=True,
                                details=f"Canonical workflow structure maintained for {fixture_name}",
                                severity="high",
                            )
                        )
                    else:
                        missing = [
                            f for f in required_fields if f not in result
                        ]
                        self.results.append(
                            ValidationResult(
                                category="Canonical Orchestration",
                                check_name=f"Workflow Structure: {fixture_name}",
                                passed=False,
                                details=f"Missing canonical fields: {missing}",
                                severity="critical",
                                remediation="Ensure all canonical fields are present in orchestration output",
                            )
                        )

                except Exception as e:
                    self.results.append(
                        ValidationResult(
                            category="Canonical Orchestration",
                            check_name=f"Workflow Execution: {fixture_name}",
                            passed=False,
                            details=f"Failed to execute canonical workflow: {str(e)}",
                            severity="critical",
                            remediation="Fix orchestration workflow implementation",
                        )
                    )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    category="Canonical Orchestration",
                    check_name="Canonical Orchestration Framework",
                    passed=False,
                    details=f"Failed to validate canonical orchestration: {str(e)}",
                    severity="critical",
                    remediation="Implement canonical orchestration workflow",
                )
            )

    async def _validate_golden_fixtures(self):
        """Validate golden test fixtures integrity and coverage."""
        print("🧪 Validating golden test fixtures...")

        try:
            from .golden_fixtures import (
                load_golden_fixtures,
                validate_fixture_integrity,
            )

            fixtures = load_golden_fixtures()

            # Check fixture count and coverage
            if len(fixtures) >= 5:  # Minimum expected fixtures
                self.results.append(
                    ValidationResult(
                        category="Golden Test Fixtures",
                        check_name="Fixture Count",
                        passed=True,
                        details=f"Sufficient fixtures loaded: {len(fixtures)}",
                        severity="medium",
                        metrics={"fixture_count": len(fixtures)},
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        category="Golden Test Fixtures",
                        check_name="Fixture Count",
                        passed=False,
                        details=f"Insufficient fixtures: {len(fixtures)} < 5",
                        severity="high",
                        remediation="Add more comprehensive test fixtures",
                    )
                )

            # Validate fixture integrity
            integrity_results = validate_fixture_integrity(fixtures)
            for check_name, passed in integrity_results.items():
                self.results.append(
                    ValidationResult(
                        category="Golden Test Fixtures",
                        check_name=f"Integrity: {check_name}",
                        passed=passed,
                        details=f"Fixture integrity check: {check_name}",
                        severity="high" if not passed else "low",
                    )
                )

            # Test fixture schema compliance
            for fixture_name, fixture_data in fixtures.items():
                if (
                    "input" in fixture_data
                    and "expected_output" in fixture_data
                ):
                    self.results.append(
                        ValidationResult(
                            category="Golden Test Fixtures",
                            check_name=f"Schema Compliance: {fixture_name}",
                            passed=True,
                            details=f"Fixture {fixture_name} has proper input/output structure",
                            severity="medium",
                        )
                    )
                else:
                    self.results.append(
                        ValidationResult(
                            category="Golden Test Fixtures",
                            check_name=f"Schema Compliance: {fixture_name}",
                            passed=False,
                            details=f"Fixture {fixture_name} missing input or expected_output",
                            severity="high",
                            remediation="Fix fixture schema structure",
                        )
                    )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    category="Golden Test Fixtures",
                    check_name="Golden Fixtures Framework",
                    passed=False,
                    details=f"Failed to validate golden fixtures: {str(e)}",
                    severity="critical",
                    remediation="Implement and validate golden test fixtures",
                )
            )

    async def _validate_telemetry_schema(self):
        """Validate telemetry schema and event structure."""
        print("📊 Validating telemetry schema...")

        try:
            from .telemetry import TelemetryHeaders, TelemetryMiddleware

            # Test telemetry header generation
            test_request = {"component": "orchestrator", "operation": "fusion"}
            headers = TelemetryHeaders.generate(test_request)

            required_header_fields = ["request_id", "timestamp", "component"]
            if all(field in headers for field in required_header_fields):
                self.results.append(
                    ValidationResult(
                        category="Telemetry Schema",
                        check_name="Header Generation",
                        passed=True,
                        details="Telemetry headers generated correctly",
                        severity="high",
                    )
                )
            else:
                missing = [
                    f for f in required_header_fields if f not in headers
                ]
                self.results.append(
                    ValidationResult(
                        category="Telemetry Schema",
                        check_name="Header Generation",
                        passed=False,
                        details=f"Missing telemetry header fields: {missing}",
                        severity="critical",
                        remediation="Fix telemetry header generation",
                    )
                )

            # Test JSON envelope structure
            test_payload = {"result": "success", "confidence": 0.85}
            envelope = TelemetryMiddleware.create_envelope(
                test_payload, headers
            )

            required_envelope_fields = ["data", "metadata", "timestamp"]
            if all(field in envelope for field in required_envelope_fields):
                self.results.append(
                    ValidationResult(
                        category="Telemetry Schema",
                        check_name="JSON Envelope Structure",
                        passed=True,
                        details="Telemetry envelope structure correct",
                        severity="high",
                    )
                )
            else:
                missing = [
                    f for f in required_envelope_fields if f not in envelope
                ]
                self.results.append(
                    ValidationResult(
                        category="Telemetry Schema",
                        check_name="JSON Envelope Structure",
                        passed=False,
                        details=f"Missing envelope fields: {missing}",
                        severity="critical",
                        remediation="Fix telemetry envelope structure",
                    )
                )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    category="Telemetry Schema",
                    check_name="Telemetry Framework",
                    passed=False,
                    details=f"Failed to validate telemetry schema: {str(e)}",
                    severity="critical",
                    remediation="Implement telemetry schema and middleware",
                )
            )

    async def _validate_constitutional_compliance(self):
        """Validate constitutional compliance and quality gates."""
        print("⚖️ Validating constitutional compliance...")

        try:
            from .constitutional_engine import ConstitutionalEngine

            engine = ConstitutionalEngine()

            # Test constitutional analysis
            test_spec = (
                "Implement a user authentication system with password hashing"
            )
            analysis = await engine.analyze_constitutional_compliance(
                test_spec
            )

            required_compliance_fields = [
                "compliance_score",
                "violations",
                "recommendations",
            ]
            if all(field in analysis for field in required_compliance_fields):
                compliance_score = analysis.get("compliance_score", 0)
                if compliance_score >= 0.75:  # Constitutional threshold
                    self.results.append(
                        ValidationResult(
                            category="Constitutional Compliance",
                            check_name="Compliance Analysis",
                            passed=True,
                            details=f"Constitutional compliance score: {compliance_score:.2f} ≥ 0.75",
                            severity="high",
                            metrics={"compliance_score": compliance_score},
                        )
                    )
                else:
                    self.results.append(
                        ValidationResult(
                            category="Constitutional Compliance",
                            check_name="Compliance Analysis",
                            passed=False,
                            details=f"Constitutional compliance score: {compliance_score:.2f} < 0.75",
                            severity="critical",
                            remediation="Improve constitutional compliance",
                        )
                    )
            else:
                missing = [
                    f for f in required_compliance_fields if f not in analysis
                ]
                self.results.append(
                    ValidationResult(
                        category="Constitutional Compliance",
                        check_name="Compliance Analysis",
                        passed=False,
                        details=f"Missing compliance analysis fields: {missing}",
                        severity="critical",
                        remediation="Fix constitutional analysis implementation",
                    )
                )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    category="Constitutional Compliance",
                    check_name="Constitutional Framework",
                    passed=False,
                    details=f"Failed to validate constitutional compliance: {str(e)}",
                    severity="critical",
                    remediation="Implement constitutional compliance framework",
                )
            )

    async def _validate_performance_gates(self):
        """Validate performance gates and runtime efficiency."""
        print("⚡ Validating performance gates...")

        try:
            from .orchestrator import HardenedOrchestrator

            orchestrator = HardenedOrchestrator()

            # Test performance with timing
            test_input = {
                "prompt": "Implement a simple calculator",
                "context": {},
            }

            start_time = time.time()
            await orchestrator.orchestrate_request(test_input)
            end_time = time.time()

            execution_time = end_time - start_time

            # Performance gate: should complete within 30 seconds
            if execution_time <= 30.0:
                self.results.append(
                    ValidationResult(
                        category="Performance Gates",
                        check_name="Execution Time",
                        passed=True,
                        details=f"Orchestration completed in {execution_time:.2f}s ≤ 30.0s",
                        severity="high",
                        metrics={
                            "execution_time": execution_time,
                            "threshold": 30.0,
                        },
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        category="Performance Gates",
                        check_name="Execution Time",
                        passed=False,
                        details=f"Orchestration too slow: {execution_time:.2f}s > 30.0s",
                        severity="critical",
                        remediation="Optimize orchestration performance",
                        metrics={
                            "execution_time": execution_time,
                            "threshold": 30.0,
                        },
                    )
                )

            # Memory usage check (if available)
            try:
                import psutil

                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024

                if memory_mb <= 500:  # 500MB limit
                    self.results.append(
                        ValidationResult(
                            category="Performance Gates",
                            check_name="Memory Usage",
                            passed=True,
                            details=f"Memory usage: {memory_mb:.1f}MB ≤ 500MB",
                            severity="medium",
                            metrics={"memory_mb": memory_mb, "threshold": 500},
                        )
                    )
                else:
                    self.results.append(
                        ValidationResult(
                            category="Performance Gates",
                            check_name="Memory Usage",
                            passed=False,
                            details=f"Memory usage too high: {memory_mb:.1f}MB > 500MB",
                            severity="high",
                            remediation="Optimize memory usage",
                            metrics={"memory_mb": memory_mb, "threshold": 500},
                        )
                    )
            except ImportError:
                self.results.append(
                    ValidationResult(
                        category="Performance Gates",
                        check_name="Memory Usage",
                        passed=True,
                        details="Memory monitoring not available (psutil not installed)",
                        severity="low",
                    )
                )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    category="Performance Gates",
                    check_name="Performance Framework",
                    passed=False,
                    details=f"Failed to validate performance gates: {str(e)}",
                    severity="critical",
                    remediation="Implement performance monitoring and gates",
                )
            )

    async def _validate_code_reasoning(self):
        """Validate code reasoning capabilities and mangle-style analysis."""
        print("🔍 Validating code reasoning capabilities...")

        try:
            import os
            import tempfile

            from .code_reasoning import CodeIngester, RuleEngine

            # Test code ingestion
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Create test Python files
                test_file = os.path.join(tmp_dir, "test_module.py")
                with open(test_file, "w") as f:
                    f.write(
                        """
def complex_function(x):
    # High complexity function
    for i in range(x):
        if i % 2 == 0:
            if i % 3 == 0:
                x += i
            else:
                x -= i
        else:
            try:
                x *= 2
            except:
                x = 0
    return x

def simple_function():
    return 42
"""
                    )

                # Test ingestion
                ingester = CodeIngester(":memory:")
                stats = ingester.ingest_repository(tmp_dir)

                if (
                    stats["files_processed"] > 0
                    and stats["symbols_extracted"] > 0
                ):
                    self.results.append(
                        ValidationResult(
                            category="Code Reasoning",
                            check_name="Code Ingestion",
                            passed=True,
                            details=f"Successfully ingested {stats['files_processed']} files, "
                            f"{stats['symbols_extracted']} symbols",
                            severity="high",
                            metrics=stats,
                        )
                    )
                else:
                    self.results.append(
                        ValidationResult(
                            category="Code Reasoning",
                            check_name="Code Ingestion",
                            passed=False,
                            details="Failed to ingest test code properly",
                            severity="critical",
                            remediation="Fix code ingestion implementation",
                        )
                    )

            # Test rule engine
            rule_engine = RuleEngine(":memory:")

            # Add a test rule
            rule_engine.add_rule(
                "test_complex_function",
                """
                SELECT s.symbol, s.file
                FROM symbol s
                JOIN complexity c ON c.symbol = s.symbol
                WHERE s.kind = 'function' AND c.score > 0.5
                """,
            )

            # Run rules (should handle empty database gracefully)
            findings = rule_engine.run_all_rules()

            if isinstance(findings, dict):
                self.results.append(
                    ValidationResult(
                        category="Code Reasoning",
                        check_name="Rule Engine",
                        passed=True,
                        details=f"Rule engine executed successfully, found {len(findings)} rule types",
                        severity="high",
                        metrics={"rules_executed": len(findings)},
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        category="Code Reasoning",
                        check_name="Rule Engine",
                        passed=False,
                        details="Rule engine failed to execute properly",
                        severity="critical",
                        remediation="Fix rule engine implementation",
                    )
                )

            # Test with sample repo
            sample_repo_path = "mangle_code_scaffold_v2/sample_repo"
            if os.path.exists(sample_repo_path):
                ingester = CodeIngester(":memory:")
                stats = ingester.ingest_repository(sample_repo_path)
                rule_engine = RuleEngine(":memory:")
                findings = rule_engine.run_all_rules()

                # Check for expected findings
                expected_findings = {
                    "untested_function": 1,  # a::compute should be untested
                    "cycle": 2,  # a.py <-> b.py circular import
                }

                for rule_name, expected_count in expected_findings.items():
                    actual_count = len(findings.get(rule_name, []))
                    if actual_count == expected_count:
                        self.results.append(
                            ValidationResult(
                                category="Code Reasoning",
                                check_name=f"Sample Analysis: {rule_name}",
                                passed=True,
                                details=f"Correctly found {actual_count} {rule_name} issues",
                                severity="medium",
                            )
                        )
                    else:
                        self.results.append(
                            ValidationResult(
                                category="Code Reasoning",
                                check_name=f"Sample Analysis: {rule_name}",
                                passed=False,
                                details=f"Expected {expected_count} {rule_name} findings, got {actual_count}",
                                severity="high",
                                remediation="Verify rule definitions and sample data",
                            )
                        )

        except Exception as e:
            self.results.append(
                ValidationResult(
                    category="Code Reasoning",
                    check_name="Code Reasoning Framework",
                    passed=False,
                    details=f"Failed to validate code reasoning: {str(e)}",
                    severity="critical",
                    remediation="Implement code reasoning capabilities",
                )
            )

    def _generate_report(self) -> dict[str, Any]:
        """Generate comprehensive validation report."""
        end_time = time.time()
        total_time = end_time - self.start_time

        # Categorize results
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        # Calculate statistics
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.passed])
        failed_checks = total_checks - passed_checks

        critical_failures = len(
            [
                r
                for r in self.results
                if not r.passed and r.severity == "critical"
            ]
        )
        high_failures = len(
            [r for r in self.results if not r.passed and r.severity == "high"]
        )

        # Overall assessment
        overall_passed = critical_failures == 0 and high_failures <= 2

        report = {
            "summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "pass_rate": (
                    passed_checks / total_checks if total_checks > 0 else 0
                ),
                "critical_failures": critical_failures,
                "high_failures": high_failures,
                "overall_passed": overall_passed,
                "execution_time": total_time,
            },
            "categories": {},
            "failures": [],
            "recommendations": [],
        }

        # Category breakdown
        for category, results in categories.items():
            category_passed = len([r for r in results if r.passed])
            category_total = len(results)
            report["categories"][category] = {
                "total": category_total,
                "passed": category_passed,
                "failed": category_total - category_passed,
                "pass_rate": (
                    category_passed / category_total
                    if category_total > 0
                    else 0
                ),
            }

        # Collect failures and recommendations
        for result in self.results:
            if not result.passed:
                report["failures"].append(
                    {
                        "category": result.category,
                        "check": result.check_name,
                        "severity": result.severity,
                        "details": result.details,
                        "remediation": result.remediation,
                    }
                )

            if result.remediation:
                report["recommendations"].append(
                    {
                        "category": result.category,
                        "check": result.check_name,
                        "severity": result.severity,
                        "remediation": result.remediation,
                    }
                )

        return report


async def run_validation_checklist() -> dict[str, Any]:
    """Convenience function to run the complete validation checklist."""
    checklist = ValidationChecklist()
    return await checklist.run_full_validation()


if __name__ == "__main__":
    # Run validation when executed directly
    async def main():
        report = await run_validation_checklist()

        print("\n" + "=" * 80)
        print("UNIFIED INTELLIGENCE LAYER VALIDATION REPORT")
        print("=" * 80)

        summary = report["summary"]
        print("\n📊 SUMMARY:")
        print(f"  Total Checks: {summary['total_checks']}")
        print(f"  Passed: {summary['passed_checks']}")
        print(f"  Failed: {summary['failed_checks']}")
        print(f"  Pass Rate: {summary['pass_rate']:.1%}")
        print(f"  Critical Failures: {summary['critical_failures']}")
        print(f"  High Priority Failures: {summary['high_failures']}")
        print(
            f"  Overall Status: {'✅ PASSED' if summary['overall_passed'] else '❌ FAILED'}"
        )
        print(f"  Execution Time: {summary['execution_time']:.2f}s")

        print("\n📈 CATEGORY BREAKDOWN:")
        for category, stats in report["categories"].items():
            status = (
                "✅"
                if stats["pass_rate"] == 1.0
                else "⚠️" if stats["pass_rate"] >= 0.8 else "❌"
            )
            print(
                f"  {status} {category}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})"
            )

        if report["failures"]:
            print("\n❌ KEY FAILURES:")
            for failure in report["failures"][:5]:  # Show top 5
                print(
                    f"  {failure['severity'].upper()}: {failure['category']} - {failure['check']}"
                )
                print(f"    {failure['details']}")
                if failure["remediation"]:
                    print(f"    💡 {failure['remediation']}")

        if report["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in report["recommendations"][:5]:  # Show top 5
                print(f"  {rec['severity'].upper()}: {rec['remediation']}")

        print("\n" + "=" * 80)

        return report

    report = asyncio.run(main())
    exit(0 if report["summary"]["overall_passed"] else 1)
