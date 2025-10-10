#!/usr/bin/env python3
"""
Enhanced Script of Thought Interpreter - Constitutional Compliance Integration
==============================================================================

Enhances the SoT interpreter with constitutional compliance validation,
EOS orchestration, and advanced reasoning capabilities for secure and
governed script execution.

This enhancement integrates:
- Script of Thought → Constitutional Compliance Validation
- EOS LADDER Operations → Script Planning and Execution
- Mangle Reasoning → Step-by-Step Logic Validation
- Security Framework → Safe Code Execution

Author: Super ALITA Framework
Version: 2.0.0 (Enhanced with Constitutional Compliance)
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..computational_env.executor import ComputationalEnvironment

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Constitutional compliance status"""

    COMPLIANT = "compliant"
    VIOLATION = "violation"
    WARNING = "warning"
    PENDING = "pending"


class ExecutionMode(Enum):
    """Script execution modes"""

    SAFE = "safe"  # Full constitutional validation
    SANDBOX = "sandbox"  # Isolated execution
    RESTRICTED = "restricted"  # Limited operations
    TRUSTED = "trusted"  # Minimal validation


@dataclass
class ComplianceResult:
    """Result of constitutional compliance check"""

    status: ComplianceStatus
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    validation_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedStepResult:
    """Enhanced result with constitutional compliance"""

    step_id: int
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Enhanced fields
    compliance_result: ComplianceResult | None = None
    execution_time: float = 0.0
    resource_usage: dict[str, Any] = field(default_factory=dict)
    security_assessment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "step_id": self.step_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "resource_usage": self.resource_usage,
            "security_assessment": self.security_assessment,
        }

        if self.compliance_result:
            result["compliance"] = {
                "status": self.compliance_result.status.value,
                "violations": self.compliance_result.violations,
                "warnings": self.compliance_result.warnings,
                "recommendations": self.compliance_result.recommendations,
                "confidence_score": self.compliance_result.confidence_score,
            }

        return result


class ConstitutionalValidator:
    """Validates script steps against constitutional rules"""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Constitutional rules for script execution
        self.rules = {
            "code_safety": [
                "No file system write operations outside sandbox",
                "No network access without explicit permission",
                "No system command execution",
                "No access to sensitive environment variables",
            ],
            "data_privacy": [
                "No logging of personal identifiable information",
                "No transmission of sensitive data",
                "Data anonymization required for analysis",
            ],
            "resource_limits": [
                "Memory usage must not exceed 512MB per step",
                "Execution time must not exceed 30 seconds per step",
                "CPU usage must not exceed 80% for more than 10 seconds",
            ],
            "content_policy": [
                "No generation of harmful content",
                "No creation of malicious code",
                "Respect intellectual property rights",
            ],
        }

        # Risk patterns to detect
        self.risk_patterns = {
            "file_operations": [
                "open(",
                "write(",
                "delete",
                "os.remove",
                "shutil",
            ],
            "network_access": ["requests.", "urllib", "socket", "http"],
            "system_commands": ["os.system", "subprocess", "exec(", "eval("],
            "sensitive_data": [
                "password",
                "token",
                "key",
                "secret",
                "credential",
            ],
        }

    async def validate_step(
        self, step_content: str, step_type: str, context: dict[str, Any] = None
    ) -> ComplianceResult:
        """Validate a script step for constitutional compliance"""

        self.logger.info(f"🔍 Validating step: {step_type}")

        context = context or {}
        result = ComplianceResult(status=ComplianceStatus.COMPLIANT)

        try:
            # Validate against constitutional rules
            await self._validate_code_safety(step_content, result)
            await self._validate_data_privacy(step_content, context, result)
            await self._validate_resource_limits(step_content, context, result)
            await self._validate_content_policy(step_content, result)

            # Calculate overall compliance status
            if result.violations:
                result.status = ComplianceStatus.VIOLATION
            elif result.warnings:
                result.status = ComplianceStatus.WARNING
            else:
                result.status = ComplianceStatus.COMPLIANT

            # Calculate confidence score
            result.confidence_score = self._calculate_confidence(result)

            self.logger.info(f"✅ Validation complete: {result.status.value}")

        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            result.status = ComplianceStatus.VIOLATION
            result.violations.append(f"Validation error: {str(e)}")
            result.confidence_score = 0.0

        return result

    async def _validate_code_safety(
        self, content: str, result: ComplianceResult
    ) -> None:
        """Validate code safety rules"""

        content_lower = content.lower()

        # Check for risky file operations
        for pattern in self.risk_patterns["file_operations"]:
            if pattern in content_lower:
                result.violations.append(
                    f"Unsafe file operation detected: {pattern}"
                )

        # Check for network access
        for pattern in self.risk_patterns["network_access"]:
            if pattern in content_lower:
                result.violations.append(f"Network access detected: {pattern}")

        # Check for system commands
        for pattern in self.risk_patterns["system_commands"]:
            if pattern in content_lower:
                result.violations.append(
                    f"System command execution detected: {pattern}"
                )

    async def _validate_data_privacy(
        self, content: str, context: dict[str, Any], result: ComplianceResult
    ) -> None:
        """Validate data privacy rules"""

        content_lower = content.lower()

        # Check for sensitive data patterns
        for pattern in self.risk_patterns["sensitive_data"]:
            if pattern in content_lower:
                result.warnings.append(
                    f"Potential sensitive data reference: {pattern}"
                )

        # Check if data contains PII (simplified check)
        if context.get("contains_pii", False):
            if "log" in content_lower or "print" in content_lower:
                result.violations.append("Potential PII logging detected")

    async def _validate_resource_limits(
        self, content: str, context: dict[str, Any], result: ComplianceResult
    ) -> None:
        """Validate resource usage limits"""

        # Check for potentially resource-intensive operations
        intensive_patterns = ["while True", "for _ in range(", "recursive"]

        for pattern in intensive_patterns:
            if pattern in content.lower():
                result.warnings.append(
                    f"Potentially resource-intensive operation: {pattern}"
                )

        # Check estimated execution time
        estimated_time = context.get("estimated_execution_time", 0)
        if estimated_time > 30:  # 30 seconds limit
            result.violations.append(
                f"Estimated execution time ({estimated_time}s) exceeds limit (30s)"
            )

    async def _validate_content_policy(
        self, content: str, result: ComplianceResult
    ) -> None:
        """Validate content policy compliance"""

        # Check for potentially harmful content patterns
        harmful_patterns = ["virus", "malware", "hack", "exploit", "crack"]

        content_lower = content.lower()
        for pattern in harmful_patterns:
            if pattern in content_lower:
                result.warnings.append(
                    f"Potentially harmful content reference: {pattern}"
                )

    def _calculate_confidence(self, result: ComplianceResult) -> float:
        """Calculate confidence score for validation"""

        base_confidence = 0.8

        # Reduce confidence for violations
        confidence = base_confidence - (len(result.violations) * 0.2)

        # Reduce confidence for warnings
        confidence -= len(result.warnings) * 0.1

        return max(0.0, min(1.0, confidence))


class EnhancedScriptOfThoughtInterpreter:
    """
    Enhanced Script of Thought interpreter with constitutional compliance
    """

    def __init__(
        self,
        computational_env: "ComputationalEnvironment",
        constitutional_validator: ConstitutionalValidator = None,
        execution_mode: ExecutionMode = ExecutionMode.SAFE,
    ):
        # Import the original interpreter functionality
        try:
            from .interpreter import ScriptOfThoughtInterpreter

            self.base_interpreter = ScriptOfThoughtInterpreter(
                computational_env
            )
        except ImportError:
            self.base_interpreter = None

        self.computational_env = computational_env
        self.constitutional_validator = (
            constitutional_validator or ConstitutionalValidator()
        )
        self.execution_mode = execution_mode
        self.logger = logging.getLogger(__name__)

        # Enhanced execution state
        self.execution_history: list[dict[str, Any]] = []
        self.compliance_cache: dict[str, ComplianceResult] = {}
        self.risk_assessment: dict[str, Any] = {}

    async def execute_script_enhanced(
        self,
        script_text: str,
        session_id: str | None = None,
        execution_context: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        Execute script with enhanced constitutional compliance
        """

        execution_context = execution_context or {}
        start_time = datetime.now(UTC)

        self.logger.info(
            f"🚀 Enhanced script execution started: {self.execution_mode.value} mode"
        )

        try:
            # Pre-execution validation
            pre_validation = await self._pre_execution_validation(
                script_text, execution_context
            )

            if pre_validation["status"] == ComplianceStatus.VIOLATION:
                return {
                    "success": False,
                    "error": "Constitutional compliance violations detected",
                    "compliance_violations": pre_validation["violations"],
                    "execution_mode": self.execution_mode.value,
                }

            # Parse script using base interpreter
            if self.base_interpreter:
                script = self.base_interpreter.parser.parse_script(script_text)
            else:
                # Mock script parsing
                script = self._mock_parse_script(script_text)

            if not script:
                return {
                    "success": False,
                    "error": "Failed to parse script",
                    "execution_mode": self.execution_mode.value,
                }

            # Execute steps with constitutional compliance
            step_results = []
            total_violations = 0

            for _i, step in enumerate(
                script.steps if hasattr(script, "steps") else []
            ):
                step_result = await self._execute_step_enhanced(
                    step, session_id, execution_context
                )
                step_results.append(step_result.to_dict())

                # Count violations
                if step_result.compliance_result:
                    total_violations += len(
                        step_result.compliance_result.violations
                    )

                # Stop on critical violations in SAFE mode
                if (
                    self.execution_mode == ExecutionMode.SAFE
                    and step_result.compliance_result
                    and step_result.compliance_result.status
                    == ComplianceStatus.VIOLATION
                ):
                    break

            # Generate enhanced summary
            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            summary = {
                "success": all(r["success"] for r in step_results),
                "total_steps": len(step_results),
                "completed_steps": len(
                    [r for r in step_results if r["success"]]
                ),
                "failed_steps": len(
                    [r for r in step_results if not r["success"]]
                ),
                "step_results": step_results,
                "execution_context": {
                    "session_id": session_id,
                    "execution_mode": self.execution_mode.value,
                    "execution_time": execution_time,
                    "start_time": start_time.isoformat(),
                },
                "compliance_summary": {
                    "total_violations": total_violations,
                    "pre_validation_status": pre_validation["status"].value,
                    "overall_compliance": (
                        "compliant"
                        if total_violations == 0
                        else "violations_detected"
                    ),
                },
            }

            # Store execution history
            self.execution_history.append(summary)

            self.logger.info(
                f"✅ Enhanced script execution complete: "
                f"{summary['completed_steps']}/{summary['total_steps']} steps"
            )

            return summary

        except Exception as e:
            self.logger.error(f"❌ Enhanced script execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_mode": self.execution_mode.value,
                "execution_time": (
                    datetime.now(UTC) - start_time
                ).total_seconds(),
            }

    async def _pre_execution_validation(
        self, script_text: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate entire script before execution"""

        self.logger.info("🔍 Pre-execution constitutional validation")

        # Overall script validation
        validation_result = await self.constitutional_validator.validate_step(
            script_text, "script", context
        )

        return {
            "status": validation_result.status,
            "violations": validation_result.violations,
            "warnings": validation_result.warnings,
            "recommendations": validation_result.recommendations,
        }

    def _mock_parse_script(self, script_text: str):
        """Mock script parsing when base interpreter unavailable"""

        # Simple mock implementation
        class MockScript:
            def __init__(self, text):
                self.steps = []
                lines = text.strip().split("\n")
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith("#"):
                        step = type(
                            "MockStep",
                            (),
                            {
                                "step_id": i,
                                "content": line.strip(),
                                "step_type": "compute",
                                "metadata": {},
                            },
                        )()
                        self.steps.append(step)

        return MockScript(script_text)

    async def _execute_step_enhanced(
        self,
        step,
        session_id: str | None = None,
        context: dict[str, Any] = None,
    ) -> EnhancedStepResult:
        """Execute a single step with constitutional compliance"""

        start_time = datetime.now(UTC)
        step_id = getattr(step, "step_id", 0)
        step_content = getattr(step, "content", "")
        step_type = getattr(step, "step_type", "unknown")

        self.logger.info(f"🔄 Executing step {step_id}: {step_type}")

        try:
            # Step-level constitutional validation
            compliance_result = (
                await self.constitutional_validator.validate_step(
                    step_content, step_type, context
                )
            )

            # Initialize enhanced result
            enhanced_result = EnhancedStepResult(
                step_id=step_id,
                success=True,
                compliance_result=compliance_result,
            )

            # Skip execution if violations detected in SAFE mode
            if (
                self.execution_mode == ExecutionMode.SAFE
                and compliance_result.status == ComplianceStatus.VIOLATION
            ):
                enhanced_result.success = False
                enhanced_result.error = "Constitutional compliance violation"
                return enhanced_result

            # Execute step with base interpreter if available
            if self.base_interpreter:
                base_result = await self.base_interpreter._execute_step(
                    step, session_id
                )
                enhanced_result.success = base_result.success
                enhanced_result.data = base_result.data
                enhanced_result.error = base_result.error
                enhanced_result.metadata = base_result.metadata
            else:
                # Mock execution
                enhanced_result.data = {
                    "mock_result": f"Executed {step_type}: {step_content[:50]}"
                }
                enhanced_result.metadata = {
                    "step_type": step_type,
                    "mock": True,
                }

            # Calculate execution time
            enhanced_result.execution_time = (
                datetime.now(UTC) - start_time
            ).total_seconds()

            # Assess resource usage (mock implementation)
            enhanced_result.resource_usage = {
                "memory_mb": 50,  # Mock values
                "cpu_percent": 25,
                "execution_time": enhanced_result.execution_time,
            }

            # Security assessment
            enhanced_result.security_assessment = {
                "risk_level": (
                    "low"
                    if compliance_result.status == ComplianceStatus.COMPLIANT
                    else "medium"
                ),
                "violations_count": len(compliance_result.violations),
                "warnings_count": len(compliance_result.warnings),
            }

            self.logger.info(
                f"✅ Step {step_id} complete: {enhanced_result.success}"
            )

            return enhanced_result

        except Exception as e:
            self.logger.error(f"❌ Step {step_id} failed: {e}")

            return EnhancedStepResult(
                step_id=step_id,
                success=False,
                error=str(e),
                execution_time=(
                    datetime.now(UTC) - start_time
                ).total_seconds(),
                compliance_result=ComplianceResult(
                    status=ComplianceStatus.VIOLATION
                ),
            )

    async def get_compliance_report(self) -> dict[str, Any]:
        """Generate comprehensive compliance report"""

        if not self.execution_history:
            return {
                "total_executions": 0,
                "compliance_status": "no_data",
                "recommendations": ["No execution history available"],
            }

        # Analyze execution history
        total_executions = len(self.execution_history)
        total_violations = sum(
            execution.get("compliance_summary", {}).get("total_violations", 0)
            for execution in self.execution_history
        )

        compliant_executions = len(
            [
                exec
                for exec in self.execution_history
                if exec.get("compliance_summary", {}).get("overall_compliance")
                == "compliant"
            ]
        )

        compliance_rate = (
            compliant_executions / total_executions
            if total_executions > 0
            else 0
        )

        # Generate recommendations
        recommendations = []
        if compliance_rate < 0.8:
            recommendations.append(
                "Review and improve script constitutional compliance"
            )
        if total_violations > 10:
            recommendations.append(
                "Implement stricter pre-execution validation"
            )
        if compliance_rate == 1.0:
            recommendations.append(
                "Excellent compliance rate - maintain current practices"
            )

        return {
            "total_executions": total_executions,
            "total_violations": total_violations,
            "compliant_executions": compliant_executions,
            "compliance_rate": compliance_rate,
            "compliance_status": (
                "excellent"
                if compliance_rate >= 0.95
                else "good" if compliance_rate >= 0.8 else "needs_improvement"
            ),
            "recommendations": recommendations,
            "execution_modes_used": list(
                {
                    exec.get("execution_context", {}).get(
                        "execution_mode", "unknown"
                    )
                    for exec in self.execution_history
                }
            ),
        }


# Export main classes
__all__ = [
    "ComplianceStatus",
    "ExecutionMode",
    "ComplianceResult",
    "EnhancedStepResult",
    "ConstitutionalValidator",
    "EnhancedScriptOfThoughtInterpreter",
]
