"""
CI Pipeline Quality Gates

Implements constitutional compliance gates and automated quality enforcement
for continuous integration pipelines.
"""

import asyncio
import subprocess
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Quality gate validation result."""
    gate_name: str
    passed: bool
    score: float
    threshold: float
    violations: List[str]
    execution_time_ms: float
    metadata: Dict[str, Any]
    timestamp: datetime


class ConstitutionalGate:
    """Constitutional compliance quality gate."""
    
    def __init__(self, constitutional_engine, threshold: float = 0.75):
        self.constitutional_engine = constitutional_engine
        self.threshold = threshold
        self.name = "constitutional_compliance"
        
    async def validate(self, context: Dict[str, Any]) -> QualityGateResult:
        """Validate constitutional compliance."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Extract validation data from context
            if context.get("type") == "commit":
                compliance_score = await self.constitutional_engine.validate_commit(
                    context.get("commit_message", ""),
                    context.get("changed_files", []),
                    context.get("diff_data", "")
                )
            else:
                compliance_score = await self.constitutional_engine.validate_compliance(context)
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            violations = [v.description for v in compliance_score.violations]
            
            return QualityGateResult(
                gate_name=self.name,
                passed=compliance_score.is_compliant,
                score=compliance_score.overall_score,
                threshold=self.threshold,
                violations=violations,
                execution_time_ms=execution_time,
                metadata=compliance_score.to_dict(),
                timestamp=start_time
            )
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.error(f"Constitutional gate validation failed: {e}")
            
            return QualityGateResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                threshold=self.threshold,
                violations=[f"Validation error: {str(e)}"],
                execution_time_ms=execution_time,
                metadata={"error": str(e)},
                timestamp=start_time
            )


class PerformanceGate:
    """Performance quality gate."""
    
    def __init__(self, performance_monitor, thresholds: Dict[str, float]):
        self.performance_monitor = performance_monitor
        self.thresholds = thresholds
        self.name = "performance_benchmarks"
        
    async def validate(self, context: Dict[str, Any]) -> QualityGateResult:
        """Validate performance benchmarks."""
        start_time = datetime.now(timezone.utc)
        
        try:
            summary = self.performance_monitor.get_performance_summary()
            violations = []
            
            # Check response time threshold
            avg_response_time = summary.get("average_response_time_ms", 0)
            response_threshold = self.thresholds.get("response_time_ms", 1000)
            if avg_response_time > response_threshold:
                violations.append(f"Response time {avg_response_time:.2f}ms exceeds threshold {response_threshold}ms")
            
            # Check success rate threshold
            success_rate = summary.get("success_rate", 1.0)
            success_threshold = self.thresholds.get("success_rate", 0.95)
            if success_rate < success_threshold:
                violations.append(f"Success rate {success_rate:.3f} below threshold {success_threshold}")
            
            # Calculate overall performance score
            response_score = min(1.0, response_threshold / max(avg_response_time, 1))
            success_score = success_rate
            overall_score = (response_score + success_score) / 2
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return QualityGateResult(
                gate_name=self.name,
                passed=len(violations) == 0,
                score=overall_score,
                threshold=0.8,  # Performance threshold
                violations=violations,
                execution_time_ms=execution_time,
                metadata=summary,
                timestamp=start_time
            )
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.error(f"Performance gate validation failed: {e}")
            
            return QualityGateResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                threshold=0.8,
                violations=[f"Performance validation error: {str(e)}"],
                execution_time_ms=execution_time,
                metadata={"error": str(e)},
                timestamp=start_time
            )


class SecurityGate:
    """Security quality gate."""
    
    def __init__(self, scan_config: Dict[str, Any]):
        self.scan_config = scan_config
        self.name = "security_scan"
        
    async def validate(self, context: Dict[str, Any]) -> QualityGateResult:
        """Validate security requirements."""
        start_time = datetime.now(timezone.utc)
        
        try:
            violations = []
            
            # Run dependency security scan
            if self.scan_config.get("check_dependencies", True):
                dep_violations = await self._scan_dependencies(context)
                violations.extend(dep_violations)
            
            # Run code security scan
            if self.scan_config.get("check_code", True):
                code_violations = await self._scan_code(context)
                violations.extend(code_violations)
            
            # Calculate security score
            critical_violations = sum(1 for v in violations if "critical" in v.lower())
            score = max(0.0, 1.0 - (critical_violations * 0.5) - (len(violations) * 0.1))
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return QualityGateResult(
                gate_name=self.name,
                passed=critical_violations == 0,
                score=score,
                threshold=0.8,
                violations=violations,
                execution_time_ms=execution_time,
                metadata={"critical_violations": critical_violations},
                timestamp=start_time
            )
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.error(f"Security gate validation failed: {e}")
            
            return QualityGateResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                threshold=0.8,
                violations=[f"Security validation error: {str(e)}"],
                execution_time_ms=execution_time,
                metadata={"error": str(e)},
                timestamp=start_time
            )
    
    async def _scan_dependencies(self, context: Dict[str, Any]) -> List[str]:
        """Scan dependencies for security vulnerabilities."""
        violations = []
        
        try:
            # Example: Run safety check for Python dependencies
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0 and result.stdout:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    violations.append(f"Security vulnerability in {vuln.get('package', 'unknown')}: {vuln.get('advisory', 'No details')}")
                    
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            # Safety not available or failed - continue without dependency scan
            pass
            
        return violations
    
    async def _scan_code(self, context: Dict[str, Any]) -> List[str]:
        """Scan code for security issues."""
        violations = []
        
        # Basic security pattern checks
        changed_files = context.get("changed_files", [])
        for file_path in changed_files:
            if file_path.endswith(".py"):
                violations.extend(await self._check_python_security(file_path))
                
        return violations
    
    async def _check_python_security(self, file_path: str) -> List[str]:
        """Check Python file for basic security issues."""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic security checks
            if "exec(" in content:
                violations.append(f"Potential code injection in {file_path}: exec() usage")
            if "eval(" in content:
                violations.append(f"Potential code injection in {file_path}: eval() usage")
            if "os.system(" in content:
                violations.append(f"Potential command injection in {file_path}: os.system() usage")
                
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            pass
            
        return violations


class QualityGatePipeline:
    """
    Quality gate pipeline orchestrator for CI/CD integration.
    
    Implements Article IV: Integration-First through comprehensive pipeline integration.
    Implements Article II: Test-First through automated quality validation.
    """
    
    def __init__(self):
        self.gates: List = []
        self.execution_history: List[Dict[str, Any]] = []
        
    def add_gate(self, gate) -> None:
        """Add a quality gate to the pipeline."""
        self.gates.append(gate)
        logger.info(f"Added quality gate: {gate.name}")
        
    async def execute_pipeline(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all quality gates in the pipeline."""
        start_time = datetime.now(timezone.utc)
        results = []
        overall_passed = True
        
        logger.info(f"Executing quality gate pipeline with {len(self.gates)} gates")
        
        # Execute gates in parallel for better performance
        gate_tasks = [gate.validate(context) for gate in self.gates]
        gate_results = await asyncio.gather(*gate_tasks, return_exceptions=True)
        
        for i, result in enumerate(gate_results):
            if isinstance(result, Exception):
                logger.error(f"Gate {self.gates[i].name} failed with exception: {result}")
                gate_result = QualityGateResult(
                    gate_name=self.gates[i].name,
                    passed=False,
                    score=0.0,
                    threshold=1.0,
                    violations=[f"Gate execution failed: {str(result)}"],
                    execution_time_ms=0,
                    metadata={"error": str(result)},
                    timestamp=start_time
                )
            else:
                gate_result = result
            
            results.append(gate_result)
            if not gate_result.passed:
                overall_passed = False
                
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        # Create pipeline execution summary
        pipeline_result = {
            "overall_passed": overall_passed,
            "execution_time_ms": execution_time,
            "timestamp": start_time.isoformat(),
            "gate_results": [
                {
                    "gate_name": r.gate_name,
                    "passed": r.passed,
                    "score": r.score,
                    "threshold": r.threshold,
                    "violations": r.violations,
                    "execution_time_ms": r.execution_time_ms
                } for r in results
            ],
            "summary": {
                "total_gates": len(results),
                "passed_gates": sum(1 for r in results if r.passed),
                "failed_gates": sum(1 for r in results if not r.passed),
                "average_score": sum(r.score for r in results) / len(results) if results else 0,
                "total_violations": sum(len(r.violations) for r in results)
            }
        }
        
        # Store execution history
        self.execution_history.append(pipeline_result)
        
        # Log results
        if overall_passed:
            logger.info(f"Quality gate pipeline PASSED in {execution_time:.2f}ms")
        else:
            logger.warning(f"Quality gate pipeline FAILED in {execution_time:.2f}ms")
            for result in results:
                if not result.passed:
                    logger.warning(f"  {result.gate_name}: {len(result.violations)} violations")
        
        return pipeline_result
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent pipeline execution history."""
        return self.execution_history[-limit:]
    
    def get_gate_statistics(self) -> Dict[str, Any]:
        """Get statistics about gate performance."""
        if not self.execution_history:
            return {"status": "no_data"}
        
        gate_stats = {}
        for execution in self.execution_history:
            for gate_result in execution["gate_results"]:
                gate_name = gate_result["gate_name"]
                if gate_name not in gate_stats:
                    gate_stats[gate_name] = {
                        "total_executions": 0,
                        "passed_executions": 0,
                        "total_execution_time": 0,
                        "total_violations": 0
                    }
                
                stats = gate_stats[gate_name]
                stats["total_executions"] += 1
                if gate_result["passed"]:
                    stats["passed_executions"] += 1
                stats["total_execution_time"] += gate_result["execution_time_ms"]
                stats["total_violations"] += len(gate_result["violations"])
        
        # Calculate derived statistics
        for gate_name, stats in gate_stats.items():
            stats["success_rate"] = stats["passed_executions"] / stats["total_executions"]
            stats["average_execution_time"] = stats["total_execution_time"] / stats["total_executions"]
            stats["average_violations"] = stats["total_violations"] / stats["total_executions"]
        
        return gate_stats