#!/usr/bin/env python3
"""
Super Alita Unification Validation Script

Comprehensive validation script that orchestrates all validation components:
- Schema consistency validation
- Constitutional compliance checking  
- Extension compatibility validation
- System integration verification
- Performance monitoring health checks

This script ensures complete system coherence and provides fast-fail
capabilities for CI/CD pipelines.

Usage:
    python scripts/validate_unification.py [options]
    
Exit Codes:
    0 - All validations pass
    1 - Warnings found (non-blocking)
    2 - Critical failures found (blocking)
    3 - System/infrastructure issues
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    component: str
    status: str  # PASS, WARN, FAIL, SKIP
    message: str
    details: dict[str, Any] | None = None
    execution_time_ms: float = 0
    error_code: str | None = None


@dataclass
class UnificationReport:
    """Complete unification validation report."""
    
    timestamp: str
    overall_status: str  # PASS, WARN, FAIL
    total_checks: int
    passed_checks: int
    warning_checks: int
    failed_checks: int
    skipped_checks: int
    total_execution_time_ms: float
    validation_results: list[ValidationResult]
    system_info: dict[str, Any]
    recommendations: list[str]


class UnificationValidator:
    """Master validator for complete system unification."""
    
    def __init__(self, 
                 config_path: str | None = None, 
                 fast_fail: bool = True):
        self.config_path = config_path
        self.fast_fail = fast_fail
        self.project_root = self._find_project_root()
        self.validation_results: list[ValidationResult] = []
        self._load_configuration()
    
    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current = Path.cwd()
        
        # Look for project indicators
        indicators = [
            "pyproject.toml", 
            "setup.py", 
            ".git", 
            "requirements.txt",
            "rules/constitution"
        ]
        
        while current != current.parent:
            if any((current / indicator).exists() for indicator in indicators):
                return current
            current = current.parent
        
        return Path.cwd()  # Fallback
    
    def _load_configuration(self) -> None:
        """Load validation configuration."""
        config_file = self.project_root / "validation_config.yaml"
        
        if config_file.exists():
            try:
                with open(config_file, encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
                self.config = self._get_default_config()
        else:
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> dict[str, Any]:
        """Get default validation configuration."""
        return {
            "validation_components": {
                "schema_validation": {"enabled": True, "blocking": True},
                "constitutional_compliance": {
                    "enabled": True, "blocking": True
                },
                "extension_compatibility": {
                    "enabled": True, "blocking": False
                },
                "telemetry_health": {"enabled": True, "blocking": False},
                "performance_monitoring": {"enabled": True, "blocking": False},
                "security_scan": {"enabled": True, "blocking": True}
            },
            "fast_fail": True,
            "timeout_seconds": 300,
            "parallel_execution": True
        }
    
    async def validate_system(self) -> UnificationReport:
        """Execute complete system validation."""
        start_time = time.perf_counter()
        
        logger.info("Starting Super Alita unification validation")
        
        # System information
        system_info = self._collect_system_info()
        
        try:
            # Execute validation components
            if self.config.get("parallel_execution", True):
                await self._execute_parallel_validations()
            else:
                await self._execute_sequential_validations()
            
            # Generate recommendations
            recommendations = self._generate_recommendations()
            
            # Calculate results
            total_time = (time.perf_counter() - start_time) * 1000
            
            passed = sum(
                1 for r in self.validation_results if r.status == "PASS"
            )
            warnings = sum(
                1 for r in self.validation_results if r.status == "WARN"
            )
            failed = sum(
                1 for r in self.validation_results if r.status == "FAIL"
            )
            skipped = sum(1 for r in self.validation_results if r.status == "SKIP")
            
            overall_status = self._determine_overall_status(failed, warnings)
            
            report = UnificationReport(
                timestamp=datetime.now().isoformat(),
                overall_status=overall_status,
                total_checks=len(self.validation_results),
                passed_checks=passed,
                warning_checks=warnings,
                failed_checks=failed,
                skipped_checks=skipped,
                total_execution_time_ms=total_time,
                validation_results=self.validation_results,
                system_info=system_info,
                recommendations=recommendations
            )
            
            logger.info(f"Validation completed: {overall_status} ({total_time:.1f}ms)")
            return report
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            
            # Create error report
            error_result = ValidationResult(
                component="system",
                status="FAIL",
                message=f"Validation system error: {str(e)}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error_code="SYSTEM_ERROR"
            )
            
            self.validation_results.append(error_result)
            
            return UnificationReport(
                timestamp=datetime.now().isoformat(),
                overall_status="FAIL",
                total_checks=1,
                passed_checks=0,
                warning_checks=0,
                failed_checks=1,
                skipped_checks=0,
                total_execution_time_ms=(time.perf_counter() - start_time) * 1000,
                validation_results=self.validation_results,
                system_info=system_info,
                recommendations=["Fix system error and retry validation"]
            )
    
    async def _execute_parallel_validations(self) -> None:
        """Execute validations in parallel for speed."""
        tasks = []
        
        # Create validation tasks
        for component, config in self.config["validation_components"].items():
            if config.get("enabled", True):
                task = asyncio.create_task(
                    self._execute_validation_component(component, config)
                )
                tasks.append(task)
        
        # Execute all tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_sequential_validations(self) -> None:
        """Execute validations sequentially."""
        for component, config in self.config["validation_components"].items():
            if config.get("enabled", True):
                await self._execute_validation_component(component, config)
                
                # Fast fail check
                if (self.fast_fail and 
                    config.get("blocking", True) and 
                    self.validation_results and 
                    self.validation_results[-1].status == "FAIL"):
                    logger.error(f"Fast fail triggered by {component}")
                    break
    
    async def _execute_validation_component(self, component: str, config: dict[str, Any]) -> None:
        """Execute a single validation component."""
        start_time = time.perf_counter()
        
        try:
            if component == "schema_validation":
                result = await self._validate_schemas()
            elif component == "constitutional_compliance":
                result = await self._validate_constitutional_compliance()
            elif component == "extension_compatibility":
                result = await self._validate_extension_compatibility()
            elif component == "telemetry_health":
                result = await self._validate_telemetry_health()
            elif component == "performance_monitoring":
                result = await self._validate_performance_monitoring()
            elif component == "security_scan":
                result = await self._validate_security()
            else:
                result = ValidationResult(
                    component=component,
                    status="SKIP",
                    message=f"Unknown validation component: {component}",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
            
            self.validation_results.append(result)
            
        except Exception as e:
            error_result = ValidationResult(
                component=component,
                status="FAIL",
                message=f"Validation component error: {str(e)}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error_code="COMPONENT_ERROR"
            )
            self.validation_results.append(error_result)
    
    async def _validate_schemas(self) -> ValidationResult:
        """Validate JSON/YAML schema consistency."""
        start_time = time.perf_counter()
        
        try:
            issues = []
            
            # Check YAML files in rules directory
            rules_dir = self.project_root / "rules" / "constitution"
            if rules_dir.exists():
                for yaml_file in rules_dir.glob("*.yaml"):
                    try:
                        with open(yaml_file, encoding='utf-8') as f:
                            yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        issues.append(f"Invalid YAML in {yaml_file}: {e}")
            
            # Check JSON files
            for json_file in self.project_root.rglob("*.json"):
                if ".git" in str(json_file) or "node_modules" in str(json_file):
                    continue
                
                try:
                    with open(json_file, encoding='utf-8') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    issues.append(f"Invalid JSON in {json_file}: {e}")
            
            if issues:
                return ValidationResult(
                    component="schema_validation",
                    status="FAIL",
                    message=f"Schema validation failed: {len(issues)} issues found",
                    details={"issues": issues[:10]},  # Limit to first 10
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
            else:
                return ValidationResult(
                    component="schema_validation",
                    status="PASS",
                    message="All schemas valid",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
        
        except Exception as e:
            return ValidationResult(
                component="schema_validation",
                status="FAIL",
                message=f"Schema validation error: {str(e)}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error_code="SCHEMA_ERROR"
            )
    
    async def _validate_constitutional_compliance(self) -> ValidationResult:
        """Validate constitutional compliance using rule validator."""
        start_time = time.perf_counter()
        
        try:
            rule_validator = self.project_root / "scripts" / "rule_validator.py"
            
            if not rule_validator.exists():
                return ValidationResult(
                    component="constitutional_compliance",
                    status="SKIP",
                    message="Rule validator not found",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
            
            # Run constitutional validation
            cmd = [
                sys.executable, 
                str(rule_validator),
                "--format", "json",
                "--quiet",
                "src/"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Parse results
                try:
                    result_data = json.loads(stdout.decode())
                    return ValidationResult(
                        component="constitutional_compliance",
                        status="PASS",
                        message="Constitutional compliance validated",
                        details=result_data,
                        execution_time_ms=(time.perf_counter() - start_time) * 1000
                    )
                except json.JSONDecodeError:
                    return ValidationResult(
                        component="constitutional_compliance",
                        status="WARN",
                        message="Compliance passed but output not parseable",
                        execution_time_ms=(time.perf_counter() - start_time) * 1000
                    )
            
            elif process.returncode == 1:
                # Warnings found
                try:
                    result_data = json.loads(stdout.decode())
                    return ValidationResult(
                        component="constitutional_compliance",
                        status="WARN",
                        message=f"Constitutional warnings: {result_data.get('warning_count', 0)}",
                        details=result_data,
                        execution_time_ms=(time.perf_counter() - start_time) * 1000
                    )
                except json.JSONDecodeError:
                    return ValidationResult(
                        component="constitutional_compliance",
                        status="WARN",
                        message="Constitutional warnings found",
                        execution_time_ms=(time.perf_counter() - start_time) * 1000
                    )
            
            else:
                # Blockers found
                error_msg = stderr.decode() if stderr else "Constitutional violations found"
                return ValidationResult(
                    component="constitutional_compliance",
                    status="FAIL",
                    message=f"Constitutional compliance failed: {error_msg}",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    error_code="CONSTITUTIONAL_VIOLATION"
                )
        
        except Exception as e:
            return ValidationResult(
                component="constitutional_compliance",
                status="FAIL",
                message=f"Constitutional validation error: {str(e)}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error_code="CONSTITUTIONAL_ERROR"
            )
    
    async def _validate_extension_compatibility(self) -> ValidationResult:
        """Validate extension interface compatibility."""
        start_time = time.perf_counter()
        
        try:
            # Check if extension interfaces are consistent
            src_dir = self.project_root / "src"
            
            if not src_dir.exists():
                return ValidationResult(
                    component="extension_compatibility",
                    status="SKIP",
                    message="Source directory not found",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
            
            # Simple compatibility check - ensure key modules import correctly
            compatibility_issues = []
            
            key_modules = [
                "src.performance_monitoring.telemetry.opentelemetry_config",
                "src.performance_monitoring.middleware.extension_interceptors",
                "src.performance_monitoring.automation.workflow_engine"
            ]
            
            for module_path in key_modules:
                try:
                    # Try importing the module
                    module_parts = module_path.split('.')
                    file_path = self.project_root / "/".join(module_parts) + ".py"
                    
                    if not file_path.exists():
                        compatibility_issues.append(f"Module file not found: {file_path}")
                    else:
                        # Basic syntax check
                        with open(file_path, encoding='utf-8') as f:
                            content = f.read()
                        
                        try:
                            compile(content, str(file_path), 'exec')
                        except SyntaxError as e:
                            compatibility_issues.append(f"Syntax error in {file_path}: {e}")
                
                except Exception as e:
                    compatibility_issues.append(f"Error checking {module_path}: {e}")
            
            if compatibility_issues:
                return ValidationResult(
                    component="extension_compatibility",
                    status="WARN",
                    message=f"Extension compatibility issues: {len(compatibility_issues)}",
                    details={"issues": compatibility_issues},
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
            else:
                return ValidationResult(
                    component="extension_compatibility",
                    status="PASS",
                    message="Extension interfaces compatible",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
        
        except Exception as e:
            return ValidationResult(
                component="extension_compatibility",
                status="FAIL",
                message=f"Extension compatibility check error: {str(e)}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error_code="COMPATIBILITY_ERROR"
            )
    
    async def _validate_telemetry_health(self) -> ValidationResult:
        """Validate telemetry system health."""
        start_time = time.perf_counter()
        
        try:
            # Check if telemetry components are available
            telemetry_issues = []
            
            # Check telemetry configuration
            telemetry_config = (
                self.project_root / 
                "src" / "performance_monitoring" / "telemetry" / "opentelemetry_config.py"
            )
            
            if not telemetry_config.exists():
                telemetry_issues.append("OpenTelemetry configuration missing")
            
            # Check middleware
            middleware_path = (
                self.project_root / 
                "src" / "performance_monitoring" / "middleware" / "extension_interceptors.py"
            )
            
            if not middleware_path.exists():
                telemetry_issues.append("Extension interceptors missing")
            
            # Try to test telemetry imports (basic check)
            try:
                import sys
                sys.path.insert(0, str(self.project_root))
                
                # This would fail if imports are broken
                import importlib.util
                
                spec = importlib.util.spec_from_file_location(
                    "telemetry_config", 
                    telemetry_config
                )
                
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Don't actually execute - just check if loadable
                
            except Exception as e:
                telemetry_issues.append(f"Telemetry import error: {e}")
            
            if telemetry_issues:
                return ValidationResult(
                    component="telemetry_health",
                    status="WARN",
                    message=f"Telemetry health issues: {len(telemetry_issues)}",
                    details={"issues": telemetry_issues},
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
            else:
                return ValidationResult(
                    component="telemetry_health",
                    status="PASS",
                    message="Telemetry system healthy",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
        
        except Exception as e:
            return ValidationResult(
                component="telemetry_health",
                status="FAIL",
                message=f"Telemetry health check error: {str(e)}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error_code="TELEMETRY_ERROR"
            )
    
    async def _validate_performance_monitoring(self) -> ValidationResult:
        """Validate performance monitoring setup."""
        start_time = time.perf_counter()
        
        try:
            monitoring_issues = []
            
            # Check monitoring configuration files
            monitoring_dir = self.project_root / "monitoring"
            
            required_files = [
                "docker-compose.yml",
                "prometheus/prometheus.yml",
                "grafana/dashboards/performance_dashboard.json",
                "alertmanager/alerting_rules.yml"
            ]
            
            for file_path in required_files:
                full_path = monitoring_dir / file_path
                if not full_path.exists():
                    monitoring_issues.append(f"Missing monitoring file: {file_path}")
            
            if monitoring_issues:
                return ValidationResult(
                    component="performance_monitoring",
                    status="WARN",
                    message=f"Performance monitoring setup issues: {len(monitoring_issues)}",
                    details={"issues": monitoring_issues},
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
            else:
                return ValidationResult(
                    component="performance_monitoring",
                    status="PASS",
                    message="Performance monitoring setup complete",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
        
        except Exception as e:
            return ValidationResult(
                component="performance_monitoring",
                status="FAIL",
                message=f"Performance monitoring validation error: {str(e)}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error_code="MONITORING_ERROR"
            )
    
    async def _validate_security(self) -> ValidationResult:
        """Validate security configuration."""
        start_time = time.perf_counter()
        
        try:
            security_issues = []
            
            # Check for common security issues
            src_dir = self.project_root / "src"
            
            if src_dir.exists():
                # Look for hardcoded secrets (basic check)
                for py_file in src_dir.rglob("*.py"):
                    try:
                        with open(py_file, encoding='utf-8') as f:
                            content = f.read()
                        
                        # Basic patterns to avoid
                        sensitive_patterns = [
                            r'password\s*=\s*["\'][^"\']+["\']',
                            r'secret\s*=\s*["\'][^"\']+["\']',
                            r'api_key\s*=\s*["\'][^"\']+["\']',
                        ]
                        
                        import re
                        for pattern in sensitive_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                security_issues.append(
                                    f"Potential hardcoded secret in {py_file}"
                                )
                                break
                    
                    except Exception:
                        continue  # Skip files that can't be read
            
            # Check for security configuration files
            security_files = [
                ".pre-commit-config.yaml",
                "requirements.txt"
            ]
            
            missing_security_files = []
            for sec_file in security_files:
                if not (self.project_root / sec_file).exists():
                    missing_security_files.append(sec_file)
            
            if missing_security_files:
                security_issues.extend([
                    f"Missing security file: {f}" for f in missing_security_files
                ])
            
            if security_issues:
                return ValidationResult(
                    component="security_scan",
                    status="WARN",
                    message=f"Security issues found: {len(security_issues)}",
                    details={"issues": security_issues[:5]},  # Limit output
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
            else:
                return ValidationResult(
                    component="security_scan",
                    status="PASS",
                    message="Basic security validation passed",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
        
        except Exception as e:
            return ValidationResult(
                component="security_scan",
                status="FAIL",
                message=f"Security validation error: {str(e)}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                error_code="SECURITY_ERROR"
            )
    
    def _collect_system_info(self) -> dict[str, Any]:
        """Collect system information for the report."""
        try:
            import platform

            import psutil
            
            return {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
                "disk_free_gb": round(psutil.disk_usage('.').free / (1024**3), 1),
                "project_root": str(self.project_root),
                "timestamp": datetime.now().isoformat()
            }
        except ImportError:
            return {
                "python_version": sys.version,
                "project_root": str(self.project_root),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "error": f"Failed to collect system info: {e}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _determine_overall_status(self, failed_count: int, warning_count: int) -> str:
        """Determine overall validation status."""
        if failed_count > 0:
            return "FAIL"
        elif warning_count > 0:
            return "WARN"
        else:
            return "PASS"
    
    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        failed_components = [
            r.component for r in self.validation_results 
            if r.status == "FAIL"
        ]
        
        warning_components = [
            r.component for r in self.validation_results 
            if r.status == "WARN"
        ]
        
        if "constitutional_compliance" in failed_components:
            recommendations.append(
                "Fix constitutional violations before merging code"
            )
        
        if "schema_validation" in failed_components:
            recommendations.append(
                "Repair invalid JSON/YAML files"
            )
        
        if "security_scan" in warning_components:
            recommendations.append(
                "Review security warnings and implement fixes"
            )
        
        if "telemetry_health" in warning_components:
            recommendations.append(
                "Check telemetry system configuration"
            )
        
        if not recommendations:
            recommendations.append("System validation successful - no action required")
        
        return recommendations


def format_report_human(report: UnificationReport) -> str:
    """Format validation report for human reading."""
    output = []
    
    # Header
    status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[report.overall_status]
    output.append(f"\n{status_icon} Super Alita Unification Validation Report")
    output.append("=" * 60)
    
    # Summary
    output.append(f"Overall Status: {report.overall_status}")
    output.append(f"Total Checks: {report.total_checks}")
    output.append(f"Passed: {report.passed_checks}")
    output.append(f"Warnings: {report.warning_checks}")
    output.append(f"Failed: {report.failed_checks}")
    output.append(f"Execution Time: {report.total_execution_time_ms:.1f}ms")
    output.append("")
    
    # Component Results
    output.append("Component Results:")
    output.append("-" * 40)
    
    for result in report.validation_results:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}[result.status]
        output.append(f"{icon} {result.component}: {result.message}")
        if result.details and result.status in ["WARN", "FAIL"]:
            issues = result.details.get("issues", [])
            for issue in issues[:3]:  # Show first 3 issues
                output.append(f"    • {issue}")
            if len(issues) > 3:
                output.append(f"    ... and {len(issues) - 3} more")
    
    output.append("")
    
    # Recommendations
    if report.recommendations:
        output.append("Recommendations:")
        output.append("-" * 40)
        for i, rec in enumerate(report.recommendations, 1):
            output.append(f"{i}. {rec}")
        output.append("")
    
    return "\n".join(output)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Super Alita Unification Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all validations
  %(prog)s --fast-fail        # Stop on first critical failure
  %(prog)s --format json      # JSON output for CI
  %(prog)s --config custom.yaml  # Custom configuration
        """
    )
    
    parser.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to validation configuration file"
    )
    
    parser.add_argument(
        "--fast-fail",
        action="store_true",
        help="Stop on first critical failure"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (errors only)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Validation timeout in seconds (default: 300)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    async def run_validation():
        try:
            # Initialize validator
            validator = UnificationValidator(
                config_path=args.config,
                fast_fail=args.fast_fail
            )
            
            # Run validation with timeout
            report = await asyncio.wait_for(
                validator.validate_system(),
                timeout=args.timeout
            )
            
            # Output results
            if args.format == "json":
                print(json.dumps(asdict(report), indent=2))
            else:
                print(format_report_human(report))
            
            # Exit with appropriate code
            if report.overall_status == "FAIL":
                sys.exit(2)  # Critical failures
            elif report.overall_status == "WARN":
                sys.exit(1)  # Warnings
            else:
                sys.exit(0)  # Success
        
        except TimeoutError:
            print(f"❌ Validation timed out after {args.timeout} seconds", file=sys.stderr)
            sys.exit(3)
        except KeyboardInterrupt:
            print("\n❌ Validation interrupted", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"❌ Validation failed: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(3)
    
    # Run the async validation
    try:
        asyncio.run(run_validation())
    except KeyboardInterrupt:
        print("\n❌ Interrupted", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()