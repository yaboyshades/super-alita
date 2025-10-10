#!/usr/bin/env python3
"""
Constitutional Rule Validator

CLI tool for validating constitutional compliance according to the six-article
framework. Provides structured JSON reports and human-friendly error messages.

Usage:
    python scripts/rule_validator.py [options] [path]
    
Exit Codes:
    0 - All rules pass
    1 - Warnings found (non-blocking)
    2 - Blockers found (must be resolved)
"""

import argparse
import ast
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class RuleViolation:
    """Represents a constitutional rule violation."""
    
    rule_id: str
    article: str
    severity: str  # BLOCKER, WARNING, INFO
    category: str
    description: str
    file_path: str
    line_number: int | None = None
    column_number: int | None = None
    suggestion: str | None = None
    auto_fix_available: bool = False
    context: str | None = None


@dataclass
class ValidationResult:
    """Complete validation result."""
    
    success: bool
    total_files_checked: int
    total_violations: int
    blocker_count: int
    warning_count: int
    info_count: int
    violations: list[RuleViolation]
    execution_time_ms: float
    timestamp: str
    ruleset_version: str


class ConstitutionalRuleEngine:
    """Engine for validating constitutional compliance rules."""
    
    def __init__(self, rules_dir: str = "rules/constitution"):
        self.rules_dir = Path(rules_dir)
        self.rules: dict[str, dict[str, Any]] = {}
        self.ruleset_metadata: dict[str, Any] = {}
        self._load_rules()
    
    def _load_rules(self) -> None:
        """Load constitutional rules from YAML files."""
        try:
            # Load ruleset metadata
            metadata_file = self.rules_dir / "ruleset_metadata.yaml"
            if metadata_file.exists():
                with open(metadata_file, encoding='utf-8') as f:
                    self.ruleset_metadata = yaml.safe_load(f)
            
            # Load individual rule files
            for rule_file in self.rules_dir.glob("article_*.yaml"):
                try:
                    with open(rule_file, encoding='utf-8') as f:
                        rule_data = yaml.safe_load(f)
                        rule_id = rule_data.get('id')
                        if rule_id:
                            self.rules[rule_id] = rule_data
                            logger.debug(
                                f"Loaded rule {rule_id} from {rule_file}"
                            )
                
                except yaml.YAMLError as e:
                    logger.error(f"Error parsing rule file {rule_file}: {e}")
                except Exception as e:
                    logger.error(f"Error loading rule file {rule_file}: {e}")
            
            logger.info(f"Loaded {len(self.rules)} constitutional rules")
            
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            self.rules = {}
    
    def validate_path(self, target_path: str) -> ValidationResult:
        """Validate a file or directory against constitutional rules."""
        start_time = time.perf_counter()
        
        violations = []
        files_checked = 0
        
        target = Path(target_path)
        
        if target.is_file():
            files_checked = 1
            violations.extend(self._validate_file(target))
        elif target.is_dir():
            # Recursively validate directory
            for file_path in self._get_target_files(target):
                files_checked += 1
                violations.extend(self._validate_file(file_path))
        else:
            raise ValueError(f"Target path does not exist: {target_path}")
        
        # Calculate execution time
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Count violations by severity
        blocker_count = sum(1 for v in violations if v.severity == "BLOCKER")
        warning_count = sum(1 for v in violations if v.severity == "WARNING")
        info_count = sum(1 for v in violations if v.severity == "INFO")
        
        success = blocker_count == 0
        
        return ValidationResult(
            success=success,
            total_files_checked=files_checked,
            total_violations=len(violations),
            blocker_count=blocker_count,
            warning_count=warning_count,
            info_count=info_count,
            violations=violations,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now().isoformat(),
            ruleset_version=(
                self.ruleset_metadata.get("ruleset", {})
                .get("version", "unknown")
            )
        )
    
    def _get_target_files(self, directory: Path) -> list[Path]:
        """Get list of files to validate in directory."""
        target_files = []
        
        # Common source file patterns
        patterns = ["*.py", "*.js", "*.ts", "*.yaml", "*.yml", "*.json"]
        
        for pattern in patterns:
            target_files.extend(directory.rglob(pattern))
        
        # Filter out common exclusions
        exclusions = {
            "__pycache__",
            ".git",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            "venv",
            "env"
        }
        
        return [
            f for f in target_files 
            if not any(exc in str(f) for exc in exclusions)
        ]
    
    def _validate_file(self, file_path: Path) -> list[RuleViolation]:
        """Validate a single file against all applicable rules."""
        violations = []
        
        try:
            # Read file content
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Apply each rule
            for rule_id, rule_data in self.rules.items():
                rule_violations = self._apply_rule(
                    rule_data, file_path, content
                )
                violations.extend(rule_violations)
        
        except Exception as e:
            logger.warning(f"Error validating file {file_path}: {e}")
        
        return violations
    
    def _apply_rule(
        self, 
        rule_data: dict[str, Any], 
        file_path: Path, 
        content: str
    ) -> list[RuleViolation]:
        """Apply a specific rule to file content."""
        violations = []
        
        # These will be used by individual check methods
        # rule_id = rule_data.get("id", "unknown")
        # article = rule_data.get("article", "unknown")
        # severity = rule_data.get("severity", "WARNING")
        # category = rule_data.get("category", "general")
        
        # Check if file should be excluded
        if self._is_file_excluded(rule_data, file_path):
            return violations
        
        # Apply rule checks based on type
        checks = rule_data.get("checks", [])
        
        for check in checks:
            check_type = check.get("type", "unknown")
            
            if check_type == "test_coverage":
                violations.extend(
                    self._check_test_coverage(rule_data, file_path, content)
                )
            elif check_type == "complexity":
                violations.extend(
                    self._check_complexity(rule_data, file_path, content)
                )
            elif check_type == "docstring_coverage":
                violations.extend(
                    self._check_docstring_coverage(rule_data, file_path, content)
                )
            elif check_type == "integration_test":
                violations.extend(
                    self._check_integration_tests(rule_data, file_path, content)
                )
            elif check_type == "breaking_change_detection":
                violations.extend(
                    self._check_breaking_changes(rule_data, file_path, content)
                )
            elif check_type == "code_pattern":
                violations.extend(
                    self._check_code_patterns(rule_data, file_path, content)
                )
        
        return violations
    
    def _is_file_excluded(self, rule_data: dict[str, Any], file_path: Path) -> bool:
        """Check if file is excluded from rule validation."""
        exceptions = rule_data.get("exceptions", [])
        
        for exception in exceptions:
            if "path" in exception:
                if str(file_path).startswith(exception["path"]):
                    return True
            
            if "pattern" in exception:
                if re.search(exception["pattern"], str(file_path)):
                    return True
        
        return False
    
    def _check_test_coverage(self, rule_data: dict[str, Any], file_path: Path, content: str) -> list[RuleViolation]:
        """Check Article II: Test-First compliance."""
        violations = []
        
        if not str(file_path).endswith('.py'):
            return violations
        
        if 'test' in str(file_path):
            return violations  # Skip test files themselves
        
        # Find public functions
        function_pattern = r'^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        functions = re.findall(function_pattern, content, re.MULTILINE)
        
        # Filter out private functions
        public_functions = [f for f in functions if not f.startswith('_')]
        
        for func_name in public_functions:
            # Check if corresponding test exists
            test_patterns = [
                f"test_{func_name}",
                f"{func_name}_test",
                f"test.*{func_name}"
            ]
            
            has_test = self._find_test_for_function(file_path, func_name, test_patterns)
            
            if not has_test:
                violations.append(RuleViolation(
                    rule_id=rule_data["id"],
                    article=rule_data["article"],
                    severity=rule_data["severity"],
                    category=rule_data["category"],
                    description=f"Function '{func_name}' lacks unit tests",
                    file_path=str(file_path),
                    suggestion=f"Create test: tests/test_{file_path.stem}.py::test_{func_name}()",
                    auto_fix_available=False
                ))
        
        return violations
    
    def _find_test_for_function(self, file_path: Path, func_name: str, test_patterns: list[str]) -> bool:
        """Check if test exists for a function."""
        # Look for test files
        test_dirs = ["tests", "test"]
        project_root = self._find_project_root(file_path)
        
        for test_dir in test_dirs:
            test_path = project_root / test_dir
            if test_path.exists():
                # Look for test files
                for test_file in test_path.rglob("*.py"):
                    try:
                        with open(test_file, encoding='utf-8') as f:
                            test_content = f.read()
                        
                        # Check if any test pattern matches
                        for pattern in test_patterns:
                            if re.search(pattern, test_content):
                                return True
                    except Exception:
                        continue
        
        return False
    
    def _find_project_root(self, file_path: Path) -> Path:
        """Find project root directory."""
        current = file_path.parent if file_path.is_file() else file_path
        
        # Look for common project root indicators
        indicators = ["pyproject.toml", "setup.py", ".git", "requirements.txt"]
        
        while current != current.parent:
            if any((current / indicator).exists() for indicator in indicators):
                return current
            current = current.parent
        
        return file_path.parent  # Fallback
    
    def _check_complexity(self, rule_data: dict[str, Any], file_path: Path, content: str) -> list[RuleViolation]:
        """Check Article III: Simplicity compliance."""
        violations = []
        
        if not str(file_path).endswith('.py'):
            return violations
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check function length
                    func_lines = node.end_lineno - node.lineno + 1
                    if func_lines > 50:  # Threshold from rule
                        violations.append(RuleViolation(
                            rule_id=rule_data["id"],
                            article=rule_data["article"],
                            severity=rule_data["severity"],
                            category=rule_data["category"],
                            description=f"Function '{node.name}' is too long ({func_lines} lines)",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            suggestion="Break down into smaller, focused functions",
                            auto_fix_available=False
                        ))
                    
                    # Check parameter count
                    param_count = len(node.args.args)
                    if param_count > 5:  # Threshold from rule
                        violations.append(RuleViolation(
                            rule_id=rule_data["id"],
                            article=rule_data["article"],
                            severity=rule_data["severity"],
                            category=rule_data["category"],
                            description=f"Function '{node.name}' has too many parameters ({param_count})",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            suggestion="Group related parameters into a class or reduce complexity",
                            auto_fix_available=False
                        ))
        
        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception as e:
            logger.debug(f"Error analyzing {file_path}: {e}")
        
        return violations
    
    def _check_docstring_coverage(self, rule_data: dict[str, Any], file_path: Path, content: str) -> list[RuleViolation]:
        """Check Article V: Clarity compliance."""
        violations = []
        
        if not str(file_path).endswith('.py'):
            return violations
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Skip private functions/classes
                    if node.name.startswith('_'):
                        continue
                    
                    # Check for docstring
                    has_docstring = (
                        node.body and
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str)
                    )
                    
                    if not has_docstring:
                        node_type = "Function" if isinstance(node, ast.FunctionDef) else "Class"
                        violations.append(RuleViolation(
                            rule_id=rule_data["id"],
                            article=rule_data["article"],
                            severity=rule_data["severity"],
                            category=rule_data["category"],
                            description=f"{node_type} '{node.name}' lacks documentation",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            suggestion="Add docstring explaining purpose, parameters, and return value",
                            auto_fix_available=False
                        ))
        
        except SyntaxError:
            pass
        except Exception as e:
            logger.debug(f"Error analyzing {file_path}: {e}")
        
        return violations
    
    def _check_integration_tests(self, rule_data: dict[str, Any], file_path: Path, content: str) -> list[RuleViolation]:
        """Check Article IV: Integration-First compliance."""
        violations = []
        
        # This is a simplified check - would need more sophisticated analysis
        project_root = self._find_project_root(file_path)
        test_dir = project_root / "tests"
        
        if not test_dir.exists():
            violations.append(RuleViolation(
                rule_id=rule_data["id"],
                article=rule_data["article"],
                severity=rule_data["severity"],
                category=rule_data["category"],
                description="No tests directory found",
                file_path=str(file_path),
                suggestion="Create tests/ directory with integration tests",
                auto_fix_available=False
            ))
        
        return violations
    
    def _check_breaking_changes(self, rule_data: dict[str, Any], file_path: Path, content: str) -> list[RuleViolation]:
        """Check Article VI: Versioning compliance."""
        violations = []
        
        # This would require git history analysis in a real implementation
        # For now, just check for version files
        version_files = ["pyproject.toml", "setup.py", "package.json"]
        project_root = self._find_project_root(file_path)
        
        has_version_file = any(
            (project_root / vf).exists() for vf in version_files
        )
        
        if not has_version_file:
            violations.append(RuleViolation(
                rule_id=rule_data["id"],
                article=rule_data["article"],
                severity=rule_data["severity"],
                category=rule_data["category"],
                description="No version management file found",
                file_path=str(file_path),
                suggestion="Add pyproject.toml or setup.py with version information",
                auto_fix_available=False
            ))
        
        return violations
    
    def _check_code_patterns(self, rule_data: dict[str, Any], file_path: Path, content: str) -> list[RuleViolation]:
        """Check Article I: Library-First compliance."""
        violations = []
        
        # Check for common anti-patterns that suggest custom implementations
        anti_patterns = [
            (r'def\s+download\s*\(', "Consider using requests library instead of custom download"),
            (r'def\s+parse_json\s*\(', "Consider using json.loads() instead of custom parser"),
            (r'def\s+format_date\s*\(', "Consider using datetime.strftime() instead of custom formatter"),
            (r'def\s+hash_\w+\s*\(', "Consider using hashlib instead of custom hash function"),
        ]
        
        for pattern, suggestion in anti_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                violations.append(RuleViolation(
                    rule_id=rule_data["id"],
                    article=rule_data["article"],
                    severity=rule_data["severity"],
                    category=rule_data["category"],
                    description="Potential custom implementation detected",
                    file_path=str(file_path),
                    line_number=line_num,
                    suggestion=suggestion,
                    auto_fix_available=False
                ))
        
        return violations


def format_human_readable(result: ValidationResult) -> str:
    """Format validation result for human-readable output."""
    output = []
    
    # Header
    status = "✅ PASS" if result.success else "❌ FAIL"
    output.append(f"\n{status} Constitutional Validation Report")
    output.append("=" * 50)
    
    # Summary
    output.append(f"Files checked: {result.total_files_checked}")
    output.append(f"Total violations: {result.total_violations}")
    output.append(f"Blockers: {result.blocker_count}")
    output.append(f"Warnings: {result.warning_count}")
    output.append(f"Execution time: {result.execution_time_ms:.1f}ms")
    output.append("")
    
    # Violations by severity
    if result.violations:
        # Group by severity
        by_severity = {"BLOCKER": [], "WARNING": [], "INFO": []}
        for violation in result.violations:
            by_severity[violation.severity].append(violation)
        
        for severity in ["BLOCKER", "WARNING", "INFO"]:
            violations = by_severity[severity]
            if not violations:
                continue
            
            icon = {"BLOCKER": "🚫", "WARNING": "⚠️", "INFO": "ℹ️"}[severity]
            output.append(f"{icon} {severity} ({len(violations)})")
            output.append("-" * 30)
            
            for violation in violations:
                output.append(f"  {violation.file_path}:{violation.line_number or '?'}")
                output.append(f"    {violation.description}")
                if violation.suggestion:
                    output.append(f"    💡 {violation.suggestion}")
                output.append("")
    
    return "\n".join(output)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Constitutional Rule Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s src/                    # Validate src directory
  %(prog)s --format json src/      # JSON output
  %(prog)s --rules custom/ src/    # Custom rules directory
        """
    )
    
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to validate (file or directory, default: current directory)"
    )
    
    parser.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)"
    )
    
    parser.add_argument(
        "--rules",
        default="rules/constitution",
        help="Rules directory (default: rules/constitution)"
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
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    try:
        # Initialize rule engine
        engine = ConstitutionalRuleEngine(args.rules)
        
        if not engine.rules:
            print("❌ No rules loaded. Check rules directory.", file=sys.stderr)
            sys.exit(2)
        
        # Validate target path
        result = engine.validate_path(args.path)
        
        # Output results
        if args.format == "json":
            # Convert dataclasses to dict for JSON serialization
            result_dict = asdict(result)
            print(json.dumps(result_dict, indent=2))
        else:
            print(format_human_readable(result))
        
        # Exit with appropriate code
        if result.blocker_count > 0:
            sys.exit(2)  # Blockers found
        elif result.warning_count > 0:
            sys.exit(1)  # Warnings found
        else:
            sys.exit(0)  # All good
    
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()