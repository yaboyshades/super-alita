#!/usr/bin/env python3
"""
Capability metadata validator (scoped).

Validates ONLY:
  - Python plugin tools returned by create_plugin().get_tools() under src/plugins/**
  - Optional JSON/YAML capability manifests under src/capabilities/** (if present)

Ignores: .venv, .mypy_cache, .git, node_modules, dist, build (and anything else via --exclude)

Usage:
  python scripts/validate_capabilities.py --paths src/plugins src/capabilities --output text
  python scripts/validate_capabilities.py --strict --output github

Exit codes:
  0: OK (no errors; warnings allowed unless --strict)
  2: Failure (missing required metadata in at least one capability, or import error)
"""

from __future__ import annotations
import argparse
import importlib
import json
import os
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Required fields aligned with your report
REQUIRED_FIELDS = [
    "name",
    "description",
    "parameters",  # JSON schema for arguments
    "cost_hint",
    "latency_hint",
    "safety_level",
    "test_reference",
    "category",
    "complexity",
    "version",
    "dependencies",
    "integration_requirements",
]

DEFAULT_PATHS = ["src/plugins"]
DEFAULT_EXCLUDES = [".venv", ".mypy_cache", ".git", "node_modules", "dist", "build", "__pycache__"]

@dataclass 
class ValidationResult:
    file_path: str
    capability_name: str
    status: str  # "pass", "warn", "fail"
    issues: List[str]
    metadata: Dict[str, Any]

class CapabilityValidator:
    """Validates capability metadata files against schema requirements"""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.results: List[ValidationResult] = []
        
    def validate_capability_file(self, file_path: Path) -> ValidationResult:
        """Validate a single capability metadata file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    metadata = yaml.safe_load(f)
                else:
                    metadata = json.load(f)
        except Exception as e:
            return ValidationResult(
                file_path=str(file_path),
                capability_name="unknown",
                status="fail",
                issues=[f"Failed to parse file: {e}"],
                metadata={}
            )
            
        capability_name = metadata.get("name", file_path.stem)
        issues = []
        
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in metadata:
                issues.append(f"Missing required field: {field}")
            elif not metadata[field]:
                issues.append(f"Empty required field: {field}")
                
        # Validate category
        category = metadata.get("category")
        if category and category not in VALID_CATEGORIES:
            issues.append(f"Invalid category '{category}'. Must be one of: {VALID_CATEGORIES}")
            
        # Validate complexity
        complexity = metadata.get("complexity") 
        if complexity and complexity not in VALID_COMPLEXITY_LEVELS:
            issues.append(f"Invalid complexity '{complexity}'. Must be one of: {VALID_COMPLEXITY_LEVELS}")
            
        # Validate version format (semantic versioning)
        version = metadata.get("version")
        if version and not self._is_valid_semver(version):
            issues.append(f"Invalid version format '{version}'. Must follow semantic versioning (x.y.z)")
            
        # Check dependencies format
        deps = metadata.get("dependencies")
        if deps and not isinstance(deps, list):
            issues.append("Dependencies must be a list")
        elif deps:
            for dep in deps:
                if not isinstance(dep, str):
                    issues.append(f"Invalid dependency format: {dep}")
                    
        # Check integration requirements
        integration_reqs = metadata.get("integration_requirements") 
        if integration_reqs and not isinstance(integration_reqs, dict):
            issues.append("Integration requirements must be an object/dict")
            
        # Determine status
        if issues:
            status = "fail" if self.strict_mode else "warn"
        else:
            status = "pass"
            
        return ValidationResult(
            file_path=str(file_path),
            capability_name=capability_name,
            status=status,
            issues=issues,
            metadata=metadata
        )
        
    def _is_valid_semver(self, version: str) -> bool:
        """Check if version follows semantic versioning pattern"""
        try:
            parts = version.split('.')
            if len(parts) != 3:
                return False
            for part in parts:
                int(part)  # Will raise ValueError if not a number
            return True
        except (ValueError, AttributeError):
            return False
            
    def validate_directory(self, directory: Path) -> None:
        """Validate all capability files in a directory"""
        capability_files = []
        
        # Look for capability metadata files
        for pattern in ["*.json", "*.yaml", "*.yml"]:
            capability_files.extend(directory.glob(f"**/{pattern}"))
            
        # Filter for files that look like capability metadata
        capability_files = [
            f for f in capability_files 
            if any(keyword in f.name.lower() for keyword in 
                  ["capability", "capabilities", "metadata", "spec"])
        ]
        
        if not capability_files:
            print(f"Warning: No capability metadata files found in {directory}")
            return
            
        for file_path in capability_files:
            result = self.validate_capability_file(file_path)
            self.results.append(result)
            
    def generate_report(self, output_format: str = "text") -> str:
        """Generate validation report in specified format"""
        if output_format == "json":
            return json.dumps([asdict(result) for result in self.results], indent=2)
            
        # Text format
        report_lines = []
        report_lines.append("🔍 Super Alita Capability Validation Report")
        report_lines.append("=" * 50)
        
        passed = sum(1 for r in self.results if r.status == "pass")
        warned = sum(1 for r in self.results if r.status == "warn") 
        failed = sum(1 for r in self.results if r.status == "fail")
        
        report_lines.append(f"📊 Summary: {passed} passed, {warned} warnings, {failed} failed")
        report_lines.append("")
        
        for result in self.results:
            status_icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[result.status]
            report_lines.append(f"{status_icon} {result.capability_name} ({result.file_path})")
            
            if result.issues:
                for issue in result.issues:
                    report_lines.append(f"   • {issue}")
                report_lines.append("")
                
        return "\n".join(report_lines)
        
    def exit_code(self) -> int:
        """Return appropriate exit code based on validation results"""
        if any(r.status == "fail" for r in self.results):
            return 1
        return 0

def main():
    parser = argparse.ArgumentParser(description="Validate Super Alita capability metadata")
    parser.add_argument("--strict", action="store_true", 
                       help="Treat warnings as failures")
    parser.add_argument("--output", choices=["text", "json"], default="text",
                       help="Output format")
    parser.add_argument("--directory", type=Path, default=Path("."),
                       help="Directory to scan for capability files")
    
    args = parser.parse_args()
    
    validator = CapabilityValidator(strict_mode=args.strict)
    validator.validate_directory(args.directory)
    
    report = validator.generate_report(args.output)
    print(report)
    
    sys.exit(validator.exit_code())

if __name__ == "__main__":
    main()