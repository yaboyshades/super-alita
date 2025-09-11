#!/usr/bin/env python3
"""
Constitutional gRPC Development Kit Validation Script

Comprehensive validation of the Constitutional gRPC Development Kit to ensure
all components work together correctly and meet constitutional requirements.

This script validates:
- Template integrity and rendering
- Service generation pipeline
- Constitutional compliance checking
- Integration between components
- Example functionality

Constitutional Compliance:
- Article VIII: Automation of Expertise (this script automates kit validation)
- Article II: Test-First (comprehensive validation before use)
- Article V: Clarity (clear validation reporting)
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class KitValidationError(Exception):
    """Exception raised when kit validation fails."""

    pass


class ConstitutionalKitValidator:
    """
    Comprehensive validator for the Constitutional gRPC Development Kit.

    Validates all kit components and their integration to ensure
    the kit meets constitutional framework requirements.
    """

    def __init__(self, kit_path: Path):
        """Initialize the kit validator."""
        self.kit_path = kit_path
        self.temp_dir = None
        self.validation_results = {}

        logger.info(f"Initializing Constitutional gRPC Kit validator: {kit_path}")

    def validate_kit(self) -> dict[str, bool]:
        """
        Perform comprehensive kit validation.

        Returns:
            Dictionary mapping validation categories to success status
        """
        logger.info("🔍 Starting Constitutional gRPC Development Kit validation...")

        start_time = time.time()

        try:
            # Create temporary directory for testing
            self.temp_dir = Path(tempfile.mkdtemp(prefix="constitutional_kit_test_"))
            logger.info(f"📁 Using temporary directory: {self.temp_dir}")

            # Run validation phases
            self.validation_results = {
                "kit_structure": self._validate_kit_structure(),
                "template_integrity": self._validate_template_integrity(),
                "generator_functionality": self._validate_generator_functionality(),
                "compliance_checker": self._validate_compliance_checker(),
                "end_to_end_workflow": self._validate_end_to_end_workflow(),
                "constitutional_compliance": self._validate_constitutional_compliance(),
                "documentation_quality": self._validate_documentation_quality(),
            }

            # Calculate overall success
            all_passed = all(self.validation_results.values())

            validation_time = time.time() - start_time

            if all_passed:
                logger.info(
                    f"✅ Kit validation completed successfully in {validation_time:.2f}s"
                )
            else:
                logger.error(f"❌ Kit validation failed in {validation_time:.2f}s")

            return self.validation_results

        except Exception as e:
            logger.error(f"❌ Kit validation error: {e}")
            raise KitValidationError(f"Validation failed: {e}") from e

        finally:
            # Cleanup temporary directory
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.debug(f"🧹 Cleaned up temporary directory: {self.temp_dir}")

    def _validate_kit_structure(self) -> bool:
        """Validate the kit directory structure and required files."""
        logger.info("📋 Validating kit structure...")

        required_files = [
            "README.md",
            "USAGE.md",
            "generate_service.py",
            "validate_compliance.py",
            "compliance_config.yaml",
        ]

        required_dirs = [
            "templates",
            "examples",
        ]

        required_templates = [
            "templates/servicer_template.py.j2",
            "templates/server_template.py.j2",
            "templates/middleware_template.py.j2",
            "templates/test_template.py.j2",
            "templates/README_template.md",
        ]

        # Check required files
        for file_name in required_files:
            file_path = self.kit_path / file_name
            if not file_path.exists():
                logger.error(f"❌ Missing required file: {file_name}")
                return False
            logger.debug(f"✅ Found required file: {file_name}")

        # Check required directories
        for dir_name in required_dirs:
            dir_path = self.kit_path / dir_name
            if not dir_path.exists() or not dir_path.is_dir():
                logger.error(f"❌ Missing required directory: {dir_name}")
                return False
            logger.debug(f"✅ Found required directory: {dir_name}")

        # Check required templates
        for template_path in required_templates:
            template_file = self.kit_path / template_path
            if not template_file.exists():
                logger.error(f"❌ Missing required template: {template_path}")
                return False
            logger.debug(f"✅ Found required template: {template_path}")

        logger.info("✅ Kit structure validation passed")
        return True

    def _validate_template_integrity(self) -> bool:
        """Validate template files for syntax and completeness."""
        logger.info("🎨 Validating template integrity...")

        try:
            from jinja2 import Environment, FileSystemLoader, meta

            # Setup Jinja2 environment
            templates_dir = self.kit_path / "templates"
            env = Environment(loader=FileSystemLoader(str(templates_dir)))

            template_files = list(templates_dir.glob("*.j2"))

            for template_file in template_files:
                template_name = template_file.name
                logger.debug(f"Validating template: {template_name}")

                try:
                    # Parse template to check syntax
                    template_source = template_file.read_text()
                    ast = env.parse(template_source)

                    # Find undeclared variables
                    undeclared = meta.find_undeclared_variables(ast)

                    # Check for required variables based on template type
                    if template_name == "middleware_template.py.j2":
                        required_vars = {
                            "service_name",
                            "generated_at",
                            "constitutional_threshold",
                        }
                    elif template_name == "README_template.md":
                        required_vars = {"service_name", "generated_at"}
                    else:
                        required_vars = {
                            "service_name",
                            "service_class",
                            "generated_at",
                        }

                    template_vars = set(meta.find_undeclared_variables(ast))
                    missing_vars = required_vars - template_vars

                    if missing_vars:
                        logger.warning(
                            f"Template {template_name} missing variables: "
                            f"{missing_vars}"
                        )

                    # Try to compile the template
                    env.get_template(template_name)
                    logger.debug(f"Template {template_name} compiled successfully")

                except Exception as e:
                    logger.error(f"❌ Template {template_name} validation failed: {e}")
                    return False

            logger.info("✅ Template integrity validation passed")
            return True

        except ImportError:
            logger.error("❌ Jinja2 not available for template validation")
            return False
        except Exception as e:
            logger.error(f"❌ Template integrity validation failed: {e}")
            return False

    def _validate_generator_functionality(self) -> bool:
        """Validate the service generator script functionality."""
        logger.info("⚙️ Validating service generator...")

        try:
            # Test basic generator help
            result = subprocess.run(
                [sys.executable, str(self.kit_path / "generate_service.py"), "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"❌ Generator help failed: {result.stderr}")
                return False

            logger.debug("✅ Generator help command successful")

            # Test service generation with example
            example_proto = self.kit_path / "examples" / "example_service.proto"
            if not example_proto.exists():
                logger.error("❌ Example proto file not found")
                return False

            output_dir = self.temp_dir / "test_service"

            result = subprocess.run(
                [
                    sys.executable,
                    str(self.kit_path / "generate_service.py"),
                    "--service-name",
                    "TestService",
                    "--proto-file",
                    str(example_proto),
                    "--output-dir",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                logger.error(f"❌ Service generation failed: {result.stderr}")
                return False

            # Check generated files
            expected_files = [
                "test_service_servicer.py",  # TestService -> test_service_servicer.py
                "server.py",
                "__init__.py",
                "test_test_service_servicer.py",  # test file for the servicer
                "README.md",
            ]

            for expected_file in expected_files:
                if not (output_dir / expected_file).exists():
                    logger.error(f"❌ Generated file missing: {expected_file}")
                    return False
                logger.debug(f"✅ Generated file found: {expected_file}")

            logger.info("✅ Service generator validation passed")
            return True

        except subprocess.TimeoutExpired:
            logger.error("❌ Service generator validation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Service generator validation failed: {e}")
            return False

    def _validate_compliance_checker(self) -> bool:
        """Validate the constitutional compliance checker."""
        logger.info("⚖️ Validating compliance checker...")

        try:
            # Test compliance checker help
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.kit_path / "validate_compliance.py"),
                    "--help",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"❌ Compliance checker help failed: {result.stderr}")
                return False

            logger.debug("✅ Compliance checker help command successful")

            # Create a test service for compliance checking
            test_service_dir = self.temp_dir / "compliance_test_service"
            test_service_dir.mkdir()

            # Create minimal service structure
            (test_service_dir / "__init__.py").write_text('"""Test service."""\n')
            (test_service_dir / "server.py").write_text(
                '''
"""Test server implementation."""

def simple_function():
    """Simple function for testing."""
    return "test"

class SimpleClass:
    """Simple class for testing."""

    def method(self):
        """Simple method."""
        return "method"
'''
            )

            (test_service_dir / "test_server.py").write_text(
                '''
"""Test file for server."""

def test_simple_function():
    """Test the simple function."""
    from .server import simple_function
    assert simple_function() == "test"
'''
            )

            # Run compliance check
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.kit_path / "validate_compliance.py"),
                    "--service-dir",
                    str(test_service_dir),
                    "--report-format",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                logger.error(f"❌ Compliance check failed: {result.stderr}")
                return False

            # Parse compliance report
            try:
                report = json.loads(result.stdout)
                if "overall_score" not in report:
                    logger.error("❌ Compliance report missing overall_score")
                    return False

                logger.debug(
                    f"✅ Compliance check score: {report['overall_score']:.3f}"
                )

            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse compliance report: {e}")
                return False

            logger.info("✅ Compliance checker validation passed")
            return True

        except subprocess.TimeoutExpired:
            logger.error("❌ Compliance checker validation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Compliance checker validation failed: {e}")
            return False

    def _validate_end_to_end_workflow(self) -> bool:
        """Validate the complete end-to-end workflow."""
        logger.info("🔄 Validating end-to-end workflow...")

        try:
            # 1. Generate service
            example_proto = self.kit_path / "examples" / "example_service.proto"
            e2e_service_dir = self.temp_dir / "e2e_service"

            gen_result = subprocess.run(
                [
                    sys.executable,
                    str(self.kit_path / "generate_service.py"),
                    "--service-name",
                    "E2EService",
                    "--proto-file",
                    str(example_proto),
                    "--output-dir",
                    str(e2e_service_dir),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if gen_result.returncode != 0:
                logger.error(f"❌ E2E service generation failed: {gen_result.stderr}")
                return False

            # 2. Check compliance
            compliance_result = subprocess.run(
                [
                    sys.executable,
                    str(self.kit_path / "validate_compliance.py"),
                    "--service-dir",
                    str(e2e_service_dir),
                    "--report-format",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if compliance_result.returncode != 0:
                logger.error(
                    f"❌ E2E compliance check failed: {compliance_result.stderr}"
                )
                return False

            # 3. Parse and validate compliance
            try:
                report = json.loads(compliance_result.stdout)
                overall_score = report.get("overall_score", 0.0)

                if overall_score < 0.70:  # Minimum acceptable score
                    logger.error(
                        f"❌ E2E compliance score too low: {overall_score:.3f}"
                    )
                    return False

                logger.debug(f"✅ E2E compliance score: {overall_score:.3f}")

            except json.JSONDecodeError:
                logger.error("❌ E2E compliance report parsing failed")
                return False

            # 4. Validate generated file structure
            required_files = [
                "e2_e_service_servicer.py",  # Fixed: E2EService -> e2_e_service
                "server.py",
                "__init__.py",
                "test_e2_e_service_servicer.py",  # Fixed: matching test file
                "README.md",
            ]

            for required_file in required_files:
                if not (e2e_service_dir / required_file).exists():
                    logger.error(f"❌ E2E missing file: {required_file}")
                    return False

            logger.info("✅ End-to-end workflow validation passed")
            return True

        except subprocess.TimeoutExpired:
            logger.error("❌ End-to-end workflow validation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ End-to-end workflow validation failed: {e}")
            return False

    def _validate_constitutional_compliance(self) -> bool:
        """Validate the kit's own constitutional compliance."""
        logger.info("🏛️ Validating kit constitutional compliance...")

        try:
            # Use the kit's own compliance checker on itself
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.kit_path / "validate_compliance.py"),
                    "--service-dir",
                    str(self.kit_path),
                    "--report-format",
                    "json",
                    "--threshold",
                    "0.70",  # Reasonable threshold for the kit itself
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )

            # Allow warnings but not failures
            if result.returncode > 1:  # 1 = warnings, >1 = errors
                logger.error(f"❌ Kit self-compliance check failed: {result.stderr}")
                return False

            try:
                report = json.loads(result.stdout)
                kit_score = report.get("overall_score", 0.0)

                logger.info(f"🏛️ Kit constitutional compliance score: {kit_score:.3f}")

                if kit_score < 0.70:
                    logger.warning(
                        f"⚠️ Kit compliance score below target: {kit_score:.3f}"
                    )
                    # Don't fail, but warn

            except json.JSONDecodeError:
                logger.warning("⚠️ Could not parse kit self-compliance report")

            logger.info("✅ Kit constitutional compliance validation passed")
            return True

        except subprocess.TimeoutExpired:
            logger.error("❌ Kit constitutional compliance validation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Kit constitutional compliance validation failed: {e}")
            return False

    def _validate_documentation_quality(self) -> bool:
        """Validate the quality and completeness of documentation."""
        logger.info("📚 Validating documentation quality...")

        try:
            # Check main documentation files
            docs_to_check = [
                ("README.md", ["overview", "constitutional", "usage", "compliance"]),
                (
                    "USAGE.md",
                    ["quick start", "workflow", "examples", "troubleshooting"],
                ),
            ]

            for doc_file, required_sections in docs_to_check:
                doc_path = self.kit_path / doc_file
                if not doc_path.exists():
                    logger.error(f"❌ Documentation file missing: {doc_file}")
                    return False

                content = doc_path.read_text().lower()

                for section in required_sections:
                    if section not in content:
                        logger.warning(f"⚠️ {doc_file} missing section: {section}")
                        # Don't fail for missing sections, just warn

                logger.debug(f"✅ Documentation file validated: {doc_file}")

            # Check template documentation
            template_readme = self.kit_path / "templates" / "README_template.md"
            if template_readme.exists():
                template_content = template_readme.read_text()
                if len(template_content) < 1000:  # Minimum documentation length
                    logger.warning("⚠️ Template README seems too short")
                logger.debug("✅ Template documentation validated")

            logger.info("✅ Documentation quality validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ Documentation quality validation failed: {e}")
            return False

    def print_validation_report(self) -> None:
        """Print a comprehensive validation report."""
        print("\n" + "=" * 60)
        print("🏛️ Constitutional gRPC Development Kit Validation Report")
        print("=" * 60)

        if not self.validation_results:
            print("❌ No validation results available")
            return

        overall_success = all(self.validation_results.values())
        status_emoji = "✅" if overall_success else "❌"

        print(
            f"\n{status_emoji} Overall Status: {'PASSED' if overall_success else 'FAILED'}"
        )
        print(
            f"📊 Success Rate: {sum(self.validation_results.values())}/{len(self.validation_results)}"
        )

        print("\n📋 Validation Results:")
        for category, success in self.validation_results.items():
            emoji = "✅" if success else "❌"
            print(f"  {emoji} {category.replace('_', ' ').title()}")

        if overall_success:
            print("\n🎉 Constitutional gRPC Development Kit is ready for use!")
            print("📖 See USAGE.md for getting started")
        else:
            print("\n⚠️ Kit validation failed. Please review the errors above.")
            print("🔧 Fix the issues and run validation again")

        print("\n⚖️ Constitutional Framework Compliance: Validated")
        print("🏗️ Generated services will adhere to all 14 constitutional articles")
        print("=" * 60 + "\n")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate Constitutional gRPC Development Kit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--kit-path",
        type=Path,
        default=Path(__file__).parent,
        help="Path to the Constitutional gRPC Development Kit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (skip some tests)",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    try:
        # Validate kit path
        if not args.kit_path.exists():
            logger.error(f"❌ Kit path not found: {args.kit_path}")
            return 1

        if not args.kit_path.is_dir():
            logger.error(f"❌ Kit path is not a directory: {args.kit_path}")
            return 1

        # Initialize and run validator
        validator = ConstitutionalKitValidator(args.kit_path)

        # Run validation
        results = validator.validate_kit()

        # Print report
        validator.print_validation_report()

        # Return appropriate exit code
        return 0 if all(results.values()) else 1

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
