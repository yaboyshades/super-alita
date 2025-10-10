"""
Mangle integration for SDD framework.
Generates facts and rules from specifications and codebase.

This module provides functionality to extract structured facts from:
- Specification documents (markdown files)
- Source code (Python AST parsing)
- Constitutional framework documents

The generated facts can be used with Mangle for deductive reasoning
about code quality, test coverage, and constitutional compliance.
"""

import ast
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class MangleFactGenerator:
    """
    Generates Mangle facts from specifications and source code.

    This class parses various project artifacts to create a knowledge graph
    represented as Mangle facts, enabling deductive reasoning about:
    - Specification completeness
    - Test coverage
    - Constitutional compliance
    - Code structure and dependencies
    """

    def __init__(self, workspace_root: str = "."):
        """
        Initialize the fact generator.

        Args:
            workspace_root: Root directory of the workspace to analyze
        """
        self.workspace_root = Path(workspace_root)
        self.specs_dir = self.workspace_root / "specs"
        self.src_dir = self.workspace_root / "src"
        self.tests_dir = self.workspace_root / "tests"

        # Track generated facts to avoid duplicates
        self._generated_facts: set[str] = set()

    def generate_all_facts(self) -> str:
        """Generate all Mangle facts from specs, code, and constitution.

        Returns:
            String containing all generated Mangle facts
        """
        logger.info("Generating all Mangle facts from workspace")

        self._generated_facts.clear()

        # Generate facts from different sources
        spec_facts = self.generate_spec_facts()
        code_facts = self.generate_code_facts()
        test_facts = self.generate_test_facts()
        constitution_facts = self.generate_constitution_facts()
        dependency_facts = self.generate_dependency_facts()

        all_facts = [
            "// Specification facts",
            spec_facts,
            "",
            "// Code structure facts",
            code_facts,
            "",
            "// Test coverage facts",
            test_facts,
            "",
            "// Constitutional framework facts",
            constitution_facts,
            "",
            "// Dependency facts",
            dependency_facts,
        ]

        result = "\n".join(all_facts)
        logger.info(f"Generated {len(self._generated_facts)} unique facts")
        return result

    def generate_facts(self) -> list[str]:
        """Return unique fact strings for compatibility with MangleBridge."""
        all_facts = self.generate_all_facts()
        return [
            line
            for line in all_facts.splitlines()
            if line and not line.startswith("//")
        ]

    def generate_spec_facts(self) -> str:
        """
        Generate facts from specification documents.

        Extracts:
        - Feature specifications
        - Acceptance criteria
        - Requirements
        - Feature relationships

        Returns:
            String containing specification facts
        """
        facts = []

        if not self.specs_dir.exists():
            logger.warning(f"Specs directory {self.specs_dir} does not exist")
            return ""

        for spec_dir in self.specs_dir.iterdir():
            if spec_dir.is_dir():
                spec_file = spec_dir / "feature-spec.md"
                if spec_file.exists():
                    content = spec_file.read_text(encoding="utf-8")
                    feature_id = spec_dir.name

                    # Extract feature information
                    title = self._extract_spec_title(content)
                    facts.append(
                        self._add_fact(
                            f'spec("{feature_id}", "{self._escape_string(title)}")'
                        )
                    )

                    # Extract acceptance criteria
                    ac_facts = self._extract_acceptance_criteria(
                        content, feature_id
                    )
                    facts.extend(ac_facts)

                    # Extract requirements
                    req_facts = self._extract_requirements(content, feature_id)
                    facts.extend(req_facts)

                    # Extract feature tags/categories
                    tag_facts = self._extract_feature_tags(content, feature_id)
                    facts.extend(tag_facts)

        return "\n".join(facts)

    def generate_code_facts(self) -> str:
        """
        Generate facts from source code.

        Extracts:
        - Function definitions
        - Class definitions
        - Module structure
        - Function calls and dependencies

        Returns:
            String containing code structure facts
        """
        facts = []

        if not self.src_dir.exists():
            logger.warning(f"Source directory {self.src_dir} does not exist")
            return ""

        for file_path in self.src_dir.rglob("*.py"):
            relative_path = file_path.relative_to(self.workspace_root)
            facts.extend(
                self._parse_python_file(file_path, str(relative_path))
            )

        return "\n".join(facts)

    def generate_test_facts(self) -> str:
        """
        Generate facts from test files.

        Extracts:
        - Test function definitions
        - Test-to-function mappings
        - Test coverage information

        Returns:
            String containing test facts
        """
        facts = []

        if not self.tests_dir.exists():
            logger.warning(f"Tests directory {self.tests_dir} does not exist")
            return ""

        for file_path in self.tests_dir.rglob("test_*.py"):
            relative_path = file_path.relative_to(self.workspace_root)
            facts.extend(self._parse_test_file(file_path, str(relative_path)))

        return "\n".join(facts)

    def generate_constitution_facts(self) -> str:
        """
        Generate facts from constitutional framework.

        Extracts:
        - Constitutional articles
        - Mandates and principles
        - Compliance rules

        Returns:
            String containing constitutional facts
        """
        facts = []

        # Look for constitution in multiple possible locations
        constitution_paths = [
            self.workspace_root / ".github" / "CONSTITUTION.md",
            self.workspace_root / "base" / "memory" / "constitution.md",
            self.workspace_root / "CONSTITUTION.md",
        ]

        for constitution_path in constitution_paths:
            if constitution_path.exists():
                content = constitution_path.read_text(encoding="utf-8")
                facts.extend(self._parse_constitution(content))
                break
        else:
            logger.warning("No constitution file found")

        return "\n".join(facts)

    def generate_dependency_facts(self) -> str:
        """
        Generate facts about project dependencies.

        Extracts:
        - Import relationships
        - Module dependencies
        - External package usage

        Returns:
            String containing dependency facts
        """
        facts = []

        # Parse requirements.txt if it exists
        req_file = self.workspace_root / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text(encoding="utf-8")
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    package = (
                        line.split("==")[0]
                        .split(">=")[0]
                        .split("<=")[0]
                        .strip()
                    )
                    facts.append(
                        self._add_fact(f'external_dependency("{package}")')
                    )

        return "\n".join(facts)

    def _parse_python_file(
        self, file_path: Path, relative_path: str
    ) -> list[str]:
        """
        Parse a Python file and extract structural facts.

        Args:
            file_path: Absolute path to the Python file
            relative_path: Relative path from workspace root

        Returns:
            List of Mangle facts for this file
        """
        facts = []

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            # Add module fact
            module_name = (
                relative_path.replace("/", ".")
                .replace("\\", ".")
                .replace(".py", "")
            )
            facts.append(
                self._add_fact(f'module("{module_name}", "{relative_path}")')
            )

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    facts.append(
                        self._add_fact(
                            f'function("{func_name}", "{relative_path}", "{module_name}")'
                        )
                    )

                    # Check for decorators that might indicate CLI commands
                    for decorator in node.decorator_list:
                        if (
                            isinstance(decorator, ast.Name)
                            and decorator.id == "click.command"
                        ):
                            facts.append(
                                self._add_fact(
                                    f'cli_command("{func_name}", "{relative_path}")'
                                )
                            )

                    # Extract docstring if present
                    if (
                        isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ):
                        docstring = node.body[0].value.value.strip()
                        facts.append(
                            self._add_fact(
                                f'function_doc("{func_name}", "{self._escape_string(docstring)}")'
                            )
                        )

                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    facts.append(
                        self._add_fact(
                            f'class("{class_name}", "{relative_path}", "{module_name}")'
                        )
                    )

                    # Extract class methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_name = item.name
                            facts.append(
                                self._add_fact(
                                    f'method("{method_name}", "{class_name}", "{relative_path}")'
                                )
                            )

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_module = alias.name
                        facts.append(
                            self._add_fact(
                                f'imports("{module_name}", "{imported_module}")'
                            )
                        )

                elif isinstance(node, ast.ImportFrom) and node.module:
                    from_module = node.module
                    for alias in node.names:
                        imported_name = alias.name
                        facts.append(
                            self._add_fact(
                                f'imports_from("{module_name}", "{from_module}", "{imported_name}")'
                            )
                        )

        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return facts

    def _parse_test_file(
        self, file_path: Path, relative_path: str
    ) -> list[str]:
        """
        Parse a test file and extract test-related facts.

        Args:
            file_path: Absolute path to the test file
            relative_path: Relative path from workspace root

        Returns:
            List of Mangle facts for this test file
        """
        facts = []

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            module_name = (
                relative_path.replace("/", ".")
                .replace("\\", ".")
                .replace(".py", "")
            )
            facts.append(
                self._add_fact(
                    f'test_module("{module_name}", "{relative_path}")'
                )
            )

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith(
                    "test_"
                ):
                    test_name = node.name
                    facts.append(
                        self._add_fact(
                            f'test_function("{test_name}", "{relative_path}")'
                        )
                    )

                    # Try to infer what function this test covers
                    # Simple heuristic: remove "test_" prefix
                    if test_name.startswith("test_"):
                        tested_function = test_name[
                            5:
                        ]  # Remove "test_" prefix
                        facts.append(
                            self._add_fact(
                                f'test_for("{test_name}", "{tested_function}")'
                            )
                        )

        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to parse test file {file_path}: {e}")

        return facts

    def _parse_constitution(self, content: str) -> list[str]:
        """
        Parse constitutional framework document.

        Args:
            content: Content of the constitution document

        Returns:
            List of constitutional facts
        """
        facts = []

        # Extract articles using various patterns
        patterns = [
            r"## Article (\w+): ([^\n]+)\n\n\*\*Mandate\*\*: ([^\n]+)",
            r"### Article (\w+): ([^\n]+)\n\n([^\n]+)",
            r"## (\w+)\. ([^\n]+)\n\n([^\n]+)",
        ]

        for pattern in patterns:
            articles = re.findall(pattern, content, re.MULTILINE)
            for article in articles:
                if len(article) >= 3:
                    article_id, article_name, mandate = (
                        article[0],
                        article[1],
                        article[2],
                    )
                    facts.append(
                        self._add_fact(
                            f'constitution_article("{article_id}", "{self._escape_string(article_name)}", "{self._escape_string(mandate.strip())}")'
                        )
                    )

        return facts

    def _extract_spec_title(self, content: str) -> str:
        """Extract specification title from content."""
        patterns = [
            r"# Feature Specification: (.*)",
            r"# (.*?) Specification",
            r"# (.*)",
            r"## (.*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()

        return "Untitled Specification"

    def _extract_acceptance_criteria(
        self, content: str, feature_id: str
    ) -> list[str]:
        """Extract acceptance criteria from specification content."""
        facts = []

        # Look for acceptance criteria section
        ac_pattern = r"## Acceptance Criteria\n(.*?)(?=##|$)"
        ac_match = re.search(ac_pattern, content, re.DOTALL)

        if ac_match:
            ac_content = ac_match.group(1)
            # Extract checklist items
            ac_items = re.findall(r"- \[ \] (.*)", ac_content)
            for i, ac in enumerate(ac_items):
                ac_id = f"AC-{i+1:02d}"
                facts.append(
                    self._add_fact(
                        f'acceptance_criterion("{ac_id}", "{feature_id}", "{self._escape_string(ac.strip())}")'
                    )
                )

        return facts

    def _extract_requirements(
        self, content: str, feature_id: str
    ) -> list[str]:
        """Extract requirements from specification content."""
        facts = []

        # Look for requirements section
        req_pattern = r"## Requirements\n(.*?)(?=##|$)"
        req_match = re.search(req_pattern, content, re.DOTALL)

        if req_match:
            req_content = req_match.group(1)
            req_items = re.findall(r"- (.*)", req_content)
            for i, req in enumerate(req_items):
                req_id = f"REQ-{i+1:02d}"
                facts.append(
                    self._add_fact(
                        f'requirement("{req_id}", "{feature_id}", "{self._escape_string(req.strip())}")'
                    )
                )

        return facts

    def _extract_feature_tags(
        self, content: str, feature_id: str
    ) -> list[str]:
        """Extract feature tags/categories from specification content."""
        facts = []

        # Look for tags in YAML frontmatter or explicit tags section
        tag_patterns = [
            r"tags:\s*\[(.*?)\]",
            r"categories:\s*\[(.*?)\]",
            r"## Tags\n(.*?)(?=##|$)",
        ]

        for pattern in tag_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                if "," in match:
                    tags = [
                        tag.strip().strip("\"'") for tag in match.split(",")
                    ]
                else:
                    tags = [
                        tag.strip() for tag in match.split("\n") if tag.strip()
                    ]

                for tag in tags:
                    if tag:
                        facts.append(
                            self._add_fact(
                                f'feature_tag("{feature_id}", "{tag}")'
                            )
                        )

        return facts

    def _add_fact(self, fact: str) -> str:
        """
        Add a fact to the generated facts set to avoid duplicates.

        Args:
            fact: The fact string to add

        Returns:
            The fact string if it is new, empty string if duplicate
        """
        if fact not in self._generated_facts:
            self._generated_facts.add(fact)
            return fact
        return ""

    def _escape_string(self, s: str) -> str:
        """
        Escape special characters in strings for Mangle facts.

        Args:
            s: String to escape

        Returns:
            Escaped string safe for use in Mangle facts
        """
        # Escape quotes and other special characters
        return s.replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")

    def get_fact_statistics(self) -> dict[str, int]:
        """
        Get statistics about generated facts.

        Returns:
            Dictionary with fact type counts
        """
        stats = {}
        for fact in self._generated_facts:
            fact_type = fact.split("(")[0] if "(" in fact else "unknown"
            stats[fact_type] = stats.get(fact_type, 0) + 1

        return stats
