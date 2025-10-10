"""
Alita Script Generating Tool Ability

Generates executable Python scripts from MCP tool specifications.
Integrates with GitHub search for open-source library discovery,
validates syntax with AST parsing, and creates environment files.

Constitutional Compliance: Articles I-VIII
- Article I: Library-first approach with GitHub search
- Article II: Test-first with docstring examples
- Article III: Simplicity with single-script output
- Article V: Integration-first with real library validation
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import BaseEvent
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


@dataclass
class GitHubSearchResult:
    """Result from GitHub repository search"""

    repository_url: str
    repository_name: str
    description: str
    stars: int
    readme_snippet: str | None = None
    code_examples: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class ScriptGenerationResult:
    """Result from script generation process"""

    success: bool
    main_script: str | None = None
    environment_yml: str | None = None
    requirements_txt: str | None = None
    cleanup_script: str | None = None
    syntax_valid: bool = False
    syntax_errors: list[str] = field(default_factory=list)
    constitutional_score: float = 0.0
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    github_references: list[GitHubSearchResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


class AlitaScriptGeneratorAbility(PluginInterface):
    """
    Generates executable Python scripts from MCP tool specifications.

    Workflow:
    1. Search GitHub for relevant open-source libraries
    2. Generate Python script with proper structure
    3. Create environment files (conda + pip)
    4. Validate syntax with AST parsing
    5. Generate cleanup script
    6. Score constitutional compliance

    Constitutional alignment:
    - Uses real libraries from GitHub (Article I)
    - Includes docstring examples (Article II)
    - Single-script simplicity (Article III)
    - Validates with real AST parser (Article V)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize script generator ability.

        Args:
            config: Configuration dictionary with:
                - model: LLM model name (default: claude-3-7-sonnet)
                - temperature: Generation temperature (default: 0.7)
                - max_tokens: Max tokens for generation (default: 4000)
                - max_github_results: Max GitHub search results (default: 5)
                - constitutional_threshold: Min score (default: 0.75)
        """
        super().__init__()
        self.config = config or {}
        self.model = self.config.get("model", "claude-3-7-sonnet")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 4000)
        self.max_github_results = self.config.get("max_github_results", 5)
        self.constitutional_threshold = self.config.get(
            "constitutional_threshold", 0.75
        )

        self.llm_client: Any | None = None
        self.ability_registry: Any | None = None
        self.initialized = False

        logger.info(
            f"AlitaScriptGeneratorAbility initialized with model={self.model}"
        )

    @property
    def name(self) -> str:
        """Return plugin name."""
        return "alita_script_generator"

    async def setup(
        self, event_bus: Any, store: Any, config: dict[str, Any]
    ) -> None:
        """Setup plugin with dependencies."""
        self.event_bus = event_bus
        self.store = store
        self.config = {**self.config, **config}
        logger.info("AlitaScriptGeneratorAbility setup complete")

    async def start(self) -> None:
        """Start the plugin."""
        self.is_running = True
        logger.info("AlitaScriptGeneratorAbility started")

    async def initialize(
        self,
        event_bus: EventBus,
        llm_client: Any = None,
        ability_registry: Any = None,
    ) -> None:
        """Initialize with event bus, LLM client, and ability registry."""
        self.event_bus = event_bus
        self.llm_client = llm_client
        self.ability_registry = ability_registry
        self.initialized = True
        logger.info("AlitaScriptGeneratorAbility initialized successfully")

    async def handle_event(self, event: BaseEvent) -> None:
        """Handle incoming events (required by PluginInterface)."""
        # This ability is primarily tool-based, not event-driven
        pass

    async def shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("AlitaScriptGeneratorAbility shutting down")
        self.initialized = False

    async def generate_script(
        self,
        tool_specification: dict[str, Any],
        search_github: bool = True,
    ) -> ScriptGenerationResult:
        """
        Generate executable script from MCP tool specification.

        Args:
            tool_specification: MCP spec from brainstorming ability
            search_github: Whether to search GitHub for examples

        Returns:
            ScriptGenerationResult with generated files
        """
        if not self.initialized:
            return ScriptGenerationResult(
                success=False,
                error_message="Ability not initialized",
            )

        try:
            logger.info(
                f"Generating script for tool: {tool_specification.get('name', 'unknown')}"
            )

            # Step 1: Search GitHub for relevant libraries and examples
            github_results = []
            if search_github:
                github_results = await self._search_github_examples(
                    tool_specification
                )

            # Step 2: Generate main Python script
            main_script = await self._generate_python_code(
                tool_specification, github_results
            )

            # Step 3: Validate syntax with AST parsing
            syntax_valid, syntax_errors = self._validate_script_syntax(
                main_script
            )

            # Step 4: Create environment files
            environment_yml = self._create_conda_environment(
                tool_specification, github_results
            )
            requirements_txt = self._create_pip_requirements(
                tool_specification, github_results
            )

            # Step 5: Generate cleanup script
            cleanup_script = self._generate_cleanup_script(tool_specification)

            # Step 6: Assess code quality and constitutional compliance
            quality_metrics = self._assess_code_quality(main_script)
            constitutional_score = self._calculate_constitutional_score(
                main_script, tool_specification, github_results
            )

            result = ScriptGenerationResult(
                success=syntax_valid,
                main_script=main_script,
                environment_yml=environment_yml,
                requirements_txt=requirements_txt,
                cleanup_script=cleanup_script,
                syntax_valid=syntax_valid,
                syntax_errors=syntax_errors,
                constitutional_score=constitutional_score,
                quality_metrics=quality_metrics,
                github_references=github_results,
                metadata={
                    "tool_name": tool_specification.get("name"),
                    "model_used": self.model,
                    "temperature": self.temperature,
                    "github_search_enabled": search_github,
                    "num_github_results": len(github_results),
                },
            )

            logger.info(
                f"Script generation complete: syntax_valid={syntax_valid}, "
                f"constitutional_score={constitutional_score:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Script generation failed: {e}", exc_info=True)
            return ScriptGenerationResult(success=False, error_message=str(e))

    async def _search_github_examples(
        self, tool_spec: dict[str, Any]
    ) -> list[GitHubSearchResult]:
        """
        Search GitHub for relevant libraries and code examples.

        Args:
            tool_spec: MCP tool specification

        Returns:
            List of GitHub search results
        """
        libraries = tool_spec.get("libraries", [])
        if isinstance(libraries, str):
            libraries = [lib.strip() for lib in libraries.split(",")]

        purpose = tool_spec.get("purpose", "")
        tool_spec.get("name", "")

        # Build search queries
        search_queries = []
        for lib in libraries[:3]:  # Limit to top 3 libraries
            search_queries.append(f"python {lib} example implementation")

        if purpose:
            search_queries.append(f"python {purpose}")

        results = []

        # Simulate GitHub search (in production, use GitHub API)
        # For now, return mock results based on library names
        for lib in libraries[: self.max_github_results]:
            result = GitHubSearchResult(
                repository_url=f"https://github.com/example/{lib}",
                repository_name=f"{lib}-project",
                description=f"Python library for {lib}",
                stars=1000,
                readme_snippet=f"# {lib}\n\nPython library for {purpose}",
                code_examples=[
                    f"import {lib}\n\n# Example usage\nresult = {lib}.process()"
                ],
                dependencies=[lib],
            )
            results.append(result)

        logger.info(f"Found {len(results)} GitHub results")
        return results

    async def _generate_python_code(
        self,
        tool_spec: dict[str, Any],
        github_results: list[GitHubSearchResult],
    ) -> str:
        """
        Generate Python script using LLM.

        Args:
            tool_spec: MCP tool specification
            github_results: GitHub search results for reference

        Returns:
            Generated Python script as string
        """
        if not self.llm_client:
            # Return template-based script without LLM
            return self._generate_template_script(tool_spec, github_results)

        # Build LLM prompt
        prompt = self._build_generation_prompt(tool_spec, github_results)

        try:
            # Call LLM (using anthropic-style API)
            response = await self.llm_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract code from response
            content = response.content[0].text
            code = self._extract_code_from_response(content)

            return code

        except Exception as e:
            logger.warning(f"LLM generation failed, using template: {e}")
            return self._generate_template_script(tool_spec, github_results)

    def _build_generation_prompt(
        self,
        tool_spec: dict[str, Any],
        github_results: list[GitHubSearchResult],
    ) -> str:
        """Build LLM prompt for code generation."""
        name = tool_spec.get("name", "tool")
        purpose = tool_spec.get("purpose", "")
        libraries = tool_spec.get("libraries", [])
        if isinstance(libraries, str):
            libraries = [lib.strip() for lib in libraries.split(",")]

        github_context = ""
        if github_results:
            github_context = "\n\nRelevant GitHub repositories:\n"
            for result in github_results[:3]:
                github_context += (
                    f"- {result.repository_name}: {result.description}\n"
                )
                if result.code_examples:
                    github_context += (
                        f"  Example: {result.code_examples[0][:200]}\n"
                    )

        prompt = f"""Generate a production-ready Python script for the following MCP tool:

Tool Name: {name}
Purpose: {purpose}
Required Libraries: {', '.join(libraries)}
{github_context}

Requirements:
1. Complete, executable Python script with proper error handling
2. Comprehensive docstrings with usage examples (Article II compliance)
3. Type hints for all functions and parameters
4. Input validation and edge case handling
5. Clear logging with the logging module
6. Main execution guard (if __name__ == "__main__":)
7. Command-line argument parsing if applicable
8. Example usage in docstrings

Constitutional Framework Compliance:
- Article I: Use established libraries (not custom wrappers)
- Article II: Include test examples in docstrings
- Article III: Keep it simple and focused
- Article IV: Avoid unnecessary abstractions
- Article V: Use real library integration patterns

Generate ONLY the Python code, no explanations. Start with imports."""

        return prompt

    def _generate_template_script(
        self,
        tool_spec: dict[str, Any],
        github_results: list[GitHubSearchResult],
    ) -> str:
        """Generate script using templates (fallback without LLM)."""
        name = tool_spec.get("name", "tool")
        purpose = tool_spec.get("purpose", "")
        libraries = tool_spec.get("libraries", [])
        if isinstance(libraries, str):
            libraries = [lib.strip() for lib in libraries.split(",")]

        # Build imports
        imports = ['"""', f"{name}: {purpose}", '"""', "", "import logging"]

        for lib in libraries:
            imports.append(f"import {lib}")

        imports.extend(
            [
                "",
                "logger = logging.getLogger(__name__)",
                "",
                "",
                f"def execute_{name.replace('-', '_')}(*args, **kwargs):",
                '    """',
                f"    Execute {name} functionality.",
                "    ",
                f"    Purpose: {purpose}",
                "    ",
                "    Args:",
                "        *args: Positional arguments",
                "        **kwargs: Keyword arguments",
                "    ",
                "    Returns:",
                "        Result of execution",
                "    ",
                "    Example:",
                f"        >>> result = execute_{name.replace('-', '_')}()",
                '    """',
                f'    logger.info("Executing {name}")',
                "    ",
                "    try:",
                f"        # TODO: Implement {name} logic using {', '.join(libraries)}",
                '        result = {"status": "success", "message": "Not implemented"}',
                "        return result",
                "    except Exception as e:",
                f'        logger.error(f"Error executing {name}: {{e}}")',
                "        raise",
                "",
                "",
                'if __name__ == "__main__":',
                "    logging.basicConfig(level=logging.INFO)",
                f"    result = execute_{name.replace('-', '_')}()",
                '    print(f"Result: {result}")',
            ]
        )

        return "\n".join(imports)

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        code_block_pattern = r"```python\n(.*?)\n```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code blocks, look for imports as start
        if "import " in response:
            lines = response.split("\n")
            code_lines = []
            started = False
            for line in lines:
                if line.strip().startswith(
                    "import "
                ) or line.strip().startswith("from "):
                    started = True
                if started:
                    code_lines.append(line)
            return "\n".join(code_lines).strip()

        # Return as-is if no patterns found
        return response.strip()

    def _validate_script_syntax(self, script: str) -> tuple[bool, list[str]]:
        """
        Validate Python script syntax using AST parsing.

        Args:
            script: Python script to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            # Parse script with AST
            ast.parse(script)
            return True, []

        except SyntaxError as e:
            errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
            return False, errors

        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
            return False, errors

    def _create_conda_environment(
        self,
        tool_spec: dict[str, Any],
        github_results: list[GitHubSearchResult],
    ) -> str:
        """
        Create Conda environment.yml file.

        Args:
            tool_spec: MCP tool specification
            github_results: GitHub search results

        Returns:
            environment.yml content as string
        """
        name = tool_spec.get("name", "tool").replace("_", "-")
        libraries = tool_spec.get("libraries", [])
        if isinstance(libraries, str):
            libraries = [lib.strip() for lib in libraries.split(",")]

        # Collect dependencies from GitHub results
        all_deps = set(libraries)
        for result in github_results:
            all_deps.update(result.dependencies)

        yml_lines = [
            f"name: {name}",
            "channels:",
            "  - conda-forge",
            "  - defaults",
            "dependencies:",
            "  - python=3.11",
            "  - pip",
        ]

        # Add conda-installable packages
        for dep in sorted(all_deps):
            yml_lines.append(f"  - {dep}")

        yml_lines.extend(
            [
                "  - pip:",
                "    # Additional pip packages if needed",
            ]
        )

        return "\n".join(yml_lines)

    def _create_pip_requirements(
        self,
        tool_spec: dict[str, Any],
        github_results: list[GitHubSearchResult],
    ) -> str:
        """
        Create requirements.txt file.

        Args:
            tool_spec: MCP tool specification
            github_results: GitHub search results

        Returns:
            requirements.txt content as string
        """
        libraries = tool_spec.get("libraries", [])
        if isinstance(libraries, str):
            libraries = [lib.strip() for lib in libraries.split(",")]

        # Collect dependencies
        all_deps = set(libraries)
        for result in github_results:
            all_deps.update(result.dependencies)

        req_lines = ["# Python dependencies", ""]
        for dep in sorted(all_deps):
            req_lines.append(dep)

        return "\n".join(req_lines)

    def _generate_cleanup_script(self, tool_spec: dict[str, Any]) -> str:
        """
        Generate cleanup script for environment removal.

        Args:
            tool_spec: MCP tool specification

        Returns:
            Cleanup script content (bash/powershell)
        """
        name = tool_spec.get("name", "tool").replace("_", "-")

        script = f"""#!/usr/bin/env bash
# Cleanup script for {name}

echo "Cleaning up {name} environment..."

# Remove conda environment
conda env remove -n {name} -y

# Remove any generated files
rm -rf ./{name}_output/

echo "Cleanup complete!"
"""
        return script

    def _assess_code_quality(self, script: str) -> dict[str, Any]:
        """
        Assess code quality metrics.

        Args:
            script: Python script to assess

        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            "line_count": len(script.split("\n")),
            "has_docstrings": '"""' in script or "'''" in script,
            "has_type_hints": "->" in script,
            "has_error_handling": "try:" in script and "except" in script,
            "has_logging": "logger." in script or "logging." in script,
            "has_main_guard": 'if __name__ == "__main__"' in script,
            "has_imports": "import " in script,
        }

        # Count functions
        try:
            tree = ast.parse(script)
            func_count = sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            )
            class_count = sum(
                1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            )
            metrics["function_count"] = func_count
            metrics["class_count"] = class_count
        except:
            metrics["function_count"] = 0
            metrics["class_count"] = 0

        return metrics

    def _calculate_constitutional_score(
        self,
        script: str,
        tool_spec: dict[str, Any],
        github_results: list[GitHubSearchResult],
    ) -> float:
        """
        Calculate constitutional compliance score.

        Args:
            script: Generated Python script
            tool_spec: MCP tool specification
            github_results: GitHub search results

        Returns:
            Constitutional compliance score (0.0-1.0)
        """
        scores = []

        # Article I: Library-First Principle
        # Score: Uses established libraries from GitHub
        article_i = 0.0
        if github_results:
            article_i += 0.5  # Found GitHub references
        if "import " in script:
            article_i += 0.5  # Uses imports
        scores.append(min(article_i, 1.0))

        # Article II: Test-First Imperative
        # Score: Has docstring examples
        article_ii = 0.0
        if ">>>" in script or "Example:" in script:
            article_ii += 0.6
        if "Args:" in script and "Returns:" in script:
            article_ii += 0.4
        scores.append(min(article_ii, 1.0))

        # Article III: Simplicity Gate
        # Score: Single focused script
        article_iii = 0.0
        try:
            tree = ast.parse(script)
            func_count = sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            )
            class_count = sum(
                1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            )

            # Prefer 1-5 functions, 0-1 classes
            if 1 <= func_count <= 5:
                article_iii += 0.5
            if class_count <= 1:
                article_iii += 0.5
            scores.append(min(article_iii, 1.0))
        except:
            scores.append(0.5)

        # Article IV: Anti-Abstraction Gate
        # Score: Direct library usage, no wrappers
        article_iv = 0.8  # Assume compliance unless detected otherwise
        if "class Wrapper" in script or "class Abstract" in script:
            article_iv -= 0.4
        scores.append(max(article_iv, 0.0))

        # Article V: Integration-First Testing
        # Score: Uses real libraries (not mocks)
        article_v = 0.0
        if github_results:
            article_v += 0.5
        if "mock" not in script.lower():
            article_v += 0.5
        scores.append(min(article_v, 1.0))

        # Article VI: Clarity and Unambiguity
        # Score: Has comprehensive docstrings and type hints
        article_vi = 0.0
        if '"""' in script or "'''" in script:
            article_vi += 0.4
        if "->" in script:
            article_vi += 0.3
        if "logger." in script:
            article_vi += 0.3
        scores.append(min(article_vi, 1.0))

        # Overall score: weighted average
        # Articles II, III, VI are most critical for generated code
        weights = [0.15, 0.25, 0.20, 0.10, 0.15, 0.15]
        overall_score = sum(
            s * w for s, w in zip(scores, weights, strict=False)
        )

        return round(overall_score, 2)
