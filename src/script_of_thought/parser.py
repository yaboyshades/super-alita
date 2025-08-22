"""
Script of Thought Parser for Super Alita

Parses Perplexity-style "script" format into structured execution plans
that can be executed by the REUG v9.0 state machine.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Types of execution steps in a Script of Thought"""

    THINK = auto()  # Internal reasoning/analysis
    SEARCH = auto()  # Information gathering
    ANALYZE = auto()  # Data analysis
    COMPUTE = auto()  # Code execution/computation
    GENERATE = auto()  # Content generation
    VALIDATE = auto()  # Validation/verification
    TOOL = auto()  # Tool invocation
    DECISION = auto()  # Decision point
    OUTPUT = auto()  # Final output/response


@dataclass
class ScriptStep:
    """A single step in a Script of Thought execution plan"""

    step_type: StepType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dependencies: list[int] = field(default_factory=list)
    step_id: int | None = None


@dataclass
class ScriptOfThought:
    """Complete Script of Thought execution plan"""

    title: str
    description: str
    steps: list[ScriptStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ScriptOfThoughtParser:
    """
    Parser for Script of Thought format

    Supports formats like:
    # Script: Analyze Data
    ## Think: Understanding the problem
    - Need to analyze user data...

    ## Search: Find relevant information
    - Query: "data analysis best practices"

    ## Code: Process the data
    ```python
    import pandas as pd
    df = pd.read_csv('data.csv')
    ```

    ## Output: Present results
    Based on the analysis...
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Regex patterns for parsing
        self.title_pattern = re.compile(
            r"^#\s*Script:\s*(.+)$", re.MULTILINE | re.IGNORECASE
        )
        self.step_pattern = re.compile(
            r"^##\s*(\w+):\s*(.+)$", re.MULTILINE | re.IGNORECASE
        )
        self.code_block_pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
        self.dependency_pattern = re.compile(r"@depends\((\d+(?:,\s*\d+)*)\)")

    def parse(self, script_text: str) -> ScriptOfThought:
        """Parse script text into structured execution plan"""
        try:
            # Extract title
            title_match = self.title_pattern.search(script_text)
            title = title_match.group(1).strip() if title_match else "Untitled Script"

            # Split into sections
            sections = self._split_into_sections(script_text)

            # Parse each section into steps
            steps: list[ScriptStep] = []
            for i, (step_type_str, content) in enumerate(sections):
                step = self._parse_step(i, step_type_str, content)
                if step:
                    steps.append(step)

            # Create script
            script = ScriptOfThought(
                title=title,
                description=f"Script with {len(steps)} execution steps",
                steps=steps,
                metadata={
                    "parsed_at": "now",
                    "total_steps": len(steps),
                    "step_types": [step.step_type.name for step in steps],
                },
            )

            self.logger.info(f"Parsed script '{title}' with {len(steps)} steps")
            return script

        except Exception as e:
            self.logger.error(f"Failed to parse script: {e}")
            raise ValueError(f"Script parsing failed: {e}") from e

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """Split script text into (step_type, content) sections"""
        sections: list[tuple[str, str]] = []
        current_section = None
        current_content = []

        lines = text.split("\n")

        for line in lines:
            step_match = self.step_pattern.match(line)

            if step_match:
                # Save previous section
                if current_section:
                    sections.append((current_section, "\n".join(current_content)))

                # Start new section
                current_section = step_match.group(1).upper()
                current_content = [step_match.group(2)]
            elif current_section:
                current_content.append(line)

        # Save final section
        if current_section:
            sections.append((current_section, "\n".join(current_content)))

        return sections

    def _parse_step(
        self, step_id: int, step_type_str: str, content: str
    ) -> ScriptStep | None:
        """Parse individual step"""
        try:
            # Map string to enum
            step_type_mapping: dict[str, StepType] = {
                "THINK": StepType.THINK,
                "SEARCH": StepType.SEARCH,
                "ANALYZE": StepType.ANALYZE,
                "COMPUTE": StepType.COMPUTE,
                "GENERATE": StepType.GENERATE,
                "VALIDATE": StepType.VALIDATE,
                "TOOL": StepType.TOOL,
                "DECISION": StepType.DECISION,
                "OUTPUT": StepType.OUTPUT,
                "CODE": StepType.COMPUTE,  # Alias
                "EXECUTE": StepType.COMPUTE,  # Alias
                "QUERY": StepType.SEARCH,  # Alias
            }

            step_type = step_type_mapping.get(step_type_str, StepType.THINK)

            # Extract metadata
            metadata: dict[str, Any] = {}
            dependencies: list[int] = []

            # Check for dependencies
            dep_match = self.dependency_pattern.search(content)
            if dep_match:
                dependencies = [int(x.strip()) for x in dep_match.group(1).split(",")]
                content = self.dependency_pattern.sub("", content)

            # Extract code blocks for COMPUTE steps
            if step_type == StepType.COMPUTE:
                code_matches = self.code_block_pattern.findall(content)
                if code_matches:
                    language, code = code_matches[0]
                    metadata["language"] = language or "python"
                    metadata["code"] = code.strip()

            # Extract search queries for SEARCH steps
            if step_type == StepType.SEARCH:
                query_pattern = re.compile(
                    r'Query:\s*["\'](.+?)["\']|Query:\s*(.+)', re.IGNORECASE
                )
                query_match = query_pattern.search(content)
                if query_match:
                    metadata["query"] = query_match.group(1) or query_match.group(2)

            return ScriptStep(
                step_type=step_type,
                content=content.strip(),
                metadata=metadata,
                dependencies=dependencies,
                step_id=step_id,
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse step {step_id}: {e}")
            return None

    def validate_script(self, script: ScriptOfThought) -> list[str]:
        """Validate script for execution readiness"""
        issues: list[str] = []

        if not script.steps:
            issues.append("Script has no executable steps")

        # Check dependency validity
        step_ids = {step.step_id for step in script.steps if step.step_id is not None}

        for step in script.steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    issues.append(
                        f"Step {step.step_id} depends on non-existent step {dep_id}"
                    )

        # Check for circular dependencies
        if self._has_circular_dependencies(script.steps):
            issues.append("Script contains circular dependencies")

        return issues

    def _has_circular_dependencies(self, steps: list[ScriptStep]) -> bool:
        """Check for circular dependencies using DFS"""
        # Build dependency graph
        graph = {}
        for step in steps:
            if step.step_id is not None:
                graph[step.step_id] = step.dependencies

        # DFS to detect cycles
        visited: set[int] = set()
        rec_stack: set[int] = set()

        def has_cycle(node: int) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True

        return False

    def parse_script(self, script_text: str) -> ScriptOfThought | None:
        """
        Parse a script text into a ScriptOfThought object
        Alias for parse() method to match interpreter expectations
        """
        return self.parse(script_text)


def example_usage():
    """Example of parsing a Script of Thought"""
    script_text = """
    # Script: Analyze User Data

    ## Think: Understanding the requirements
    The user wants to analyze their data for trends and insights.
    Need to understand the data structure first.

    ## Search: Find data analysis techniques
    Query: "pandas data analysis best practices"

    ## Code: Load and examine data
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load the data
    df = pd.read_csv('user_data.csv')
    print(df.head())
    print(df.info())
    ```

    ## Code: Perform analysis @depends(2)
    ```python
    # Calculate basic statistics
    summary = df.describe()

    # Plot trends
    df.plot(x='date', y='value', kind='line')
    plt.title('Data Trends Over Time')
    plt.show()
    ```

    ## Output: Present findings
    Based on the analysis, here are the key insights:
    - Trend shows steady growth
    - Seasonal patterns evident in Q3/Q4
    - Recommend continued monitoring
    """

    parser = ScriptOfThoughtParser()
    script = parser.parse(script_text)

    print(f"Parsed script: {script.title}")
    print(f"Steps: {len(script.steps)}")

    for step in script.steps:
        print(f"- {step.step_type.name}: {step.content[:50]}...")

    # Validate
    issues = parser.validate_script(script)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Script validation passed!")


if __name__ == "__main__":
    example_usage()
