"""
Script of Thought Interpreter for Super Alita

Executes parsed Script of Thought plans using the computational environment.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .parser import ScriptOfThoughtParser, ScriptStep, StepType

if TYPE_CHECKING:
    from ..computational_env.executor import ComputationalEnvironment

logger = logging.getLogger(__name__)


class StepExecutionResult:
    """Result of executing a single step"""

    def __init__(
        self,
        step_id: int,
        success: bool,
        data: Any = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.step_id = step_id
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step_id": self.step_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


class ScriptOfThoughtInterpreter:
    """
    Interprets and executes Script of Thought plans

    Takes parsed script steps and executes them using available tools
    and computational resources.
    """

    def __init__(self, computational_env: "ComputationalEnvironment"):
        self.parser = ScriptOfThoughtParser()
        self.computational_env = computational_env
        self.execution_results: dict[int, StepExecutionResult] = {}
        self.logger = logging.getLogger(__name__)

    async def execute_script(
        self, script_text: str, session_id: str | None = None
    ) -> dict[str, Any]:
        """
        Execute a complete Script of Thought

        Args:
            script_text: Raw script text to parse and execute
            session_id: Session ID for computational environment context

        Returns:
            dict: Execution results with step details
        """
        try:
            # Parse the script
            script = self.parser.parse_script(script_text)
            if not script:
                return {
                    "success": False,
                    "error": "Failed to parse script",
                    "step_results": [],
                }

            # Execute each step
            self.execution_results.clear()
            step_results = []

            for step in script.steps:
                if step.step_id is not None:
                    result = await self._execute_step(step, session_id)
                    self.execution_results[step.step_id] = result
                    step_results.append(result.to_dict())

                    # Stop on failure if step is critical
                    if not result.success and step.metadata.get("critical", False):
                        break

            # Generate summary
            summary: dict[str, Any] = {
                "success": all(r.success for r in self.execution_results.values()),
                "total_steps": len(script.steps),
                "completed_steps": len(
                    [r for r in self.execution_results.values() if r.success]
                ),
                "failed_steps": len(
                    [r for r in self.execution_results.values() if not r.success]
                ),
                "step_results": step_results,
                "execution_context": {
                    "session_id": session_id,
                    "script_hash": hash(script_text),
                },
            }

            return summary

        except Exception as e:
            self.logger.error(f"Script execution failed: {e}")
            return {"success": False, "error": str(e), "step_results": []}

    async def _execute_step(
        self, step: ScriptStep, session_id: str | None = None
    ) -> StepExecutionResult:
        """Execute a single script step"""
        try:
            if step.step_type == StepType.SEARCH:
                return await self._execute_search_step(step, session_id)
            elif step.step_type == StepType.ANALYZE:
                return await self._execute_analyze_step(step, session_id)
            elif step.step_type == StepType.COMPUTE:
                return await self._execute_compute_step(step, session_id)
            elif step.step_type == StepType.GENERATE:
                return await self._execute_generate_step(step, session_id)
            elif step.step_type == StepType.VALIDATE:
                return await self._execute_validate_step(step, session_id)
            else:
                return StepExecutionResult(
                    step_id=step.step_id or 0,
                    success=False,
                    error=f"Unknown step type: {step.step_type}",
                )

        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            return StepExecutionResult(
                step_id=step.step_id or 0, success=False, error=str(e)
            )

    async def _execute_search_step(
        self, step: ScriptStep, session_id: str | None = None
    ) -> StepExecutionResult:
        """Execute a search step"""
        # For now, simulate search - in practice this would integrate with
        # actual search tools (web search, document search, etc.)
        query = step.content

        # Mock search result
        search_result = {
            "query": query,
            "results": [
                {
                    "title": f"Mock result for: {query}",
                    "snippet": "Mock search snippet",
                },
            ],
            "result_count": 1,
        }

        return StepExecutionResult(
            step_id=step.step_id or 0,
            success=True,
            data=search_result,
            metadata={"step_type": "search", "query": query},
        )

    async def _execute_analyze_step(
        self, step: ScriptStep, session_id: str | None = None
    ) -> StepExecutionResult:
        """Execute an analysis step"""
        # Use data analysis tool
        analysis_request = step.content

        # Extract data from previous steps if referenced
        data_to_analyze = self._extract_analysis_data(step)

        if not data_to_analyze:
            return StepExecutionResult(
                step_id=step.step_id or 0,
                success=False,
                error="No data available for analysis",
            )

        # Execute analysis using computational environment
        result = await self.computational_env.execute_tool(
            "data_analysis",
            {
                "operation": "summary",
                "data": data_to_analyze,
                "options": {"request": analysis_request},
            },
            session_id=session_id,
        )

        return StepExecutionResult(
            step_id=step.step_id or 0,
            success=result["success"],
            data=result.get("data"),
            error=result.get("error"),
            metadata={"step_type": "analyze", "request": analysis_request},
        )

    async def _execute_compute_step(
        self, step: ScriptStep, session_id: str | None = None
    ) -> StepExecutionResult:
        """Execute a computation step"""
        code = step.content

        # Execute code using computational environment
        result = await self.computational_env.execute_code(code, session_id=session_id)

        return StepExecutionResult(
            step_id=step.step_id or 0,
            success=result["success"],
            data={
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "return_value": result.get("return_value"),
            },
            error=result.get("error"),
            metadata={"step_type": "compute", "code_length": len(code)},
        )

    async def _execute_generate_step(
        self, step: ScriptStep, session_id: str | None = None
    ) -> StepExecutionResult:
        """Execute a generation step"""
        # For now, this is a placeholder - in practice this would integrate
        # with LLM generation capabilities
        generation_prompt = step.content

        # Mock generation
        generated_content = f"Generated content based on: {generation_prompt}"

        return StepExecutionResult(
            step_id=step.step_id or 0,
            success=True,
            data={"generated_content": generated_content},
            metadata={"step_type": "generate", "prompt": generation_prompt},
        )

    async def _execute_validate_step(
        self, step: ScriptStep, session_id: str | None = None
    ) -> StepExecutionResult:
        """Execute a validation step"""
        validation_criteria = step.content

        # Get previous step results for validation
        previous_results = [
            result for result in self.execution_results.values() if result.success
        ]

        if not previous_results:
            return StepExecutionResult(
                step_id=step.step_id or 0,
                success=False,
                error="No previous results to validate",
            )

        # Simple validation - check if we have successful results
        validation_result = {
            "criteria": validation_criteria,
            "passed": len(previous_results) > 0,
            "validated_steps": len(previous_results),
            "details": "Basic validation passed - previous steps executed successfully",
        }

        return StepExecutionResult(
            step_id=step.step_id or 0,
            success=validation_result["passed"],
            data=validation_result,
            metadata={"step_type": "validate", "criteria": validation_criteria},
        )

    def _extract_analysis_data(self, step: ScriptStep) -> list[Any]:
        """Extract data from previous steps for analysis"""
        # Look for numeric data in previous step results
        data: list[Any] = []

        for result in self.execution_results.values():
            if result.success and result.data:
                if isinstance(result.data, dict):
                    # Extract numeric values from dictionaries
                    for value in result.data.values():
                        if isinstance(value, int | float):
                            data.append(value)
                        elif isinstance(value, list):
                            data.extend(
                                [v for v in value if isinstance(v, int | float)]
                            )
                elif isinstance(result.data, int | float):
                    data.append(result.data)
                elif isinstance(result.data, list):
                    data.extend([v for v in result.data if isinstance(v, int | float)])

        # If no data found, return sample data
        if not data:
            data = [1, 2, 3, 4, 5]  # Sample data for demonstration

        return data


async def example_usage():
    """Example of using the Script of Thought interpreter"""
    from ..computational_env.executor import ComputationalEnvironment

    # Create computational environment and interpreter
    comp_env = ComputationalEnvironment()
    interpreter = ScriptOfThoughtInterpreter(comp_env)

    # Example script
    script = """
# Search for information
SEARCH: machine learning algorithms

# Analyze the data
ANALYZE: performance metrics of different algorithms

# Compute statistics
COMPUTE:
import statistics
data = [85, 92, 78, 96, 88, 84, 90]
mean_score = statistics.mean(data)
std_dev = statistics.stdev(data)
print(f"Mean: {mean_score}, Std Dev: {std_dev}")
mean_score

# Generate insights
GENERATE: insights based on the performance analysis

# Validate results
VALIDATE: ensure all computations are reasonable
"""

    # Execute the script
    result = await interpreter.execute_script(script)
    print("Script execution result:")
    print(f"Success: {result['success']}")
    print(f"Steps: {result['completed_steps']}/{result['total_steps']}")

    for step_result in result["step_results"]:
        print(
            f"Step {step_result['step_id']}: {'✓' if step_result['success'] else '✗'}"
        )
        if step_result["error"]:
            print(f"  Error: {step_result['error']}")


if __name__ == "__main__":
    asyncio.run(example_usage())
