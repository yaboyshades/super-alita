"""Agent workflow configurations for repository and paper processing.

Defines lightweight workflow schemas and an executor that calls tools via the
runtime ability registry. This allows composing multi-step tasks like Paper→Code.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WorkflowStep(BaseModel):
    step_type: str  # "tool" | "model" | "condition"
    tool_id: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    condition: str | None = None
    model: str | None = None
    prompt_template: str | None = None


class AgentWorkflow(BaseModel):
    name: str
    description: str
    tools: list[str]
    steps: list[WorkflowStep]
    variables: dict[str, Any] = Field(default_factory=dict)


PAPER_TO_CODE_WORKFLOW = AgentWorkflow(
    name="PaperToCodeTranslation",
    description="Translate research paper algorithms into production code",
    tools=[
        "paper_extract_text",
        "paper_generate_summary",
        "repo_list_files",
        "repo_read_file",
        "repo_write_file",
        "repo_search_code",
    ],
    steps=[
        WorkflowStep(
            step_type="tool",
            tool_id="paper_extract_text",
            args={"pdf_path": "{{paper_path}}"},
        ),
        WorkflowStep(
            step_type="tool",
            tool_id="paper_generate_summary",
            args={
                "pdf_path": "{{paper_path}}",
                "focus_areas": ["abstract", "methods", "algorithms"],
            },
        ),
        WorkflowStep(
            step_type="tool",
            tool_id="repo_list_files",
            args={"directory": "{{target_directory}}", "pattern": "**/*.py"},
        ),
        WorkflowStep(
            step_type="tool",
            tool_id="repo_read_file",
            args={"file_path": "{{template_file}}"},
        ),
        WorkflowStep(
            step_type="model",
            model="enhanced_consensus",
            prompt_template=(
                "Based on the paper summary and existing code template, implement the algorithm:\n\n"
                "Paper Summary: {{paper_summary}}\n"
                "Code Template: {{template_content}}\n"
                "Target Directory: {{target_directory}}\n\n"
                "Generate production-ready Python code that implements the paper's algorithm."
            ),
        ),
        WorkflowStep(
            step_type="tool",
            tool_id="repo_write_file",
            args={
                "file_path": "{{target_directory}}/{{algorithm_name}}.py",
                "content": "{{generated_code}}",
            },
        ),
    ],
)


CODE_REFACTOR_WORKFLOW = AgentWorkflow(
    name="PaperBasedRefactoring",
    description="Refactor existing code based on research paper insights",
    tools=[
        "paper_extract_text",
        "repo_search_code",
        "repo_read_file",
        "repo_write_file",
    ],
    steps=[
        WorkflowStep(
            step_type="tool",
            tool_id="paper_extract_text",
            args={"pdf_path": "{{paper_path}}"},
        ),
        WorkflowStep(
            step_type="tool",
            tool_id="repo_search_code",
            args={"query": "{{search_pattern}}", "file_pattern": "**/*.py"},
        ),
        WorkflowStep(
            step_type="tool",
            tool_id="repo_read_file",
            args={"file_path": "{{target_file}}"},
        ),
        WorkflowStep(
            step_type="model",
            model="enhanced_consensus",
            prompt_template=(
                "Refactor the existing code based on research paper insights:\n\n"
                "Paper Content: {{paper_content}}\n"
                "Current Code: {{current_code}}\n\n"
                "Apply the paper's techniques to improve the code while maintaining compatibility."
            ),
        ),
        WorkflowStep(
            step_type="tool",
            tool_id="repo_write_file",
            args={"file_path": "{{target_file}}", "content": "{{refactored_code}}"},
        ),
    ],
)


class WorkflowExecutor:
    """Execute agent workflows via the ability registry."""

    def __init__(self, ability_registry: Any):
        self.ability_registry = ability_registry

    async def execute_workflow(
        self, workflow: AgentWorkflow, context: dict[str, Any]
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        execution_context: dict[str, Any] = {**workflow.variables, **context}
        for step in workflow.steps:
            try:
                if step.step_type == "tool" and step.tool_id:
                    args = self._subst(step.args, execution_context)
                    res = await self.ability_registry.execute(step.tool_id, args)
                    key = f"step_{step.tool_id}_result"
                    execution_context[key] = res
                    results[step.tool_id] = res
                elif step.step_type == "model" and step.prompt_template:
                    prompt = self._subst(step.prompt_template, execution_context)
                    # Route to existing consensus tool for simplicity
                    res = await self.ability_registry.execute(
                        "deepconf_consensus",
                        {"prompt": prompt, "method": "weighted_vote", "num_samples": 3},
                    )
                    execution_context["model_result"] = res
                    results["model_generation"] = res
            except Exception as e:
                results[f"error_{step.step_type}"] = str(e)
                break
        return results

    def _subst(self, template: Any, ctx: dict[str, Any]) -> Any:
        if isinstance(template, str):
            out = template
            for k, v in ctx.items():
                out = out.replace(f"{{{{{k}}}}}", str(v))
            return out
        if isinstance(template, dict):
            return {k: self._subst(v, ctx) for k, v in template.items()}
        if isinstance(template, list):
            return [self._subst(x, ctx) for x in template]
        return template
