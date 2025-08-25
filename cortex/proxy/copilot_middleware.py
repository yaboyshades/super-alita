import glob
import re
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cortex.api.endpoints.automation import router as automation_router
from cortex.common.logging import get_logger, setup_logging
from cortex.tools.formatters import format_code_with_black
from cortex.tools.testing import run_linters, run_tests

setup_logging()
log = get_logger("cortex.proxy")

app = FastAPI(title="Cortex Copilot Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AugmentIn(BaseModel):
    message: str
    context: dict[str, Any] = {}


class AugmentOut(BaseModel):
    messages: list[dict[str, str]]
    max_tokens: int = 4000
    temperature: float = 0.1


class ProcessIn(BaseModel):
    response: str
    original_prompt: str


class FileChange(BaseModel):
    file: str
    content: str
    description: str | None = None
    formatted: bool = False
    lint_passed: bool = False


class TestResult(BaseModel):
    passed: bool
    output: str
    coverage: float | None = None


class ProcessOut(BaseModel):
    original_response: str
    changes: list[FileChange]
    test_results: TestResult | None = None
    lint_results: dict[str, Any] | None = None


def _list_repo_files(limit: int = 50) -> list[str]:
    files = []
    ignore_patterns = [
        ".git/",
        "node_modules/",
        "cortex-extension/out/",
        "__pycache__/",
        ".vscode/",
    ]
    patterns = [
        "**/*.py",
        "**/*.ts",
        "**/*.tsx",
        "**/*.js",
        "**/*.jsx",
        "**/*.json",
        "**/*.md",
        "**/*.yml",
        "**/*.yaml",
    ]
    for pattern in patterns:
        for file in glob.glob(pattern, recursive=True):
            if not any(ignored in file for ignored in ignore_patterns):
                files.append(file)
    return files[:limit]


def _build_system_message(situation_brief: str, relevant_files: list[str]) -> str:
    files_str = "\n".join(f"- {f}" for f in relevant_files[:10])
    return f"""You are a coding copilot. Use the situation brief and relevant files to produce precise, minimal diffs.

Situation Brief:
{situation_brief}

Relevant files in this repository:
{files_str}

Rules:
- Always include code blocks labeled with '# file: <path>' on the first line
- Optional '# desc: <description>' on the second line
- Keep changes minimal and focused
- Maintain existing code style and patterns
- Include tests if changing behavior
- Follow PEP 8 and project coding standards
"""


def _simple_situation_brief() -> str:
    return "Current priorities: address failing tests, improve developer experience, maintain code quality. Prefer small, focused changes."


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cortex-proxy"}


@app.post("/v1/augment-prompt", response_model=AugmentOut)
async def augment_prompt(inp: AugmentIn):
    user_msg = inp.message
    situation_brief = _simple_situation_brief()
    files = _list_repo_files(limit=20)
    sys_msg = _build_system_message(situation_brief, files)
    log.info(
        "augment_prompt",
        extra={
            "user_msg_length": len(user_msg),
            "file_count": len(files),
            "top_files": files[:5],
        },
    )
    return AugmentOut(
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=4000,
        temperature=0.1,
    )


@app.post("/v1/process-response", response_model=ProcessOut)
async def process_response(inp: ProcessIn):
    text = inp.response
    changes: list[FileChange] = []
    pattern = r"```(?:\\w+)?\\n(.*?)```"
    for match in re.finditer(pattern, text, flags=re.DOTALL):
        block = match.group(1).strip()
        lines = block.splitlines()
        if not lines:
            continue
        first_line = lines[0].strip()
        if first_line.startswith("# file:"):
            file_path = first_line.split(":", 1)[1].strip()
            description = None
            content_start = 1
            if len(lines) > 1 and lines[1].strip().startswith("# desc:"):
                description = lines[1].split(":", 1)[1].strip()
                content_start = 2
            content = "\n".join(lines[content_start:])

            formatted_content = content
            formatting_success = True

            if file_path.endswith(".py"):
                try:
                    formatted_content = format_code_with_black(content)
                    formatting_success = True
                except Exception as e:
                    log.warning("black_formatting_failed", extra={"error": str(e)})
                    formatting_success = False

                try:
                    lint_result = run_linters(content, file_path)
                    lint_passed = lint_result.get("ruff", {}).get("success", False)
                except Exception as e:
                    log.warning("ruff_linting_failed", extra={"error": str(e)})
                    lint_passed = False
            else:
                lint_passed = True

            changes.append(
                FileChange(
                    file=file_path,
                    content=formatted_content,
                    description=description,
                    formatted=formatting_success,
                    lint_passed=lint_passed,
                )
            )

    test_results = None
    if any(change.file.endswith(".py") for change in changes):
        try:
            r = run_tests()
            test_results = TestResult(
                passed=r["passed"], output=r["output"], coverage=r.get("coverage")
            )
        except Exception as e:
            log.error("test_execution_failed", extra={"error": str(e)})

    log.info(
        "process_response",
        extra={"changes_count": len(changes), "test_run": test_results is not None},
    )
    return ProcessOut(
        original_response=text, changes=changes, test_results=test_results
    )


# Mount automation API
app.include_router(automation_router)
