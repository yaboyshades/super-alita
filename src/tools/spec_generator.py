from __future__ import annotations

import json
import re
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ToolSpec:
    name: str
    description: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    examples: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "examples": self.examples,
        }, indent=2)

    def to_markdown(self) -> str:
        return textwrap.dedent(f"""
        # Tool Spec: {self.name}

        - Description: {self.description}
        - Inputs: `{json.dumps(self.inputs)}`
        - Outputs: `{json.dumps(self.outputs)}`

        ## Examples
        {json.dumps(self.examples, indent=2)}
        """)


def sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_\-]+", "-", text.strip().lower()).strip("-")
    return slug[:60]


def draft_tool_spec(need_statement: str) -> ToolSpec:
    """Draft a structured tool spec from a plain need statement.

    Heuristics only; developer is expected to refine as needed.
    """
    name = sanitize_slug(need_statement.replace("I need", "").replace("to", "").strip() or "new_tool")
    if not name:
        name = "new_tool"
    name = name.replace("-", "_")

    # Simple IO scaffolding
    inputs = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Input source or path"},
            "options": {"type": "object", "additionalProperties": True},
        },
        "required": ["source"],
    }
    outputs = {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "metadata": {"type": "object", "additionalProperties": True},
        },
        "required": ["result"],
    }
    examples = [
        {"inputs": {"source": "PATH_OR_URL", "options": {}}, "outputs": {"result": "...", "metadata": {}}}
    ]
    return ToolSpec(name=name, description=need_statement.strip(), inputs=inputs, outputs=outputs, examples=examples)


def write_tool_spec(spec: ToolSpec, root: str = ".") -> dict[str, str]:
    int(time.time())
    md_path = Path(root) / "docs" / "tool_specs"
    json_path = Path(root) / "tools" / "specs"
    md_path.mkdir(parents=True, exist_ok=True)
    json_path.mkdir(parents=True, exist_ok=True)

    md_file = md_path / f"{spec.name}.md"
    json_file = json_path / f"{spec.name}.json"

    md_file.write_text(spec.to_markdown(), encoding="utf-8")
    json_file.write_text(spec.to_json(), encoding="utf-8")

    return {"markdown": str(md_file), "json": str(json_file)}


def render_issue_body(spec: ToolSpec) -> str:
    return textwrap.dedent(f"""
    ### New Tool Request: `{spec.name}`

    - Reason: {spec.description}
    - Owner: TBD
    - Priority: P2

    #### I/O Schema
    ```json
    {spec.to_json()}
    ```

    #### Acceptance Criteria
    - Validates input schema
    - Provides deterministic output for same input
    - Includes unit tests and docs
    - Integrates via `src/core/tool_types.py`
    """)


if __name__ == "__main__":
    # Simple CLI usage
    import argparse

    ap = argparse.ArgumentParser(description="Draft a tool spec and optional issue body")
    ap.add_argument("need", help="Need statement, e.g., 'Convert a PDF to text'", nargs="+")
    ap.add_argument("--root", default=".")
    ap.add_argument("--print-issue", action="store_true")
    args = ap.parse_args()

    spec = draft_tool_spec(" ".join(args.need))
    paths = write_tool_spec(spec, root=args.root)
    print(json.dumps({"spec": paths}, indent=2))
    if args.print_issue:
        print("\n\n--- ISSUE BODY ---\n")
        print(render_issue_body(spec))

