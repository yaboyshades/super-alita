# MCP + Copilot Agent Mode (VS Code Insiders)

## Quickstart
1. Create venv and install deps:
```
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\pip install -e .
.\.venv\Scripts\pre-commit install
```
2. Open in VS Code **Insiders**. Trust the workspace.
3. Ensure Python interpreter = `.venv\Scripts\python.exe`.
4. Run command: **MCP: Show Installed Servers** (should list `myCustomPythonAgent`).
5. In Copilot Chat, switch **Mode: Agent**. Try prompts:
   - `find_missing_docstrings root=src include_tests=false`
   - `format_and_lint_selection target_path=src`
   - `apply_result_pattern_refactor file_path=path\to\file.py function_name=foo dry_run=true`

## Notes
- Tools favor `dry_run` to show diffs first.
- Ruff runs before Black for stable formatting.
- For big repos, narrow targets (folders/files) for speed.
