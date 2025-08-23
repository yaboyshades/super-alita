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

## Gemini API Key
Super Alita relies on Google's Gemini models for many LLM features. Set your
API key in the environment (or a `.env` file) before running the agent:

```bash
export GEMINI_API_KEY="your-key-here"
```

If you prefer a `.env` file, create one in the project root containing:

```
GEMINI_API_KEY=your-key-here
```

**Never** commit your API key or `.env` file to version control.

## Documentation
Additional design and reference guides live in the `docs/` directory:

- `docs/architecture.md` outlines the "minimal predefinition, maximal self-evolution" architecture.
- `docs/testing.md` explains how to run the test suite and property-based checks.

## Running Tests
Install dependencies and execute the runtime suite:

```bash
make deps
make test
```

The runtime tests use in-memory fakes and do not require Redis.
For demo scripts such as `complete_agent_demo.py`, run a local Redis server
(`docker run -p 6379:6379 redis`) to enable the event bus; otherwise the demo
logs a connection error and falls back to a no-op bus.

The `tests/` folder covers core utilities, planner logic, plugins, and integration flows.
