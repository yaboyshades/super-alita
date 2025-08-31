# Generating Tests with GitHub Copilot (Alita DX Kit)

Use Copilot Chat to accelerate creation of both unit and integration tests for Alita components and `.alita` language sources.

## Unit Tests

1. Open the target module or file.
2. Select a function or class and invoke inline chat with a prompt like:
   `Generate pytest unit tests covering edge cases (empty input, large input, invalid types) for this function.`
3. Use the `/tests` slash command (if available in your Copilot setup) to request a comprehensive suite.
4. Review assertions for determinism (avoid brittle timing or randomness without seeding).

## Integration Tests

Be explicit about external dependencies:

Prompt example:

```text
Write an integration test for the deposit function using a mock NotificationSystem. Assert:
- balance increases by amount
- notification is sent exactly once
- audit log contains an entry with the correct transaction id
```

## Coverage Improvements

Ask Copilot:

```text
What additional tests are needed for full coverage of this file? List missing branches.
```
Then iteratively request only the missing test cases.

## Refactoring + Tests Workflow

1. Use `/explain` to understand legacy code.
2. Apply small refactors (rename, extract function) with built‑in VS Code actions.
3. Regenerate or update affected tests; ensure pre/post behavior parity using snapshot or fixture comparisons.
4. Run `pytest -q` locally; fix flaky tests immediately.

## Tips

- Seed randomness: `random.seed(1234)` in tests using stochastic components.
- Use parametrization for matrix style edge coverage.
- Prefer `pytest.raises` context managers over manual try/except.
- Keep test functions < 30 lines; extract helpers for shared setup.

## Next Steps

Integrate test generation into CI by adding a lint stage that fails if newly added public functions lack corresponding tests (heuristic name matching), or explore adding an agentic test suggestion tool via the existing MCP server.
