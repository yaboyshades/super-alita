# Python Coding Standards

PYTHON CODING STANDARDS FOR THIS PROJECT:

- Follow PEP 8 style guide strictly
- Use type hints for all function parameters and returns
- Prefer f-strings over .format() or % formatting
- Use dataclasses or Pydantic models for data structures
- Always handle exceptions explicitly with proper logging
- Use async/await for I/O operations
- Write docstrings in Google or NumPy style
- Use pathlib for file operations, not os.path
- Prefer composition over inheritance
- Use context managers for resource management
- Keep functions under 20 lines when possible
- Use descriptive variable names (no single letters except loop counters)

TESTING STANDARDS:

- Use pytest with fixtures for setup/teardown
- Achieve >90% test coverage where feasible
- Use parametrized tests for multiple test cases
- Mock external dependencies properly
- Include integration tests for API endpoints

