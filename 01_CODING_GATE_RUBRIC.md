# Coding Gate Rubric (Score ≥85 to Pass)

- **[25 pts] Test Coverage & Quality:**
  - [ ] All acceptance criteria from `10_requirements_ledger.csv` are covered by tests.
  - [ ] Tests include positive, negative, and edge cases.
  - [ ] All tests are passing.
- **[20 pts] Specification Compliance:**
  - [ ] Implementation correctly satisfies the requirement description.
  - [ ] Code adheres to any contracts defined in `11_dependencies_manifest.csv`.
- **[20 pts] Code Quality & Simplicity:**
  - [ ] Code is clean, readable, and follows project style guides.
  - [ ] No unnecessary complexity (passes the "Simplicity Gate").
  - [ ] Linter (`flake8`, `eslint`, etc.) passes with zero warnings.
- **[20 pts] Documentation:**
  - [ ] All public functions, classes, and modules have clear docstrings.
  - [ ] The `AGENTS.md` (if applicable) is updated.
- **[15 pts] Ledger & Housekeeping:**
  - [ ] The `status` of the requirement is updated in `10_requirements_ledger.csv`.
  - [ ] This gate check is logged in `99_gate_log.csv`.
