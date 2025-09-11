# Constitutional Ability Creation Process

## Overview
Defines the standardized, constitutional process for creating new abilities in Super-Alita. It enforces:
- Article I (Library-First): standalone, reusable components
- Article II (Test-First): write tests first, they must fail initially
- Article III (Simplicity Gate): minimal necessary complexity
- Article VI (Knowledge Codification): document patterns/decisions

## Step-by-Step

### Phase 1: Spec + Tests (Test-First)
- Generate failing tests:
  - `python tools/generate_ability.py --name "my_ability" --mode test-only`
- Run tests to confirm failure (expected):
  - `pytest tests/abilities/test_my_ability_ability.py -q`
- Capture acceptance criteria in a short spec under `docs/abilities/` (optional).

### Phase 2: Implementation
- Scaffold implementation:
  - `python tools/generate_ability.py --name "my_ability" --mode implementation-only`
- Implement `_execute_core` per the spec.
- Iterate TDD until tests pass:
  - `pytest tests/abilities/test_my_ability_ability.py -q`

### Phase 3: Registration
- Import and register in `src/main.py` (ability registry) and optionally expose an API route:
  - `from src.abilities.my_ability_ability import MyAbilityAbility`
  - Register with `SimpleAbilityRegistry` and `/ability/execute/my_ability` if needed.

### Phase 4: QA
- Fast quality gates:
  - `python -m compileall -q src tests`
  - `ruff check .`
  - `black . -l 88 --check`
  - `mypy --strict src src/core src/sandbox || true`
- System validation (if present): `python validate_deployment.py`

## Templates & Tooling
- Test template: `templates/ability_test_template/test____abilityName____ability.py`
- Implementation template: `templates/ability_implementation_template/___abilityName____ability.py`
- Generator tool: `tools/generate_ability.py`

## Success Criteria
- All new tests pass; coverage ≥ 70% for new code paths
- No placeholder/mock code; secure input handling; event emission
- Consistent style (PEP 8, type hints), passes Ruff/Black

