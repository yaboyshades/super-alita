# Ability Registry Schema

## Overview
Defines the constitutional naming, interface, and validation rules for abilities.

## Naming Rules
- Ability `name`: snake_case, regex `^[a-z][a-z0-9_]*[a-z0-9]$`
- Ability `version`: semantic version, regex `^\d+\.\d+\.\d+$`

## Interface (BaseAbility)
Implemented in `src/abilities/base_ability.py`:
- Metadata: `name`, `description`, `version`, `author?`
- Methods: `initialize(event_bus)`, `validate_input(data)`, `execute(data)`, `health_check()`, `shutdown()`
- Optional: `input_schema`, `output_schema`

## Validation Helpers
`src/abilities/registry.py` provides:
- `validate_ability_registration(obj) -> (bool, [errors])`
- `list_ability_names(mapping) -> [names]`

## Usage Example
```python
from src.abilities.registry import validate_ability_registration
from src.abilities.text_processor_ability import TextProcessorAbility

ability = TextProcessorAbility()
ok, errors = validate_ability_registration(ability)
if not ok:
    raise ValueError(f"Invalid ability: {errors}")
```

## Test Guidance
- Add unit tests under `tests/abilities/` validating naming and interface detection.
- Do not require external services; mock event bus when needed.

