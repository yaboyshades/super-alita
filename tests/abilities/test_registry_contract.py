from __future__ import annotations

from typing import Any

import pytest

from src.abilities.registry import validate_ability_registration


class _GoodAbility:
    name = "good_ability"
    description = "A valid ability"
    version = "1.0.0"

    def __init__(self) -> None:
        self.event_bus = None

    async def initialize(self, event_bus: Any) -> bool:  # noqa: ARG002
        self.event_bus = event_bus
        return True

    def validate_input(self, input_data: Any) -> bool:  # noqa: ANN401, ARG002
        return True

    async def execute(
        self, input_data: dict[str, Any]
    ) -> dict[str, Any]:  # noqa: ARG002
        return {"success": True, "data": {}}

    async def health_check(self) -> dict[str, Any]:
        return {"status": "healthy"}

    async def shutdown(self) -> None:
        return None


class _BadAbilityMissingStuff:
    # Missing description and version, methods incomplete
    name = "BadAbility"  # invalid (not snake_case)


@pytest.mark.parametrize(
    "cls,valid",
    [(_GoodAbility, True), (_BadAbilityMissingStuff, False)],
)
def test_validate_ability_registration(cls, valid):  # noqa: ANN001
    inst = cls()
    ok, errors = validate_ability_registration(inst)
    if valid:
        assert ok, f"Expected valid, got errors: {errors}"
    else:
        assert not ok
        assert errors, "Expected validation errors for invalid ability"


class _GoodAbilityWithSchemas(_GoodAbility):
    input_schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }
    output_schema = {
        "type": "object",
        "properties": {"success": {"type": "boolean"}},
        "required": ["success"],
    }


class _BadAbilitySchemas(_GoodAbility):
    input_schema = None  # type: ignore[assignment]
    output_schema = {"properties": {}}  # missing 'type'


@pytest.mark.parametrize(
    "cls,valid",
    [(_GoodAbilityWithSchemas, True), (_BadAbilitySchemas, False)],
)
def test_schema_enforcement(cls, valid):  # noqa: ANN001
    inst = cls()
    ok, errors = validate_ability_registration(inst, enforce_schemas=True)
    if valid:
        assert ok, f"Expected schemas valid, got errors: {errors}"
    else:
        assert not ok and errors


class _BadAbilityMissingSuccess(_GoodAbility):
    input_schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }
    output_schema = {
        "type": "object",
        "properties": {"data": {"type": "object"}},
        "required": ["data"],
    }


def test_schema_requires_success():
    inst = _BadAbilityMissingSuccess()
    ok, errors = validate_ability_registration(inst, enforce_schemas=True)
    assert not ok
    assert any("success" in e for e in errors)
