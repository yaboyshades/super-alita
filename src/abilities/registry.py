from __future__ import annotations

import re
from typing import Any

SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*[a-z0-9]$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def validate_ability_registration(
    obj: Any,
    *,
    enforce_schemas: bool = False,
) -> tuple[bool, list[str]]:
    """Validate an ability instance for constitutional compliance.

    Checks:
      - required attributes: name, description, version
      - naming: snake_case for name, semantic version for version
      - required methods: initialize, validate_input, execute, health_check, shutdown
    Returns: (is_valid, errors)
    """
    errors: list[str] = []

    # Attributes
    for attr in ("name", "description", "version"):
        if not hasattr(obj, attr):
            errors.append(f"missing attribute: {attr}")

    # Name format
    name = getattr(obj, "name", None)
    if not isinstance(name, str) or not SNAKE_RE.match(name):
        errors.append("invalid name (expected snake_case)")

    # Version format
    version = getattr(obj, "version", None)
    if not isinstance(version, str) or not SEMVER_RE.match(version):
        errors.append("invalid version (expected x.y.z)")

    # Methods
    for method in (
        "initialize",
        "validate_input",
        "execute",
        "health_check",
        "shutdown",
    ):
        if not callable(getattr(obj, method, None)):
            errors.append(f"missing method: {method}")

    # Optional schema checks
    if enforce_schemas:
        in_schema = getattr(obj, "input_schema", None)
        out_schema = getattr(obj, "output_schema", None)

        # Input schema checks
        if not isinstance(in_schema, dict):
            errors.append("input_schema missing or not a dict")
        else:
            if in_schema.get("type") != "object":
                errors.append("input_schema.type must be 'object'")
            props = in_schema.get("properties")
            if not isinstance(props, dict):
                errors.append("input_schema.properties must be a dict")
            req = in_schema.get("required", [])
            if req is not None and not isinstance(req, list):
                errors.append(
                    "input_schema.required must be a list if present"
                )
            # If required present, required keys should exist in properties
            if isinstance(req, list) and isinstance(props, dict):
                for k in req:
                    if k not in props:
                        errors.append(
                            f"input_schema.required key '{k}' missing in properties"
                        )

        # Output schema checks
        if not isinstance(out_schema, dict):
            errors.append("output_schema missing or not a dict")
        else:
            if out_schema.get("type") != "object":
                errors.append("output_schema.type must be 'object'")
            oprops = out_schema.get("properties")
            if not isinstance(oprops, dict):
                errors.append("output_schema.properties must be a dict")
            oreq = out_schema.get("required", [])
            if oreq is not None and not isinstance(oreq, list):
                errors.append(
                    "output_schema.required must be a list if present"
                )
            # success must be declared and required
            if isinstance(oprops, dict):
                success_prop = oprops.get("success")
                if not isinstance(success_prop, dict):
                    errors.append(
                        "output_schema.properties.success missing or not a dict"
                    )
                else:
                    stype = success_prop.get("type")
                    if stype != "boolean":
                        errors.append(
                            "output_schema.properties.success.type must be 'boolean'"
                        )
            if isinstance(oreq, list) and "success" not in oreq:
                errors.append("output_schema.required must include 'success'")

    return (not errors), errors


def list_ability_names(abilities: dict[str, Any]) -> list[str]:
    """List registered ability names from a mapping."""
    return sorted(abilities.keys())
