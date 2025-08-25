from __future__ import annotations

"""Ability registry with JSON Schema validation and file persistence.

Contracts are loaded from and saved to ``REUG_TOOL_REGISTRY_DIR`` when set,
enabling runtime discovery across process restarts.
"""

import json
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema.exceptions import SchemaError, ValidationError

from .config import SETTINGS


class AbilityRegistry:
    """Registry responsible for tracking and executing tool abilities."""

    def __init__(self, registry_dir: str | None = None) -> None:
        self._known: set[str] = set()
        self._contracts: dict[str, dict[str, Any]] = {}

        dir_path = registry_dir or SETTINGS.tool_registry_dir
        self._registry_dir: Path | None = None
        if dir_path:
            self._registry_dir = Path(dir_path)
            self._registry_dir.mkdir(parents=True, exist_ok=True)
            for file in self._registry_dir.glob("*.json"):
                with file.open("r", encoding="utf-8") as f:
                    contract = json.load(f)
                tid = contract.get("tool_id")
                if tid:
                    self._contracts[tid] = contract
                    self._known.add(tid)

        # seed with basic echo tool if absent
        if "echo" not in self._known:
            self._contracts["echo"] = {
                "tool_id": "echo",
                "description": "Echo back the provided payload",
                "input_schema": {
                    "type": "object",
                    "properties": {"payload": {"type": "string"}},
                },
                "output_schema": {"type": "object"},
            }
            self._known.add("echo")

    # --- query helpers -------------------------------------------------
    def get_available_tools_schema(self) -> list[dict[str, Any]]:
        return list(self._contracts.values())

    def knows(self, tool_name: str) -> bool:
        return tool_name in self._known

    def validate_args(self, tool_name: str, args: dict[str, Any]) -> bool:
        contract = self._contracts.get(tool_name)
        if not contract:
            return False
        schema = contract.get("input_schema", {})
        try:
            jsonschema.validate(instance=args, schema=schema)
            return True
        except ValidationError:
            return False

    # --- lifecycle -----------------------------------------------------
    async def health_check(self, contract: dict[str, Any]) -> bool:  # pragma: no cover - placeholder
        return True

    async def register(self, contract: dict[str, Any]) -> None:
        tid = contract.get("tool_id")
        if not tid:
            raise ValueError("contract missing tool_id")

        try:
            if "input_schema" in contract:
                jsonschema.Draft7Validator.check_schema(contract["input_schema"])
            if "output_schema" in contract:
                jsonschema.Draft7Validator.check_schema(contract["output_schema"])
        except SchemaError as exc:  # pragma: no cover - validation path
            raise ValueError(f"invalid schema: {exc}") from exc

        self._contracts[tid] = contract
        self._known.add(tid)

        if self._registry_dir:
            path = self._registry_dir / f"{tid}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(contract, f, indent=2)

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "echo":
            return {"echo": args.get("payload", "")}
        return {"ok": True, "tool": tool_name, "args": args}
