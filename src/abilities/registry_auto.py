from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
from typing import Any


async def auto_register_abilities(registry: Any) -> None:
    """Discover abilities in src/abilities/ and register them dynamically.

    Conventions supported (in priority order):
      1) async def register_abilities(registry) -> None
      2) ABILITY_CONTRACTS: list[dict], EXECUTORS: dict[str, callable]
      3) Provider classes with get_tools()/get_tool_executor(name)

    Include/Exclude via env:
      ALITA_ABILITIES_INCLUDE: comma separated module names (without .py)
      ALITA_ABILITIES_EXCLUDE: comma separated module names
    """
    abilities_dir = Path(__file__).parent
    include = set(
        x.strip() for x in (os.getenv("ALITA_ABILITIES_INCLUDE", "") or "").split(",") if x.strip()
    )
    exclude = set(
        x.strip() for x in (os.getenv("ALITA_ABILITIES_EXCLUDE", "") or "").split(",") if x.strip()
    )

    for file in abilities_dir.glob("*.py"):
        name = file.stem
        if name.startswith("_") or name == "registry_auto":
            continue
        if include and name not in include:
            continue
        if name in exclude:
            continue
        mod_name = f"src.abilities.{name}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        # 1) register_abilities(registry)
        try:
            fn = getattr(mod, "register_abilities", None)
            if callable(fn):
                res = fn(registry)
                if inspect.isawaitable(res):
                    await res  # type: ignore
                continue
        except Exception:
            pass

        # 2) ABILITY_CONTRACTS + EXECUTORS mapping
        try:
            contracts = getattr(mod, "ABILITY_CONTRACTS", None)
            executors = getattr(mod, "EXECUTORS", None)
            if isinstance(contracts, list) and isinstance(executors, dict):
                for c in contracts:
                    tid = c.get("tool_id")
                    exec_fn = executors.get(tid)
                    if tid and callable(exec_fn):
                        registry.register_tool(contract=c, executor=exec_fn)  # type: ignore
                continue
        except Exception:
            pass

        # 3) Provider classes
        try:
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if not hasattr(cls, "get_tools"):
                    continue
                try:
                    provider = cls()  # type: ignore[call-arg]
                except Exception:
                    continue
                get_tools = getattr(provider, "get_tools", None)
                get_exec = getattr(provider, "get_tool_executor", None)
                if callable(get_tools) and callable(get_exec):
                    tools = get_tools() or []
                    for t in tools:
                        tname = t.get("name")
                        contract = {
                            "tool_id": tname,
                            "description": t.get("description", ""),
                            "input_schema": t.get("parameters") or {"type": "object"},
                            "output_schema": {"type": "object"},
                        }
                        exec_fn = get_exec(tname)
                        if tname and callable(exec_fn):
                            registry.register_tool(contract=contract, executor=exec_fn)  # type: ignore
        except Exception:
            pass
