"""EventBus handlers for CodeAct."""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass
from typing import Any

from src.core.structures import generate_atom_id

from .actions import IPythonRunCell
from .observation import Observation
from .runner import CodeActRunner

logger = logging.getLogger(__name__)

CODEACT_NAMESPACE = uuid.UUID("7c1f2a2e-73bd-4d3c-8e34-4f6e0c7e8c20")


@dataclass(slots=True)
class CodeActStartRequest:
    code: str


@dataclass(slots=True)
class CodeActStepRequest:
    code: str


def _bond_id(src: str, bond_type: str, tgt: str) -> str:
    seed = f"{src}|{bond_type}|{tgt}"
    return str(uuid.uuid5(CODEACT_NAMESPACE, seed))


async def _emit_observation(bus: Any, code: str, obs: Observation) -> None:
    code_atom = {
        "atom_id": generate_atom_id("CODE", "", code, CODEACT_NAMESPACE),
        "atom_type": "CODE",
        "content": code,
    }
    obs_atom = obs.to_atom(CODEACT_NAMESPACE)
    await bus.emit({"event_type": "batch_atoms_created", "atoms": [code_atom, obs_atom]})
    bond = {
        "bond_id": _bond_id(code_atom["atom_id"], "OBSERVED", obs_atom["atom_id"]),
        "source_id": code_atom["atom_id"],
        "target_id": obs_atom["atom_id"],
        "bond_type": "OBSERVED",
    }
    await bus.emit({"event_type": "batch_bonds_added", "bonds": [bond]})
    await bus.emit({"event_type": "ui_notification", "message": obs_atom["content"]})


async def handle_start(event: CodeActStartRequest, bus: Any, runner: CodeActRunner) -> Observation:
    logger.debug("CodeAct start: %s", event.code)
    initial = IPythonRunCell(code=event.code)
    obs = await runner.run(initial)
    await _emit_observation(bus, event.code, obs)
    return obs


async def handle_step(event: CodeActStepRequest, bus: Any, runner: CodeActRunner) -> Observation:
    logger.debug("CodeAct step: %s", event.code)
    action = IPythonRunCell(code=event.code)
    obs = await runner.run(action)
    await _emit_observation(bus, event.code, obs)
    return obs
