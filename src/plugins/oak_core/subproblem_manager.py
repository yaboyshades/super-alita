"""Subproblem Manager for κ-weighted reward-respecting objectives."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from src.core.plugin_interface import PluginInterface
from src.neural.atom import Atom


class Subproblem(BaseModel):
    """κ-weighted reward-respecting objective."""
    id: str
    feature_id: str
    name: str

    kappa: float = Field(default=1.0, ge=0.0, le=10.0)           # Intrinsic attainment weight
    extrinsic_weight: float = Field(default=1.0, ge=0.0, le=1.0)

    termination_features: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=100, ge=1, le=2000)

    created_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    linked_options: list[str] = Field(default_factory=list)

    # Perf
    success_rate: float = 0.0
    avg_steps_to_termination: float = 50.0
    avg_reward: float = 0.0
    total_attempts: int = 0

    def compute_reward(self, extrinsic: float, feature_achieved: bool, terminal_value: float) -> float:
        intrinsic = self.kappa * float(feature_achieved)
        return self.extrinsic_weight * extrinsic + intrinsic + terminal_value


class SubproblemManager(PluginInterface):
    """
    Manages reward-respecting subproblems for feature attainment.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cfg: dict[str, Any] = {}
        self.feature_to_sub: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return "oak_subproblem_manager"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        self.subproblems: dict[str, Subproblem] = {}
        self.feature_to_subproblems: dict[str, list[str]] = {}
        
        # Legacy config approach using get_config
        self.min_utility_threshold = self.get_config("min_utility_threshold", 0.2)
        self.kappa_range = self.get_config("kappa_range", (0.1, 5.0))
        self.kappa_values = self.get_config("kappa_values", [0.5, 1.0, 2.0])
        
        # Modern config approach using self.cfg
        self.cfg.update(config or {})
        self.cfg.setdefault("min_utility_threshold", 0.2)
        self.cfg.setdefault("max_per_feature", 5)
        
        # Subscribe to both legacy and modern event patterns
        await self.subscribe("feature_utility_update", self.handle_utility_update)
        await self.subscribe("option_created", self.link_option_to_subproblem)
        await self.subscribe("subproblem_terminated", self.handle_termination)
        await self.subscribe("oak.feature_utility_updated", self.handle_feature_utility)
        await self.subscribe("oak.option_completed", self.handle_option_completed)

    async def start(self) -> None:
        await super().start()

    async def shutdown(self) -> None:
        await super().shutdown()

    def generate_subproblem_id(self, feature_id: str, kappa: float) -> str:
        ns = uuid.UUID('6ba7b811-9dad-11d1-80b4-00c04fd430c8')
        return str(uuid.uuid5(ns, f"subproblem_{feature_id}_k{kappa:.2f}"))

    async def handle_option_completed(self, event: Any) -> None:
        """Handle completion of options linked to subproblems."""
        pass  # Implementation would track option outcomes

    async def handle_feature_utility(self, event: Any) -> None:
        feature_id = getattr(event, "feature_id", None)
        utility = float(getattr(event, "utility", 0.0))
        if not feature_id:
            return
        existing = self.feature_to_sub.get(feature_id, [])
        if utility < float(self.cfg["min_utility_threshold"]) or len(existing) >= int(self.cfg["max_per_feature"]):
            return
        for k in self.kappa_values:
            sp = await self._create_subproblem(feature_id, k)
            if sp:
                await self.emit_event(
                    "subproblem_defined",
                    subproblem_id=sp.id,
                    feature_id=feature_id,
                    kappa=k,
                    timestamp=datetime.now(UTC),
                )

    async def handle_utility_update(self, event: dict[str, Any]):
        fid = event.get("feature_id")
        utility = float(event.get("value", 0.0))
        if not fid or utility <= self.min_utility_threshold:
            return
        existing = {self.subproblems[s].kappa for s in self.feature_to_subproblems.get(fid, []) if s in self.subproblems}
        if utility > 0.7 and len(existing) < 5:
            new_k = min(self.kappa_range[1], (max(existing) * 1.5) if existing else 3.0)
            if new_k not in existing:
                sp = await self._create_subproblem(fid, new_k)
                if sp:
                    await self.emit_event(
                        "subproblem_defined",
                        subproblem_id=sp.id,
                        feature_id=fid,
                        kappa=new_k,
                        timestamp=datetime.now(UTC),
                    )

    async def _create_subproblem(self, feature_id: str, kappa: float) -> Subproblem | None:
        sp_id = self.generate_subproblem_id(feature_id, kappa)
        if sp_id in self.subproblems:
            return None
        sp = Subproblem(
            id=sp_id,
            feature_id=feature_id,
            name=f"attain_{feature_id}_k{kappa:.1f}",
            kappa=kappa,
            extrinsic_weight=1.0 / (1.0 + kappa),
            termination_features=[feature_id],
            created_by="subproblem_manager",
        )
        self.subproblems[sp_id] = sp
        self.feature_to_subproblems.setdefault(feature_id, []).append(sp_id)
        sp_dict = sp.model_dump()
        new_atom = Atom(
            atom_type="subproblem",
            title=sp.name,
            content=json.dumps(sp_dict),
            meta={"feature_id": feature_id, "kappa": kappa}
        )
        await self.store.persist(
            event_type="atom_created",
            payload=new_atom.to_dict()
        )
        return sp

    async def link_option_to_subproblem(self, event: dict[str, Any]):
        opt_id = event.get("option_id")
        sp_id = event.get("subproblem_id")
        sp = self.subproblems.get(sp_id)
        if not opt_id or not sp:
            return
        if opt_id not in sp.linked_options:
            sp.linked_options.append(opt_id)
            content_dict = {"source": sp_id, "target": opt_id, "type": "defines_objective_for"}
            new_atom = Atom(
                atom_type="bond",
                title=f"subproblem_option_link_{sp_id}_{opt_id}",
                content=json.dumps(content_dict),
                meta={"subtype": "subproblem_option"}
            )
            await self.store.persist(
                event_type="atom_created",
                payload=new_atom.to_dict()
            )
            await self.emit_event(
                "subproblem_option_linked",
                subproblem_id=sp_id,
                option_id=opt_id,
                timestamp=datetime.now(UTC),
            )

    async def handle_termination(self, event: dict[str, Any]):
        sp_id = event.get("subproblem_id")
        success = bool(event.get("success", False))
        steps = int(event.get("steps", 0))
        reward = float(event.get("reward", 0.0))
        sp = self.subproblems.get(sp_id)
        if not sp:
            return
        alpha = 0.1
        sp.total_attempts += 1
        sp.success_rate = (1 - alpha) * sp.success_rate + alpha * (1.0 if success else 0.0)
        if steps > 0:
            sp.avg_steps_to_termination = (1 - alpha) * sp.avg_steps_to_termination + alpha * steps
        sp.avg_reward = (1 - alpha) * sp.avg_reward + alpha * reward
        if sp.total_attempts % 10 == 0:
            await self._adapt_kappa(sp)

    async def _adapt_kappa(self, sp: Subproblem):
        eff = sp.success_rate / max(1.0, sp.avg_steps_to_termination / 100.0)
        old = sp.kappa
        if eff > 0.7 and sp.avg_reward > 0:
            sp.kappa = min(self.kappa_range[1], sp.kappa * 1.1)
        elif eff < 0.3 or sp.avg_reward < -1.0:
            sp.kappa = max(self.kappa_range[0], sp.kappa * 0.9)
        sp.extrinsic_weight = 1.0 / (1.0 + sp.kappa)
        if abs(sp.kappa - old) > 0.01:
            await self.emit_event(
                "subproblem_updated",
                subproblem_id=sp.id,
                old_kappa=old,
                new_kappa=sp.kappa,
                reason="performance_adaptation",
                timestamp=datetime.now(UTC),
            )
