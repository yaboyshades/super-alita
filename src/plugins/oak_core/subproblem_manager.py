from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

from src.core.plugin_interface import PluginInterface


@dataclass
class Subproblem:
    id: str
    feature_id: str
    kappa: float
    success_count: int = 0
    attempt_count: int = 0
    avg_cost: float = 0.0
    created_at: float = 0.0


class SubproblemManager(PluginInterface):
    """Defines and adapts subproblems around high-utility features.

    Emits:
      - oak.subproblem_defined
      - oak.subproblem_updated
    Subscribes:
      - oak.feature_utility_updated
      - oak.option_completed
    """

    @property
    def name(self) -> str:  # type: ignore[override]
        return "oak_subproblem_manager"

    def __init__(self) -> None:
        super().__init__()
        self.cfg: dict[str, Any] = {
            "min_utility_threshold": 0.2,
            "initial_kappa": 1.0,
            "kappa_adaptation_rate": 0.1,
            "min_kappa": 0.1,
            "max_kappa": 10.0,
            "max_per_feature": 1,
        }
        self.subproblems: Dict[str, Subproblem] = {}
        self.feature_to_sub: Dict[str, List[str]] = {}

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:  # type: ignore[override]
        await super().setup(event_bus, store, config)
        self.cfg.update(config or {})
        await self.subscribe("oak.feature_utility_updated", self.handle_feature_utility)
        await self.subscribe("oak.option_completed", self.handle_option_completed)

    async def start(self) -> None:  # type: ignore[override]
        await super().start()

    async def shutdown(self) -> None:  # type: ignore[override]
        await super().shutdown()

    async def handle_feature_utility(self, event: Any) -> None:
        feature_id = getattr(event, "feature_id", None)
        utility = float(getattr(event, "utility", 0.0))
        if not feature_id:
            return
        existing = self.feature_to_sub.get(feature_id, [])
        if utility < float(self.cfg["min_utility_threshold"]) or len(existing) >= int(self.cfg["max_per_feature"]):
            return
        kappa = float(self.cfg["initial_kappa"]) * 1.0
        sub_id = self._subproblem_id(feature_id, kappa)
        if sub_id in self.subproblems:
            return
        sub = Subproblem(
            id=sub_id,
            feature_id=feature_id,
            kappa=kappa,
            created_at=time.time(),
        )
        self.subproblems[sub_id] = sub
        self.feature_to_sub.setdefault(feature_id, []).append(sub_id)
        await self.emit_event(
            "oak.subproblem_defined",
            subproblem_id=sub_id,
            feature_id=feature_id,
            kappa=kappa,
        )

    async def handle_option_completed(self, event: Any) -> None:
        sub_id = getattr(event, "subproblem_id", None)
        success = bool(getattr(event, "success", False))
        cost = float(getattr(event, "cost", 0.0))
        if not sub_id or sub_id not in self.subproblems:
            return
        sp = self.subproblems[sub_id]
        sp.attempt_count += 1
        if success:
            sp.success_count += 1
        # EWMA for avg cost
        if sp.attempt_count == 1:
            sp.avg_cost = cost
        else:
            sp.avg_cost = 0.9 * sp.avg_cost + 0.1 * cost

        rate = sp.success_count / max(1, sp.attempt_count)
        eff = 1.0 / (1.0 + max(0.0, sp.avg_cost))
        new_kappa = sp.kappa
        if rate > 0.7 and eff > 0.5:
            new_kappa = sp.kappa * (1.0 + float(self.cfg["kappa_adaptation_rate"]))
        elif rate < 0.3 or eff < 0.2:
            new_kappa = sp.kappa * (1.0 - float(self.cfg["kappa_adaptation_rate"]))
        new_kappa = min(float(self.cfg["max_kappa"]), max(float(self.cfg["min_kappa"]), new_kappa))
        if abs(new_kappa - sp.kappa) > 1e-3:
            sp.kappa = new_kappa
            await self.emit_event("oak.subproblem_updated", subproblem_id=sub_id, new_kappa=new_kappa)

    @staticmethod
    def _subproblem_id(feature_id: str, kappa: float) -> str:
        ns = uuid.NAMESPACE_URL
        name = f"sub:{feature_id}:{kappa:.4f}"
        return f"subproblem_{uuid.uuid5(ns, name)}"

