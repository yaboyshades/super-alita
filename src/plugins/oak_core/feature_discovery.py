from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.core.plugin_interface import PluginInterface


@dataclass
class Feature:
    id: str
    base_ids: List[str]
    feature_type: str
    utility: float = 0.0
    alpha: float = 0.01  # IDBD-like adaptive step size (simplified)
    gradient_trace: float = 0.0
    hessian_trace: float = 0.0
    usage_count: int = 0
    evaluator: Optional[Callable[[dict[str, Any]], float]] = None
    created_at: float = 0.0


class FeatureDiscoveryEngine(PluginInterface):
    """Online discovery and utility-tracking of features/abstractions.

    Emits:
      - oak.feature_created
      - oak.features_discovered
      - oak.feature_utility_updated
    Subscribes:
      - deliberation_tick
      - oak.feature_utility_updated
    """

    @property
    def name(self) -> str:  # type: ignore[override]
        return "oak_feature_discovery"

    def __init__(self) -> None:
        super().__init__()
        self.features: Dict[str, Feature] = {}
        self.feature_emas: Dict[str, Dict[str, float]] = {}
        self.recent_observations: deque[dict[str, Any]] = deque(maxlen=64)
        self.cfg: dict[str, Any] = {
            "max_features": 128,
            "proposal_rate_limit": 4,
            "utility_ema_decay": 0.99,
            "idbd_meta_rate": 0.01,
        }

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:  # type: ignore[override]
        await super().setup(event_bus, store, config)
        self.cfg.update(config or {})
        await self.subscribe("deliberation_tick", self.handle_tick)
        await self.subscribe("oak.feature_utility_updated", self.handle_utility_update)

    async def start(self) -> None:  # type: ignore[override]
        await super().start()

    async def shutdown(self) -> None:  # type: ignore[override]
        await super().shutdown()

    @staticmethod
    def _feature_id(feature_type: str, base_ids: List[str]) -> str:
        namespace = uuid.NAMESPACE_URL
        name = f"{feature_type}:{':'.join(sorted(base_ids))}"
        return f"feature_{uuid.uuid5(namespace, name)}"

    def _propose_candidates(self) -> List[Feature]:
        """Very lightweight candidate generator.

        - Seeds at least one primitive feature if empty
        - Limits new proposals per tick
        """
        proposals: List[Feature] = []
        if not self.features:
            fid = self._feature_id("primitive", ["s0"])
            proposals.append(
                Feature(
                    id=fid,
                    base_ids=["s0"],
                    feature_type="primitive",
                    evaluator=lambda s: float(s.get("s0", 0.0)),
                    created_at=time.time(),
                )
            )
        return proposals[: self.cfg["proposal_rate_limit"]]

    async def handle_tick(self, event: Any) -> None:
        if len(self.features) >= self.cfg["max_features"]:
            return
        candidates = self._propose_candidates()
        if not candidates:
            return
        new_ids: List[str] = []
        for cand in candidates:
            if cand.id in self.features:
                continue
            self.features[cand.id] = cand
            new_ids.append(cand.id)
            await self.emit_event(
                "oak.feature_created",
                feature_id=cand.id,
                base_ids=cand.base_ids,
                feature_type=cand.feature_type,
                created_at=cand.created_at,
            )
        if new_ids:
            await self.emit_event("oak.features_discovered", feature_ids=new_ids)

    async def handle_utility_update(self, event: Any) -> None:
        # Accept flexible fields from BaseEvent
        feature_id = getattr(event, "feature_id", None)
        signal_type = getattr(event, "signal_type", None)
        value = getattr(event, "value", None)
        components = getattr(event, "components", None) or {}
        if not feature_id or feature_id not in self.features or value is None:
            return

        # Initialize EMA bins
        if feature_id not in self.feature_emas:
            self.feature_emas[feature_id] = {
                "play": 0.0,
                "prediction": 0.0,
                "planning": 0.0,
                "novelty": 0.0,
            }

        decay = float(self.cfg["utility_ema_decay"])
        emas = self.feature_emas[feature_id]

        # Update the EMA for any provided components
        for comp_name, comp_value in components.items():
            current = emas.get(comp_name, comp_value)
            emas[comp_name] = decay * current + (1.0 - decay) * float(comp_value)

        # IDBD-like step-size adaptation
        feat = self.features[feature_id]
        pred = feat.utility
        err = float(value) - pred
        feat.gradient_trace = decay * feat.gradient_trace + err
        feat.hessian_trace = decay * feat.hessian_trace + err * err
        if feat.hessian_trace > 1e-8:
            feat.alpha = max(
                1e-6,
                min(0.1, feat.alpha + float(self.cfg["idbd_meta_rate"]) * feat.gradient_trace * err / feat.hessian_trace),
            )

        # Fuse utilities (simple weighted sum)
        fused = 0.3 * emas.get("play", 0.0) + 0.3 * emas.get("prediction", 0.0) + 0.3 * emas.get("planning", 0.0) + 0.1 * emas.get("novelty", 0.0)
        feat.utility = pred + feat.alpha * (fused - pred)
        feat.usage_count += 1

        await self.emit_event(
            "oak.feature_utility_updated",
            feature_id=feature_id,
            utility=feat.utility,
            components=emas.copy(),
        )

