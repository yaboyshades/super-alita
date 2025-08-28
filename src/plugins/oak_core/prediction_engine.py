"""GVF Prediction Engine with lightweight ETD(λ)-style emphasis."""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic import BaseModel, Field
from src.core.plugin_interface import PluginInterface


class GVF(BaseModel):
    id: str
    name: str
    option_id: str
    discount: float = Field(default=0.99, ge=0.0, le=1.0)
    cumulant_type: str  # 'reward', 'feature', 'duration', 'success'
    cumulant_params: Dict[str, Any] = Field(default_factory=dict)

    learning_rate: float = 1e-3
    lambda_param: float = Field(default=0.9, ge=0.0, le=1.0)
    emphasis_decay: float = 0.9
    emphasis: float = 0.0

    prediction_error: float = 1.0
    td_error_ema: float = 0.0
    updates: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: Optional[datetime] = None


class GVFNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class PredictionEngine(PluginInterface):
    """
    Creates simple GVFs per option and trains them online with TD(0),
    augmented by an emphasis term (ETD-style) for off-policy robustness.
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "oak_prediction_engine"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        self.device = torch.device(self.get_config("device", "cpu"))
        self.state_dim = self.get_config("state_dim", 100)

        self.gvfs: Dict[str, GVF] = {}
        self.networks: Dict[str, GVFNetwork] = {}
        self.optimizers: Dict[str, torch.optim.Adam] = {}
        self.by_option: Dict[str, List[str]] = {}

    async def start(self) -> None:
        await super().start()
        await self.subscribe("option_created", self.create_gvfs_for_option)
        await self.subscribe("state_transition", self.update_predictions)

    def _gvf_id(self, option_id: str, kind: str) -> str:
        ns = uuid.UUID('6ba7b813-9dad-11d1-80b4-00c04fd430c8')
        return str(uuid.uuid5(ns, f"gvf_{option_id}_{kind}"))

    async def create_gvfs_for_option(self, event: Dict[str, Any]):
        opt_id = event.get("option_id")
        if not opt_id:
            return
        
        for kind in ("duration", "attainment"):
            gid = self._gvf_id(opt_id, kind)
            if gid in self.gvfs:
                continue
            g = GVF(
                id=gid,
                name=f"{kind}@{opt_id}",
                option_id=opt_id,
                cumulant_type=kind,
                discount=0.9 if kind == "duration" else 0.95,
                learning_rate=1e-3,
                lambda_param=0.0,
                emphasis_decay=0.9,
            )
            net = GVFNetwork(self.state_dim).to(self.device)
            opt = torch.optim.Adam(net.parameters(), lr=g.learning_rate)
            self.gvfs[gid] = g
            self.networks[gid] = net
            self.optimizers[gid] = opt
            self.by_option.setdefault(opt_id, []).append(gid)
            await self.emit_event(
                "gvf_created",
                gvf_id=gid,
                option_id=opt_id,
                prediction_type=kind,
                timestamp=datetime.now(timezone.utc),
            )

    def _state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        vec = torch.zeros(self.state_dim, dtype=torch.float32)
        i = 0
        for k, v in state.items():
            if i >= self.state_dim:
                break
            if isinstance(v, (int, float, bool)):
                vec[i] = float(v); i += 1
            elif k == "features" and isinstance(v, list):
                for j in range(min(10, len(v))):
                    if i + j < self.state_dim: vec[i + j] = 1.0
                i += 10
        return vec.to(self.device)

    def _cumulant(self, g: GVF, reward: float, next_state: Dict[str, Any]) -> float:
        if g.cumulant_type == "duration":
            return 1.0
        if g.cumulant_type == "attainment":
            target = g.cumulant_params.get("feature_id")
            if target and target in (next_state.get("features") or []):
                return 1.0
            return 0.0
        if g.cumulant_type == "reward":
            return float(reward)
        return 0.0

    async def update_predictions(self, event: Dict[str, Any]):
        opt_id = event.get("option_id")
        if not opt_id:
            return
        s = event.get("state", {}) or {}
        ns = event.get("next_state", {}) or {}
        reward = float(event.get("reward", 0.0))
        done = bool(event.get("done", False))

        for gid in self.by_option.get(opt_id, []):
            g = self.gvfs.get(gid)
            net = self.networks.get(gid)
            optim = self.optimizers.get(gid)
            if not (g and net and optim):
                continue

            s_t = self._state_to_tensor(s)
            ns_t = self._state_to_tensor(ns)

            with torch.no_grad():
                v = net(s_t).squeeze()
                v_next = net(ns_t).squeeze()

            c = self._cumulant(g, reward, ns)
            gamma = 0.0 if done else g.discount
            target = torch.tensor(c + gamma * float(v_next), dtype=torch.float32, device=self.device)
            pred = net(s_t).squeeze()

            # ETD-style emphasis update (very lightweight)
            g.emphasis = g.emphasis_decay * g.emphasis + 1.0
            loss = 0.5 * (pred - target).pow(2) * g.emphasis

            optim.zero_grad()
            loss.backward()
            optim.step()

            err = float(torch.abs(pred - target).item())
            g.updates += 1
            g.td_error_ema = 0.9 * g.td_error_ema + 0.1 * err
            g.last_updated = datetime.now(timezone.utc)

            await self.emit_event(
                "prediction_error",
                gvf_id=gid,
                option_id=opt_id,
                error=err,
                prediction_type=g.cumulant_type,
                timestamp=datetime.now(timezone.utc),
            )
