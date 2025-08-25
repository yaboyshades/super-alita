from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from src.core.plugin_interface import PluginInterface


class GVFNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


@dataclass
class GVF:
    id: str
    option_id: str
    prediction_type: str
    network: GVFNetwork
    optim: optim.Optimizer
    errors: Deque[float]


class PredictionEngine(PluginInterface):
    """Lightweight prediction engine producing GVF-style signals.

    Emits:
      - oak.gvf_created
      - oak.prediction_error
    Subscribes:
      - oak.option_created
      - oak.state_transition
    """

    @property
    def name(self) -> str:  # type: ignore[override]
        return "oak_prediction_engine"

    def __init__(self) -> None:
        super().__init__()
        self.cfg: dict[str, Any] = {
            "state_dim": 8,
            "learning_rate": 1e-3,
        }
        self.gvfs: Dict[str, GVF] = {}
        self.by_option: Dict[str, List[str]] = {}

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:  # type: ignore[override]
        await super().setup(event_bus, store, config)
        self.cfg.update(config or {})
        await self.subscribe("oak.option_created", self.handle_option_created)
        await self.subscribe("oak.state_transition", self.handle_state_transition)

    async def start(self) -> None:  # type: ignore[override]
        await super().start()

    async def shutdown(self) -> None:  # type: ignore[override]
        await super().shutdown()

    async def handle_option_created(self, event: Any) -> None:
        option_id = getattr(event, "option_id", None)
        if not option_id:
            return
        # Create a couple of simple GVFs per option
        for gvf_type in ("duration", "attainment"):
            gvf_id = f"gvf_{option_id}_{gvf_type}"
            if gvf_id in self.gvfs:
                continue
            net = GVFNetwork(int(self.cfg["state_dim"]))
            opt = optim.Adam(net.parameters(), lr=float(self.cfg["learning_rate"]))
            self.gvfs[gvf_id] = GVF(gvf_id, option_id, gvf_type, net, opt, deque(maxlen=64))
            self.by_option.setdefault(option_id, []).append(gvf_id)
            await self.emit_event("oak.gvf_created", gvf_id=gvf_id, option_id=option_id, prediction_type=gvf_type)

    async def handle_state_transition(self, event: Any) -> None:
        option_id = getattr(event, "option_id", None)
        if not option_id:
            return
        s = getattr(event, "state", None) or []
        ns = getattr(event, "next_state", None) or []
        reward = float(getattr(event, "reward", 0.0))
        done = bool(getattr(event, "done", False))
        gvf_ids = self.by_option.get(option_id, [])
        for gid in gvf_ids:
            g = self.gvfs.get(gid)
            if not g:
                continue
            s_t = torch.tensor(s, dtype=torch.float32)
            ns_t = torch.tensor(ns, dtype=torch.float32)
            with torch.no_grad():
                p = g.network(s_t).squeeze()
                p_next = g.network(ns_t).squeeze()
            # simple TD(0) target using reward as cumulant
            gamma = 0.9 if not done else 0.0
            target = reward + gamma * float(p_next)
            pred = g.network(s_t)
            loss = 0.5 * (pred.squeeze() - target) ** 2
            g.optim.zero_grad()
            loss.backward()
            g.optim.step()
            err = float(torch.abs(pred.squeeze() - target).item())
            g.errors.append(err)
            await self.emit_event(
                "oak.prediction_error",
                gvf_id=gid,
                option_id=option_id,
                error=err,
                prediction_type=g.prediction_type,
            )

