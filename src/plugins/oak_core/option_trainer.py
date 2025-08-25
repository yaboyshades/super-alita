from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.core.plugin_interface import PluginInterface


class OptionNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
        self.termination = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.trunk(x)
        return self.actor(h), self.critic(h), self.termination(h)


@dataclass
class Transition:
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    log_prob: float
    value: float


class OptionTrainer(PluginInterface):
    """Learns option policies and termination; emits training telemetry.

    Emits:
      - oak.option_created
      - oak.option_training_update
    Subscribes:
      - oak.subproblem_defined
      - oak.state_transition
      - deliberation_tick
    """

    @property
    def name(self) -> str:  # type: ignore[override]
        return "oak_option_trainer"

    def __init__(self) -> None:
        super().__init__()
        self.cfg: dict[str, Any] = {
            "state_dim": 8,
            "action_dim": 4,
            "learning_rate": 3e-4,
            "batch_size": 32,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ppo_epsilon": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_replay_size": 2000,
        }
        self.options: Dict[str, OptionNetwork] = {}
        self.optim: Dict[str, optim.Optimizer] = {}
        self.replay: Dict[str, Deque[Transition]] = {}

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:  # type: ignore[override]
        await super().setup(event_bus, store, config)
        self.cfg.update(config or {})
        await self.subscribe("oak.subproblem_defined", self.handle_subproblem_defined)
        await self.subscribe("oak.state_transition", self.handle_state_transition)
        await self.subscribe("deliberation_tick", self.handle_training_tick)

    async def handle_subproblem_defined(self, event: Any) -> None:
        sub_id = getattr(event, "subproblem_id", None)
        if not sub_id:
            return
        opt_id = f"option_{sub_id}"
        if opt_id in self.options:
            return
        net = OptionNetwork(int(self.cfg["state_dim"]), int(self.cfg["action_dim"]))
        self.options[opt_id] = net
        self.optim[opt_id] = optim.Adam(net.parameters(), lr=float(self.cfg["learning_rate"]))
        self.replay[opt_id] = deque(maxlen=int(self.cfg["max_replay_size"]))
        await self.emit_event(
            "oak.option_created",
            option_id=opt_id,
            subproblem_id=sub_id,
            state_dim=int(self.cfg["state_dim"]),
            action_dim=int(self.cfg["action_dim"]),
        )

    async def handle_state_transition(self, event: Any) -> None:
        opt_id = getattr(event, "option_id", None)
        if not opt_id or opt_id not in self.options:
            return
        s = getattr(event, "state", None) or []
        a = int(getattr(event, "action", 0))
        r = float(getattr(event, "reward", 0.0))
        ns = getattr(event, "next_state", None) or []
        done = bool(getattr(event, "done", False))
        # quick log_prob/value snapshot
        with torch.no_grad():
            logits, value, _ = self.options[opt_id](torch.tensor(s, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = float(dist.log_prob(torch.tensor(a)))
            v = float(value.squeeze())
        self.replay[opt_id].append(Transition(s, a, r, ns, done, log_prob, v))

    async def handle_training_tick(self, event: Any) -> None:
        for opt_id in list(self.options.keys()):
            buf = self.replay.get(opt_id)
            if not buf or len(buf) < int(self.cfg["batch_size"]):
                continue
            await self._train_batch(opt_id, list(buf)[-int(self.cfg["batch_size"]):])

    async def _train_batch(self, opt_id: str, batch: List[Transition]) -> None:
        net = self.options[opt_id]
        optim_ = self.optim[opt_id]
        states = torch.tensor([t.state for t in batch], dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.tensor([t.next_state for t in batch], dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)
        old_log_probs = torch.tensor([t.log_prob for t in batch], dtype=torch.float32)
        old_values = torch.tensor([t.value for t in batch], dtype=torch.float32)

        with torch.no_grad():
            _, next_values, _ = net(next_states)
            next_values = next_values.squeeze()
            deltas = rewards + float(self.cfg["gamma"]) * next_values * (1.0 - dones) - old_values
            adv = torch.zeros_like(deltas)
            gae = 0.0
            gamma = float(self.cfg["gamma"]) ; lam = float(self.cfg["gae_lambda"]) 
            for t in reversed(range(len(deltas))):
                gae = float(deltas[t]) + gamma * lam * gae
                adv[t] = gae
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            returns = adv + old_values

        logits, values, _ = net(states)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        ratio = (new_log_probs - old_log_probs).exp()
        eps = float(self.cfg["ppo_epsilon"]) 
        clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
        value_loss = 0.5 * (values.squeeze() - returns).pow(2).mean()
        entropy = dist.entropy().mean()
        loss = policy_loss + float(self.cfg["value_coef"]) * value_loss - float(self.cfg["entropy_coef"]) * entropy
        optim_.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optim_.step()

        await self.emit_event(
            "oak.option_training_update",
            option_id=opt_id,
            policy_loss=float(policy_loss.item()),
            value_loss=float(value_loss.item()),
            entropy=float(entropy.item()),
            avg_reward=float(rewards.mean().item()),
        )
