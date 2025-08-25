from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

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
    advantage: float = 0.0
    ret: float = 0.0


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
        self.rollouts: Dict[str, List[List[Transition]]] = {}
        self.current: Dict[str, List[Transition]] = {}

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:  # type: ignore[override]
        await super().setup(event_bus, store, config)
        self.cfg.update(config or {})
        await self.subscribe("oak.subproblem_defined", self.handle_subproblem_defined)
        await self.subscribe("oak.state_transition", self.handle_state_transition)
        await self.subscribe("deliberation_tick", self.handle_training_tick)

    async def start(self) -> None:  # type: ignore[override]
        await super().start()

    async def shutdown(self) -> None:  # type: ignore[override]
        await super().shutdown()

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
        self.rollouts[opt_id] = []
        self.current[opt_id] = []
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
        traj = self.current.setdefault(opt_id, [])
        traj.append(Transition(s, a, r, ns, done, log_prob, v))
        if done:
            rollouts = self.rollouts.setdefault(opt_id, [])
            rollouts.append(traj.copy())
            self.current[opt_id] = []

    async def handle_training_tick(self, event: Any) -> None:
        for opt_id in list(self.options.keys()):
            rollouts = self.rollouts.get(opt_id, [])
            if not rollouts:
                continue
            for traj in rollouts:
                await self._train_rollout(opt_id, traj)
            self.rollouts[opt_id] = []

    async def _train_rollout(self, opt_id: str, traj: List[Transition]) -> None:
        net = self.options[opt_id]
        optim_ = self.optim[opt_id]
        gamma = float(self.cfg["gamma"])
        lam = float(self.cfg["gae_lambda"])

        with torch.no_grad():
            next_values: List[float] = []
            for t in traj:
                if t.done:
                    next_values.append(0.0)
                else:
                    _, nv, _ = net(torch.tensor(t.next_state, dtype=torch.float32))
                    next_values.append(float(nv.squeeze()))
            gae = 0.0
            for i in reversed(range(len(traj))):
                delta = traj[i].reward + gamma * next_values[i] - traj[i].value
                gae = delta + gamma * lam * gae
                traj[i].advantage = gae
                traj[i].ret = gae + traj[i].value
            adv = torch.tensor([t.advantage for t in traj])
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for i, t in enumerate(traj):
                t.advantage = float(adv[i])

        mb = int(self.cfg["batch_size"])
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        for start in range(0, len(traj), mb):
            mb_slice = traj[start : start + mb]
            states = torch.tensor([t.state for t in mb_slice], dtype=torch.float32)
            actions = torch.tensor([t.action for t in mb_slice], dtype=torch.long)
            old_log_probs = torch.tensor([t.log_prob for t in mb_slice], dtype=torch.float32)
            returns = torch.tensor([t.ret for t in mb_slice], dtype=torch.float32)
            adv = torch.tensor([t.advantage for t in mb_slice], dtype=torch.float32)

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
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

        await self.emit_event(
            "oak.option_training_update",
            option_id=opt_id,
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy=float(np.mean(entropies)),
            avg_reward=float(np.mean([t.reward for t in traj])),
        )
