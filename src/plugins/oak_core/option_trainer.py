"""PPO-based option learning with intrinsic feature-attainment shaping."""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pydantic import BaseModel, Field

from src.core.plugin_interface import PluginInterface
from src.neural.store import MessageStore


class OptionNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
        )
        self.actor = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Linear(hidden_dim, 1)
        self.termination = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.trunk(state)
        return self.actor(h), self.critic(h), self.termination(h)


class Transition(BaseModel):
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    log_prob: float
    value: float
    features_achieved: List[str] = Field(default_factory=list)


class Option(BaseModel):
    id: str
    name: str
    subproblem_id: str
    target_features: List[str]
    state_dim: int = 100
    action_dim: int = 10
    hidden_dim: int = 128

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    episodes_trained: int = 0
    success_rate: float = 0.0
    avg_episode_length: float = 0.0
    avg_reward: float = 0.0

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: Optional[datetime] = None


class OptionTrainer(PluginInterface):
    """Trains options using PPO and emits training telemetry."""

 
    def __init__(self):
        super().__init__()
   
    Emits:
      - oak.option_created
      - oak.option_training_update
    Subscribes:
      - oak.subproblem_defined
      - oak.state_transition
      - deliberation_tick
    Attributes:
      ppo_epochs: Number of PPO optimization epochs per rollout (default 4).
    """
    

    @property
    def name(self) -> str:
        return "oak_option_trainer"


    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        self.device = torch.device(self.get_config("device", "cpu"))

        self.options: Dict[str, Option] = {}
        self.networks: Dict[str, OptionNetwork] = {}
        self.optimizers: Dict[str, torch.optim.Adam] = {}
        self.replay_buffers: Dict[str, deque] = {}

        self.batch_size = self.get_config("batch_size", 64)
        self.buffer_size = self.get_config("buffer_size", 10000)
        self.update_frequency = self.get_config("update_frequency", 100)
        self.ppo_epochs = self.get_config("ppo_epochs", 4)
        self.step_count = 0
        self.active_executions: Dict[str, Dict[str, Any]] = {}

    async def start(self) -> None:
        await super().start()
        await self.subscribe("subproblem_defined", self.create_option)
        await self.subscribe("state_transition", self.handle_transition)
        await self.subscribe("option_initiated", self.handle_option_start)
        await self.subscribe("option_terminated", self.handle_option_end)

    def generate_option_id(self, subproblem_id: str) -> str:
        ns = uuid.UUID('6ba7b812-9dad-11d1-80b4-00c04fd430c8')
        return str(uuid.uuid5(ns, f"option_{subproblem_id}"))

    async def create_option(self, event: Dict[str, Any]):
        sp_id = event.get("subproblem_id")
        feature_id = event.get("feature_id")
        kappa = event.get("kappa", 1.0)
        if not sp_id:
        
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
            "ppo_epochs": 4,
        }
        self.options: Dict[str, OptionNetwork] = {}
        self.optim: Dict[str, optim.Optimizer] = {}
        self.rollouts: Dict[str, List[List[Transition]]] = {}
        self.current: Dict[str, List[Transition]] = {}
        self.ppo_epochs = int(self.cfg["ppo_epochs"])

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:  # type: ignore[override]
        await super().setup(event_bus, store, config)
        self.cfg.update(config or {})
        self.ppo_epochs = int(self.cfg.get("ppo_epochs", 4))
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
        opt_id = self.generate_option_id(sp_id)
        if opt_id in self.options:
            return

        option = Option(
            id=opt_id,
            name=f"opt_{feature_id}_k{kappa:.1f}",
            subproblem_id=sp_id,
            target_features=[feature_id] if feature_id else [],
        )
        net = OptionNetwork(option.state_dim, option.action_dim, option.hidden_dim).to(self.device)
        optim = torch.optim.Adam(net.parameters(), lr=option.learning_rate)

        self.options[opt_id] = option
        self.networks[opt_id] = net
        self.optimizers[opt_id] = optim
        self.replay_buffers[opt_id] = deque(maxlen=self.buffer_size)

        await self.emit_event(
            "option_created",
            option_id=opt_id,
            subproblem_id=sp_id,
            target_features=option.target_features,
            timestamp=datetime.now(timezone.utc),
        )

    async def handle_option_start(self, event: Dict[str, Any]):
        opt_id = event.get("option_id")
        state = event.get("state", {})
        if opt_id in self.options:
            self.active_executions[opt_id] = {
                "start_state": state,
                "trajectory": [],
                "start_time": datetime.now(timezone.utc),
            }

    def _state_to_vector(self, state: Dict[str, Any]) -> np.ndarray:
        vec = np.zeros(100, dtype=np.float32)
        i = 0
        for k, v in state.items():
            if i >= len(vec): break
            if isinstance(v, (int, float, bool)):
                vec[i] = float(v); i += 1
            elif k == "features" and isinstance(v, list):
                for j in range(min(10, len(v))):
                    if i + j < len(vec): vec[i + j] = 1.0
                i += 10
        return vec

    def _intrinsic_reward(self, option: Option, features_achieved: List[str], next_state: Dict[str, Any]) -> float:
        for t in option.target_features:
            if t in features_achieved or t in (next_state.get("features") or []):
                return 1.0
        return 0.0

    async def handle_transition(self, event: Dict[str, Any]):
        self.step_count += 1
        state = event.get("state", {}) or {}
        action = int(event.get("action", 0))
        reward = float(event.get("reward", 0.0))
        next_state = event.get("next_state", {}) or {}
        done = bool(event.get("done", False))
        features_achieved = event.get("features", []) or []

        for opt_id, exec_ in list(self.active_executions.items()):
            option = self.options.get(opt_id)
            net = self.networks.get(opt_id)
            if not option or not net:
                continue

            intr = self._intrinsic_reward(option, features_achieved, next_state)
            total_r = reward + intr

            s_vec = self._state_to_vector(state)
            with torch.no_grad():
                probs, v, _ = net(torch.FloatTensor(s_vec).to(self.device))
                dist = Categorical(probs)
                logp = float(dist.log_prob(torch.tensor(action)))
                val = float(v.squeeze().item())

            tr = Transition(
                state=s_vec.tolist(),
                action=action,
                reward=total_r,
                next_state=self._state_to_vector(next_state).tolist(),
                done=done,
                log_prob=logp,
                value=val,
                features_achieved=features_achieved,
            )
            self.replay_buffers[opt_id].append(tr)
            exec_["trajectory"].append(tr)

        if self.step_count % self.update_frequency == 0:
            await self._train_all()

    async def _train_all(self):
        for opt_id in list(self.options.keys()):
            if len(self.replay_buffers[opt_id]) >= self.batch_size:
                await self._train_ppo(opt_id)

    async def _train_ppo(self, opt_id: str):
        opt = self.options[opt_id]
        net = self.networks[opt_id]
        optimizer = self.optimizers[opt_id]
        buf = self.replay_buffers[opt_id]

        idx = np.random.choice(len(buf), min(self.batch_size, len(buf)), replace=False)
        batch = [buf[i] for i in idx]

        states = torch.FloatTensor([t.state for t in batch]).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t.next_state for t in batch]).to(self.device)
        dones = torch.FloatTensor([t.done for t in batch]).to(self.device)
        old_logps = torch.FloatTensor([t.log_prob for t in batch]).to(self.device)
        old_vals = torch.FloatTensor([t.value for t in batch]).to(self.device)

        with torch.no_grad():
            _, next_vals, _ = net(next_states)
            next_vals = next_vals.squeeze()
            adv = torch.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + opt.gamma * next_vals[t] * (1 - dones[t]) - old_vals[t]
                gae = delta + opt.gamma * opt.gae_lambda * gae
                adv[t] = gae
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            ret = adv + old_vals

        for _ in range(self.ppo_epochs):
            probs, vals, term = net(states)
            vals = vals.squeeze()
            dist = Categorical(probs)
            new_logps = dist.log_prob(actions)
            ratio = torch.exp(new_logps - old_logps)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - opt.clip_epsilon, 1 + opt.clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(vals, ret)
            entropy = dist.entropy().mean()

            term_targets = (torch.zeros_like(term.squeeze()))
            # simple termination shaping: if any feature attained in transition, encourage termination
            for i, tr in enumerate(batch):
                if any(f in tr.features_achieved for f in opt.target_features):
                    term_targets[i] = 1.0
            term_loss = F.binary_cross_entropy(term.squeeze(), term_targets)

            loss = policy_loss + opt.value_coef * value_loss + term_loss - opt.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), opt.max_grad_norm)
            optimizer.step()
            
            for i, t in enumerate(traj):
                t.advantage = float(adv[i])

        mb = int(self.cfg["batch_size"])
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        for _ in range(self.ppo_epochs):
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
                

        opt.episodes_trained += 1
        opt.last_updated = datetime.now(timezone.utc)
        await self.emit_event(
            "option_training_update",
            option_id=opt_id,
            episodes_trained=opt.episodes_trained,
            avg_reward=float(rewards.mean().item()),
            timestamp=datetime.now(timezone.utc),
        )

    async def handle_option_end(self, event: Dict[str, Any]):
        opt_id = event.get("option_id")
        success = bool(event.get("success", False))
        exec_ = self.active_executions.pop(opt_id, None)
        opt = self.options.get(opt_id)
        if not exec_ or not opt:
            return
        traj = exec_["trajectory"]
        alpha = 0.1
        opt.success_rate = (1 - alpha) * opt.success_rate + alpha * (1.0 if success else 0.0)
        opt.avg_episode_length = (1 - alpha) * opt.avg_episode_length + alpha * len(traj)
        tot_r = sum(t.reward for t in traj) if traj else 0.0
        opt.avg_reward = (1 - alpha) * opt.avg_reward + alpha * tot_r
        await self.emit_event(
            "option_completed",
            option_id=opt_id,
            subproblem_id=opt.subproblem_id,
            success=success,
            cost=len(traj),
            reward=tot_r,
            features=opt.target_features,
            timestamp=datetime.now(timezone.utc),
        )
