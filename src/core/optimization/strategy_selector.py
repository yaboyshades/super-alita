from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .bandits import (
    BanditAlgorithm,
    EpsilonGreedyBandit,
    ThompsonSamplingBandit,
    UCB1Bandit,
)

StrategyAlgo = Literal["thompson", "ucb1", "epsilon"]


@dataclass
class StrategyDecision:
    decision_id: str
    task_type: str
    arm_id: str
    arm_name: str
    metadata: dict[str, Any]
    algorithm: str
    confidence: float
    timestamp: float


class StrategySelector:
    """
    Selects reasoning styles per task type using multi-armed bandits.

    - Loads arms and algorithm per task type from `config/strategies.json`
    - Uses Thompson/UCB1/Epsilon-greedy to select an arm
    - Accepts feedback to update rewards
    - Persists stats back to `config/strategies.json`
    """

    def __init__(self, config_path: str = "config/strategies.json") -> None:
        self.config_file = Path(config_path)
        self._cfg: dict[str, Any] = {}
        self._bandits: dict[str, BanditAlgorithm] = {}
        self._load()

    def _load(self) -> None:
        if not self.config_file.exists():
            self._cfg = {"version": "0.0.0", "task_types": {}}
            return
        with open(self.config_file, encoding="utf-8") as f:
            self._cfg = json.load(f)

    def _save(self) -> None:
        self._cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self._cfg, f, indent=2)

    def _get_algo(self, task_type: str) -> BanditAlgorithm:
        if task_type in self._bandits:
            return self._bandits[task_type]

        spec = self._cfg.get("task_types", {}).get(task_type)
        if not spec:
            raise KeyError(f"Unknown task_type: {task_type}")

        algo: StrategyAlgo = spec.get("algorithm", "thompson")
        if algo == "thompson":
            bandit = ThompsonSamplingBandit()
        elif algo == "ucb1":
            bandit = UCB1Bandit()
        elif algo == "epsilon":
            bandit = EpsilonGreedyBandit(epsilon=float(spec.get("epsilon", 0.1)))
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

        # Add arms
        for arm in spec.get("arms", []):
            bandit.add_arm(arm_id=arm["id"], name=arm.get("name", arm["id"]), metadata=arm.get("metadata", {}))

        self._bandits[task_type] = bandit
        return bandit

    def select(self, task_type: str, context: dict[str, Any] | None = None) -> StrategyDecision:
        bandit = self._get_algo(task_type)
        decision = bandit.select_arm(context=context or {"task_type": task_type})

        # Resolve metadata from config
        arms = {arm["id"]: arm for arm in self._cfg["task_types"][task_type].get("arms", [])}
        meta = arms.get(decision.arm_id, {}).get("metadata", {})

        return StrategyDecision(
            decision_id=decision.decision_id,
            task_type=task_type,
            arm_id=decision.arm_id,
            arm_name=decision.arm_name,
            metadata=meta,
            algorithm=decision.algorithm,
            confidence=decision.confidence,
            timestamp=decision.timestamp,
        )

    def feedback(self, task_type: str, decision_id: str, reward: float) -> bool:
        bandit = self._get_algo(task_type)
        ok = bandit.update_reward(decision_id=decision_id, reward=reward)
        if ok:
            # Persist minimal stats back to config
            stats = bandit.get_statistics()
            self._cfg["task_types"][task_type]["stats"] = stats
            self._save()
        return ok

