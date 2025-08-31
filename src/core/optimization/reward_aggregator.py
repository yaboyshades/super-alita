from __future__ import annotations

import os
from typing import Any


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute_reward_from_result(result: dict[str, Any]) -> float:
    """Compute a scalar reward from TaskResult-like dict.

    Signals (bounded, no prompt bloat):
    - success (primary)
    - latency vs. budget (penalize over-budget)
    - cost vs. budget (penalize over-budget)

    Reward in [0,1]. Weights are configurable via env if needed later.
    """
    success = 1.0 if bool(result.get("success", False)) else 0.0
    latency = float(result.get("execution_time") or 0.0)
    metrics: dict[str, Any] = result.get("performance_metrics", {}) or {}
    cost = float(metrics.get("cost_usd") or 0.0)

    lat_budget = float(
        os.getenv("REWARD_LATENCY_BUDGET", os.getenv("PROMPT_LATENCY_BUDGET", "2.0"))
    )
    cost_budget = float(
        os.getenv("REWARD_COST_BUDGET", os.getenv("PROMPT_COST_BUDGET", "0.25"))
    )

    # Convert to [0,1], where 1 means fully within budget
    lat_score = _clamp(1.0 - (latency / lat_budget) if lat_budget > 0 else 1.0)
    cost_score = _clamp(1.0 - (cost / cost_budget) if cost_budget > 0 else 1.0)

    # Weights: success dominant, latency and cost supplementary
    w_success = 0.6
    w_latency = 0.25
    w_cost = 0.15

    reward = (w_success * success) + (w_latency * lat_score) + (w_cost * cost_score)
    return _clamp(reward)
