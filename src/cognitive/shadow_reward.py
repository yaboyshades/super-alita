from __future__ import annotations

import asyncio
import ast
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:  # numpy is in requirements
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

logger = logging.getLogger(__name__)


class SimplePythonReward:
    """Lightweight, deterministic reward for Python code quality.

    Heuristics:
      - parses (syntax) -> base 0.6
      - function count >0 -> +0.1
      - return annotations present -> +0.1
      - any docstrings -> +0.1
      - no 'NotImplementedError'/'TODO'/'FIXME' -> +0.1
    Clipped to [0, 1].
    """

    async def compute_reward(self, code: str, _context: Optional[Dict[str, Any]] = None) -> float:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0

        score = 0.6
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if funcs:
            score += 0.1
        # Check annotations
        if any(f.returns is not None for f in funcs):
            score += 0.1
        # Docstrings
        has_doc = any(ast.get_docstring(f) for f in funcs) or (ast.get_docstring(tree) is not None)
        if has_doc:
            score += 0.1
        # Placeholders
        up = code.upper()
        if ("NOTIMPLEMENTEDERROR" in up) or ("TODO" in up) or ("FIXME" in up):
            score -= 0.1
        return float(max(0.0, min(1.0, score)))


@dataclass
class RewardComparison:
    stub_score: float
    torch_score: float
    correlation: float
    timestamp: datetime
    code_sample: str
    context: Dict[str, Any]


class ShadowRewardDeployment:
    """Shadow-deploy an alternate reward model with progressive rollout.

    - Always returns the stub (production) score immediately
    - Computes alt score concurrently and logs comparison
    - When rollout > 0, will occasionally substitute alt score
    """

    def __init__(self, stub_model: Any, torch_model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.stub_model = stub_model
        self.torch_model = torch_model
        self.config = config or {}
        self.comparisons: List[RewardComparison] = []
        self.correlation_threshold: float = float(self.config.get("correlation_threshold", 0.8))
        self.correlation_window: int = int(self.config.get("correlation_window", 100))
        self.torch_enabled_percentage: float = 0.0

    async def compute_reward_with_shadow(self, code: str, context: Optional[Dict[str, Any]] = None) -> float:
        context = context or {}
        # Production stub score
        stub_score = await self.stub_model.compute_reward(code, context)
        # Shadow compute alt score
        torch_task = asyncio.create_task(self._safe_torch_compute(code, context))
        # Log comparison async
        asyncio.create_task(self._collect_shadow_comparison(code, context, stub_score, torch_task))

        # Progressive rollout (best-effort)
        use_alt = False
        try:
            import random

            if self.torch_enabled_percentage > 0 and random.random() < self.torch_enabled_percentage:
                use_alt = True
        except Exception:
            pass
        if use_alt:
            try:
                alt = await torch_task
                return alt
            except Exception as e:  # pragma: no cover
                logger.warning("shadow alt failed: %s", e)
        return stub_score

    async def _safe_torch_compute(self, code: str, context: Dict[str, Any]) -> float:
        try:
            return await asyncio.wait_for(self.torch_model.compute_reward(code, context), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("torch reward timeout")
            return 0.5
        except Exception as e:  # pragma: no cover
            logger.error("torch reward error: %s", e)
            return 0.5

    async def _collect_shadow_comparison(self, code: str, context: Dict[str, Any], stub_score: float, torch_future: asyncio.Task) -> None:
        try:
            torch_score = await torch_future
            cmp = RewardComparison(
                stub_score=stub_score,
                torch_score=torch_score,
                correlation=0.0,
                timestamp=datetime.now(),
                code_sample=code[:200],
                context=context,
            )
            self.comparisons.append(cmp)
            if len(self.comparisons) > self.correlation_window * 2:
                self.comparisons = self.comparisons[-self.correlation_window :]
            await self._update_rollout()
        except Exception as e:  # pragma: no cover
            logger.warning("collect comparison failed: %s", e)

    async def _update_rollout(self) -> None:
        if np is None:
            return
        window = self.comparisons[-self.correlation_window :] if self.comparisons else []
        if len(window) < 10:
            return
        s = [c.stub_score for c in window]
        t = [c.torch_score for c in window]
        try:
            corr = float(np.corrcoef(s, t)[0, 1])
        except Exception:
            corr = 0.0
        for c in window:
            c.correlation = corr
        if corr >= 0.9:
            self.torch_enabled_percentage = min(1.0, self.torch_enabled_percentage + 0.1)
        elif corr >= 0.8:
            self.torch_enabled_percentage = min(0.5, self.torch_enabled_percentage + 0.05)
        elif corr < 0.6:
            self.torch_enabled_percentage = max(0.0, self.torch_enabled_percentage - 0.1)
        logger.info("shadow corr=%.3f, rollout=%.1f%%", corr, 100 * self.torch_enabled_percentage)

    def get_metrics(self) -> Dict[str, Any]:
        if not self.comparisons:
            return {"status": "collecting", "samples": 0, "rollout": self.torch_enabled_percentage}
        window = self.comparisons[-min(len(self.comparisons), 50) :]
        s = [c.stub_score for c in window]
        t = [c.torch_score for c in window]
        corr = 0.0
        if np is not None and len(window) > 1:
            try:
                corr = float(np.corrcoef(s, t)[0, 1])
            except Exception:
                corr = 0.0
        return {
            "status": "active",
            "samples": len(self.comparisons),
            "recent": len(window),
            "correlation": corr,
            "rollout": self.torch_enabled_percentage,
            "mean_stub": sum(s) / len(s) if s else 0.0,
            "mean_torch": sum(t) / len(t) if t else 0.0,
        }

