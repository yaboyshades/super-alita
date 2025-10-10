from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:  # soft dependency
    from z3 import (  # type: ignore
        Int,
        IntVal,
        Solver,
        sat,
    )

    Z3_AVAILABLE = True
except Exception:  # pragma: no cover - if z3 not present
    Z3_AVAILABLE = False


@dataclass
class ConstraintAnalysis:
    variables: set[str]
    operators: set[str]
    nesting_depth: int
    estimated_cost: float


class ConstraintComplexityAnalyzer:
    """Heuristic constraint complexity analyzer suitable for gating timeouts."""

    def _extract_variables(self, c: dict[str, Any]) -> set[str]:
        vars_: set[str] = set()
        for key in ("left", "right"):
            val = c.get(key)
            if isinstance(val, str) and val.isidentifier():
                vars_.add(val)
        # Collect listed vars
        if isinstance(c.get("variables"), list):
            for v in c["variables"]:
                if isinstance(v, str):
                    vars_.add(v)
        return vars_

    def _extract_operators(self, c: dict[str, Any]) -> set[str]:
        op = str(c.get("op") or c.get("operator") or "").strip()
        return {op} if op else set()

    def _estimate_cost(self, c: dict[str, Any], var_count: int) -> float:
        kind = str(c.get("type") or "").lower()
        base = 1.0
        if kind in {"behavioral_constraint", "relation", "ineq", "eq"}:
            base = 2.5
        if kind in {"performance_constraint"}:
            base = 2.0
        return base + 0.3 * var_count

    async def analyze(
        self, constraints: list[dict[str, Any]]
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "constraints": {},
            "total_variables": 0,
            "overall_complexity": 0.0,
        }
        all_vars: set[str] = set()
        total_cost = 0.0
        for idx, c in enumerate(constraints):
            vars_ = self._extract_variables(c)
            ops = self._extract_operators(c)
            cost = self._estimate_cost(c, len(vars_))
            all_vars |= vars_
            total_cost += cost
            out["constraints"][str(idx)] = {
                "variables": sorted(vars_),
                "operators": sorted(ops),
                "nesting_depth": 0,
                "estimated_cost": cost,
            }
        out["total_variables"] = len(all_vars)
        out["overall_complexity"] = total_cost
        return out


class ScalableZ3Verifier:
    """Minimal z3 integration with constraint minimization and adaptive timeouts.

    Supported constraint formats (dicts):
      - {"type": "eq"|"ineq"|"relation", "op": "==","!=","<",">","<=",">=", "left": <str|int>, "right": <str|int>}
      - Variables are inferred from identifiers in left/right.
    """

    def __init__(self, base_timeout: int = 10, max_timeout: int = 60) -> None:
        self.base_timeout = max(1, int(base_timeout))
        self.max_timeout = max(self.base_timeout, int(max_timeout))
        self.analyzer = ConstraintComplexityAnalyzer()

    async def analyze_constraints(
        self, constraints: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await self.analyzer.analyze(constraints)

    async def minimize_constraints(
        self, constraints: list[dict[str, Any]], analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        # Keep type/correctness constraints always; defer performance/style when cost is high
        essential: list[dict[str, Any]] = []
        nice: list[dict[str, Any]] = []
        cost_map: dict[str, float] = {}
        cm = analysis.get("constraints", {})
        for idx, c in enumerate(constraints):
            key = str(idx)
            cost = float(cm.get(key, {}).get("estimated_cost", 1.0))
            cost_map[key] = cost
            kind = str(c.get("type") or "").lower()
            if kind in {
                "type_constraint",
                "correctness",
                "eq",
                "ineq",
                "relation",
            }:
                essential.append(c)
            elif kind in {"performance_constraint", "style_constraint"}:
                nice.append(c)
            else:
                essential.append(c)

        # add nice-to-have if within budget
        budget = 10.0
        current = sum(
            cost_map.get(str(constraints.index(c)), 1.0) for c in essential
        )
        for c in nice:
            ckey = str(constraints.index(c))
            ccost = cost_map.get(ckey, 1.0)
            if current + ccost <= budget:
                essential.append(c)
                current += ccost
        return essential

    def _adaptive_timeout(self, analysis: dict[str, Any]) -> int:
        comp = float(analysis.get("overall_complexity", 1.0))
        to = int(self.base_timeout * (1.0 + comp / 5.0))
        return min(max(1, to), self.max_timeout)

    async def verify(
        self, constraints: list[dict[str, Any]], timeout_s: int | None = None
    ) -> dict[str, Any]:
        if not Z3_AVAILABLE:
            return {"is_valid": False, "error": "z3-solver not installed"}
        solver = Solver()
        solver.set("timeout", int(1000 * (timeout_s or self.base_timeout)))
        # Collect variables
        var_objs: dict[str, Any] = {}

        def _to_term(t: Any):
            if isinstance(t, int):
                return IntVal(t)
            if isinstance(t, str):
                if t.isidentifier():
                    if t not in var_objs:
                        var_objs[t] = Int(t)
                    return var_objs[t]
                try:
                    return IntVal(int(t))
                except Exception:
                    # unsupported literal -> create a fresh int symbol
                    if t not in var_objs:
                        var_objs[t] = Int(t)
                    return var_objs[t]
            # Fallback
            return IntVal(0)

        # Build constraints
        for c in constraints:
            op = str(c.get("op") or c.get("operator") or "").strip()
            left = _to_term(c.get("left"))
            right = _to_term(c.get("right"))
            if op in {"==", "eq"}:
                solver.add(left == right)
            elif op in {"!=", "neq"}:
                solver.add(left != right)
            elif op == ">":
                solver.add(left > right)
            elif op == "<":
                solver.add(left < right)
            elif op == ">=":
                solver.add(left >= right)
            elif op == "<=":
                solver.add(left <= right)
            else:
                # Unknown op -> ignore but log
                logger.debug("unknown op: %s", op)

        res = solver.check()
        return {"is_valid": bool(res == sat), "solver_result": str(res)}
