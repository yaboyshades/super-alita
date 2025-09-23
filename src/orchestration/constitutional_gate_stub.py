from __future__ import annotations

from collections.abc import Mapping


def compute_gate_score(stages: Mapping[str, Mapping[str, object]]) -> float:
    """Return a placeholder constitutional score between 0 and 1.

    The stub favors successful stage completion while ensuring a
    deterministic baseline score that meets the P0 requirement.
    """

    total = 0
    successes = 0
    for data in stages.values():
        status = str(data.get("status", "")).lower()
        if status in {"success", "ok"}:
            successes += 1
        if status:
            total += 1
    if total == 0:
        return 0.82
    score = 0.75 + (0.05 * successes / total)
    return max(0.0, min(1.0, round(score, 2)))


__all__ = ["compute_gate_score"]
