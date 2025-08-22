from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

# Type-safe imports
if TYPE_CHECKING:
    from ...core.neural_atom import NeuralAtom
else:
    # Runtime fallback
    try:
        from ...core.neural_atom import NeuralAtom
    except ImportError:
        # Fallback definition if import fails
        class NeuralAtom:
            def __init__(self, metadata):
                self.metadata = metadata


class EnergyState(BaseModel):
    """Performance-as-energy: lower = more stable/reactive."""

    latency_eV: float = 0.0  # normalized latency cost
    cost_eV: float = 0.0  # normalized dollar cost
    success_prob: float = 1.0  # 0..1


class OrbitalContext(BaseModel):
    """Where reactions can occur."""

    session_id: str
    scope: str = "local"  # "local" | "restricted" | "public"
    resources: dict[str, Any] = Field(
        default_factory=dict
    )  # e.g. {"cpu": "1", "mem":"512MB"}


class CatalystQuark(BaseModel):
    """A guard that modifies context without being 'consumed'."""

    context_modifiers: dict[str, Any] = Field(default_factory=dict)
    pre_hooks: list[str] = Field(default_factory=list)  # e.g., ["validate_jwt"]
    post_hooks: list[str] = Field(default_factory=list)  # e.g., ["log_audit"]


# ---------- IO overlap & bond strength ---------------------------------------


def _schema_props(schema: dict[str, Any]) -> set[str]:
    """Extract property names from a schema dictionary."""
    props = (schema or {}).get("properties") or {}
    return set(props.keys()) if isinstance(props, dict) else set()


def io_overlap(atom_a: NeuralAtom, atom_b: NeuralAtom) -> float:
    """
    Simple Jaccard overlap on input properties.
    For now, use metadata capabilities as proxy for IO compatibility.
    """
    if not hasattr(atom_a, "metadata") or not hasattr(atom_b, "metadata"):
        return 0.0

    # Use capabilities as proxy for IO schema
    A = set(getattr(atom_a.metadata, "capabilities", []))
    B = set(getattr(atom_b.metadata, "capabilities", []))

    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def bond_strength(
    atom_a: NeuralAtom, atom_b: NeuralAtom, max_latency_ms: int = 2000
) -> float:
    """
    Bond ~ IO-compatibility + performance affinity (lower latency gap → stronger).
    """
    io_score = io_overlap(atom_a, atom_b)

    # Get latency metrics from atoms
    la = getattr(atom_a, "average_latency_ms", max_latency_ms)
    lb = getattr(atom_b, "average_latency_ms", max_latency_ms)

    # normalize absolute latency gap (smaller gap → closer to 1.0)
    gap = abs((la or max_latency_ms) - (lb or max_latency_ms))
    perf_affinity = 1.0 - min(1.0, gap / max_latency_ms)

    # Get success rates
    sr_a = getattr(getattr(atom_a, "metadata", None), "success_rate", 0.0) or 0.0
    sr_b = getattr(getattr(atom_b, "metadata", None), "success_rate", 0.0) or 0.0
    reliability = 0.5 * (sr_a + sr_b)

    # weights tuned for stability & composability
    return float(0.6 * io_score + 0.25 * perf_affinity + 0.15 * reliability)
