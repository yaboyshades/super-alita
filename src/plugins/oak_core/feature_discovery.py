"""Feature Discovery Engine with IDBD-style adaptive meta-learning."""

from __future__ import annotations

import json
import uuid
from collections import deque
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from src.core.plugin_interface import PluginInterface
from src.neural.atom import Atom


class Feature(BaseModel):
    """Feature with IDBD adaptive learning rates and multi-source utility."""
    id: str
    name: str
    composition_type: str  # 'primitive', 'conjunction', 'sequence', 'function', 'contrast'
    base_features: list[str] = Field(default_factory=list)

    # IDBD parameters
    learning_rate: float = Field(default=0.01, ge=0.0001, le=1.0)
    meta_learning_rate: float = Field(default=0.001, ge=0.0, le=0.1)
    gradient_trace: float = Field(default=0.0)
    hessian_trace: float = Field(default=0.0)

    # Multi-source utility signals
    play_utility: float = Field(default=0.0, ge=0.0, le=1.0)
    prediction_utility: float = Field(default=0.0, ge=0.0, le=1.0)
    planning_utility: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_score: float = Field(default=1.0, ge=0.0, le=1.0)
    combined_utility: float = Field(default=0.0, ge=0.0, le=1.0)

    # Metadata
    activation_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activated: datetime | None = None
    evaluator: str | None = None  # Serialized form (for registry/traceability)

    def compute_combined_utility(self) -> float:
        weights = {"play": 0.4, "prediction": 0.3, "planning": 0.2, "novelty": 0.1}
        return (
            weights["play"] * self.play_utility +
            weights["prediction"] * self.prediction_utility +
            weights["planning"] * self.planning_utility +
            weights["novelty"] * self.novelty_score
        )


class FeatureExtractor(nn.Module):
    """Neural feature extractor for function-based features."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim), nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FeatureDiscoveryEngine(PluginInterface):
    """Online discovery and utility-tracking of features/abstractions.

    Emits:
      - oak.feature_created
      - oak.features_discovered
      - oak.feature_utility_updated
    Subscribes:
      - deliberation_tick
      - oak.feature_utility_updated
    
    Continual feature discovery with IDBD adaptive meta-learning.
    Generates features via primitives, conjunctions, sequences, contrasts, and NN functions.
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "oak_feature_discovery"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        self.max_features = self.get_config("max_features", 1000)
        self.idbd_meta_rate = self.get_config("idbd_meta_rate", 0.01)
        self.state_dim = self.get_config("state_dim", 100)
        self.discovery_rate_limit = self.get_config("discovery_rate_limit", 10)

        self.features: dict[str, Feature] = {}
        self.feature_graph: dict[str, set[str]] = {}

        self.feature_extractor = FeatureExtractor(self.state_dim, 64, 32)
        self.extractor_optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-3)
        self.cfg.update(config or {})
        await self.subscribe("deliberation_tick", self.handle_tick)
        await self.subscribe("oak.feature_utility_updated", self.handle_utility_update)

        self.temporal_patterns = deque(maxlen=100)

    async def start(self) -> None:
        await super().start()
        await self.subscribe("observation", self.handle_observation)
        await self.subscribe("option_success", self.handle_option_feedback)
        await self.subscribe("prediction_error", self.handle_prediction_feedback)
        await self.subscribe("planning_usage", self.handle_planning_feedback)

    def generate_feature_id(self, composition: list[str], type_: str) -> str:
        ns = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        return str(uuid.uuid5(ns, f"{type_}:{'_'.join(sorted(composition))}"))

    async def handle_observation(self, event: dict[str, Any]):
        obs = event.get("data", {}) or {}
        primitives = await self._extract_primitives(obs)

        candidates: list[Feature] = []
        candidates += self._generate_conjunctions(primitives)
        candidates += self._generate_sequences()
        candidates += self._generate_functions(obs)
        candidates += self._generate_contrasts(primitives)

        added = []
        for c in candidates[: self.discovery_rate_limit]:
            if await self._evaluate_and_add(c):
                added.append(c.id)

        self.temporal_patterns.append([f.id for f in primitives[:5]])
        self._idbd_adapt_all()

        if added:
            await self.emit_event(
                "features_discovered",
                feature_ids=added,
                total_features=len(self.features),
                timestamp=datetime.now(UTC),
            )

    async def _extract_primitives(self, obs: dict[str, Any]) -> list[Feature]:
        out: list[Feature] = []
        for key in obs:
            fid = self.generate_feature_id([key], "primitive")
            if fid not in self.features:
                f = Feature(
                    id=fid,
                    name=f"prim_{key}",
                    composition_type="primitive",
                    evaluator=f"lambda s: s.get('{key}', 0.0)"
                )
                self.features[fid] = f
                f_dict = f.model_dump()
                new_atom = Atom(
                    atom_type="feature",
                    title=f.name,
                    content=json.dumps(f_dict),
                    meta={"subtype": "primitive"}
                )
                await self.store.persist(
                    event_type="atom_created",
                    payload=new_atom.to_dict()
                )
                await self.emit_event(
                    "feature_created",
                    feature_id=fid,
                    composition_type="primitive",
                    timestamp=datetime.now(UTC),
                )
            out.append(self.features[fid])
        return out

    def _generate_conjunctions(self, features: list[Feature]) -> list[Feature]:
        out: list[Feature] = []
        tops = sorted(features, key=lambda x: x.combined_utility, reverse=True)[:8]
        for i, a in enumerate(tops):
            for b in tops[i+1:]:
                comp = [a.id, b.id]
                fid = self.generate_feature_id(comp, "conjunction")
                if fid in self.features:
                    continue
                out.append(Feature(
                    id=fid,
                    name=f"conj_{a.name}_{b.name}",
                    composition_type="conjunction",
                    base_features=comp,
                    evaluator=f"lambda s: min({a.evaluator}, {b.evaluator})",
                ))
        return out

    def _generate_sequences(self) -> list[Feature]:
        out: list[Feature] = []
        if len(self.temporal_patterns) < 3:
            return out
        seq = []
        for step in list(self.temporal_patterns)[-3:]:
            seq += step[:2]
        if seq:
            fid = self.generate_feature_id(seq, "sequence")
            if fid not in self.features:
                out.append(Feature(
                    id=fid,
                    name=f"seq_{len(seq)}",
                    composition_type="sequence",
                    base_features=seq[:6],
                    evaluator="lambda s: 1.0",
                ))
        return out

    def _generate_functions(self, obs: dict[str, Any]) -> list[Feature]:
        out: list[Feature] = []
        vec = self._obs_to_vector(obs)
        with torch.no_grad():
            feats = self.feature_extractor(torch.FloatTensor(vec).unsqueeze(0)).squeeze().numpy()
        top = np.argsort(np.abs(feats))[-3:]
        for idx in top:
            val = float(feats[idx])
            if abs(val) < 0.1:
                continue
            fid = self.generate_feature_id([f"func_{idx}_{val:.3f}"], "function")
            if fid in self.features:
                continue
            out.append(Feature(
                id=fid,
                name=f"func_{idx}",
                composition_type="function",
                evaluator=f"lambda s: {val}",
            ))
        return out

    def _generate_contrasts(self, feats: list[Feature]) -> list[Feature]:
        out: list[Feature] = []
        if len(feats) < 2:
            return out
        pairs = min(3, len(feats)//2)
        idxs = np.random.choice(len(feats), size=2*pairs, replace=False).reshape(pairs, 2)
        for i, j in idxs:
            a, b = feats[i], feats[j]
            fid = self.generate_feature_id([a.id, b.id], "contrast")
            if fid in self.features:
                continue
            out.append(Feature(
                id=fid,
                name=f"contrast_{a.name}_{b.name}",
                composition_type="contrast",
                base_features=[a.id, b.id],
                evaluator=f"lambda s: abs({a.evaluator} - {b.evaluator})",
            ))
        return out

    def _obs_to_vector(self, obs: dict[str, Any]) -> np.ndarray:
        vec = np.zeros(self.state_dim, dtype=np.float32)
        i = 0
        for k, v in obs.items():
            if i >= self.state_dim:
                break
            if isinstance(v, (int, float, bool)):
                vec[i] = float(v)
            else:
                vec[i] = (hash(str(v)) % 1000) / 1000.0
            i += 1
        return vec

    async def _evaluate_and_add(self, f: Feature) -> bool:
        novelty = self._compute_novelty(f)
        f.novelty_score = novelty
        f.combined_utility = novelty * 0.5
        if novelty <= 0.3 or len(self.features) >= self.max_features:
            return False

        f.learning_rate = 0.01
        f.meta_learning_rate = self.idbd_meta_rate
        self.features[f.id] = f
        self.feature_graph[f.id] = set(f.base_features)

        f_dict = f.model_dump()
        new_atom = Atom(
            atom_type="feature",
            title=f.name,
            content=json.dumps(f_dict),
            meta={"subtype": f.composition_type}
        )
        await self.store.persist(
            event_type="atom_created",
            payload=new_atom.to_dict()
        )
        await self.emit_event(
            "feature_created",
            feature_id=f.id,
            composition_type=f.composition_type,
            timestamp=datetime.now(UTC),
        )
        return True

    def _compute_novelty(self, f: Feature) -> float:
        if f.composition_type == "primitive":
            return 1.0
        similar = 0
        for e in self.features.values():
            if e.composition_type != f.composition_type:
                continue
            if not f.base_features:
                continue
            if len(set(f.base_features) & set(e.base_features)) > 0.5 * len(f.base_features):
                similar += 1
        return 1.0 / (1.0 + similar)

    def _idbd_adapt_all(self):
        for f in self.features.values():
            if f.activation_count == 0:
                continue
            prev = f.combined_utility
            f.combined_utility = f.compute_combined_utility()
            delta = f.combined_utility - prev
            decay = 0.9
            f.gradient_trace = decay * f.gradient_trace + delta
            f.hessian_trace = decay * f.hessian_trace + abs(delta)
            if f.hessian_trace > 1e-8:
                upd = f.meta_learning_rate * f.gradient_trace * delta / f.hessian_trace
                f.learning_rate *= float(np.exp(np.clip(upd, -0.1, 0.1)))
                f.learning_rate = float(np.clip(f.learning_rate, 0.0001, 1.0))

    async def handle_option_feedback(self, event: dict[str, Any]):
        feats = event.get("features", []) or []
        success = bool(event.get("success", False))
        for fid in feats:
            f = self.features.get(fid)
            if not f:
                continue
            tgt = 1.0 if success else 0.0
            delta = tgt - f.play_utility
            f.play_utility = float(np.clip(f.play_utility + f.learning_rate * delta, 0.0, 1.0))
            f.activation_count += 1
            f.last_activated = datetime.now(UTC)
            await self.emit_event(
                "feature_utility_update",
                feature_id=fid,
                signal_type="play",
                value=f.play_utility,
                components={
                    "play": f.play_utility,
                    "prediction": f.prediction_utility,
                    "planning": f.planning_utility,
                    "novelty": f.novelty_score,
                },
                timestamp=datetime.now(UTC),
            )

    async def handle_prediction_feedback(self, event: dict[str, Any]):
        feats = event.get("features", []) or []
        error = float(event.get("error", 1.0))
        for fid in feats:
            f = self.features.get(fid)
            if not f:
                continue
            tgt = 1.0 - min(1.0, error)
            delta = tgt - f.prediction_utility
            f.prediction_utility = float(np.clip(f.prediction_utility + f.learning_rate * delta, 0.0, 1.0))
            await self.emit_event(
                "feature_utility_update",
                feature_id=fid,
                signal_type="prediction",
                value=f.prediction_utility,
                components={
                    "play": f.play_utility,
                    "prediction": f.prediction_utility,
                    "planning": f.planning_utility,
                    "novelty": f.novelty_score,
                },
                timestamp=datetime.now(UTC),
            )

    async def handle_planning_feedback(self, event: dict[str, Any]):
        feats = event.get("features", []) or []
        val = float(event.get("value", 0.0))
        tgt = 1.0 / (1.0 + np.exp(-val))
        for fid in feats:
            f = self.features.get(fid)
            if not f:
                continue
            delta = tgt - f.planning_utility
            f.planning_utility = float(np.clip(f.planning_utility + f.learning_rate * delta, 0.0, 1.0))
            await self.emit_event(
                "feature_utility_update",
                feature_id=fid,
                signal_type="planning",
                value=f.planning_utility,
                components={
                    "play": f.play_utility,
                    "prediction": f.prediction_utility,
                    "planning": f.planning_utility,
                    "novelty": f.novelty_score,
                },
                timestamp=datetime.now(UTC),
            )
