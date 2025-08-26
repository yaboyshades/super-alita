from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List


class TrainingPhase(Enum):
    BOOTSTRAP = "bootstrap"
    GUIDED = "guided"
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"


@dataclass
class PhaseConfig:
    cortex_intervention_threshold: float
    self_attempt_first: bool
    cortex_confidence_requirement: float
    knowledge_validation_required: bool


class CortexWeaningOrchestrator:
    """Manages progressive reduction of cortex dependency."""

    def __init__(self) -> None:
        self.current_phase = TrainingPhase.BOOTSTRAP
        self.window_scores: List[float] = []
        self.window_size = 20
        self.hysteresis = 0.05
        self.min_sustain = 3
        self.phase_configs: Dict[TrainingPhase, PhaseConfig] = {
            TrainingPhase.BOOTSTRAP: PhaseConfig(
                cortex_intervention_threshold=0.8,
                self_attempt_first=False,
                cortex_confidence_requirement=0.6,
                knowledge_validation_required=False,
            ),
            TrainingPhase.GUIDED: PhaseConfig(
                cortex_intervention_threshold=0.5,
                self_attempt_first=True,
                cortex_confidence_requirement=0.7,
                knowledge_validation_required=True,
            ),
            TrainingPhase.SUPERVISED: PhaseConfig(
                cortex_intervention_threshold=0.3,
                self_attempt_first=True,
                cortex_confidence_requirement=0.8,
                knowledge_validation_required=True,
            ),
            TrainingPhase.AUTONOMOUS: PhaseConfig(
                cortex_intervention_threshold=0.1,
                self_attempt_first=True,
                cortex_confidence_requirement=0.9,
                knowledge_validation_required=True,
            ),
        }

    async def should_use_cortex(self, confidence: float, context: Dict[str, Any]) -> bool:
        cfg = self.phase_configs[self.current_phase]
        return confidence < cfg.cortex_intervention_threshold

    async def advance_phase_if_ready(self, autonomy_score: float) -> bool:
        self._push_score(autonomy_score)
        thresholds = {
            TrainingPhase.BOOTSTRAP: 0.30,
            TrainingPhase.GUIDED: 0.50,
            TrainingPhase.SUPERVISED: 0.70,
            TrainingPhase.AUTONOMOUS: 0.85,
        }
        t_current = thresholds.get(self.current_phase, 1.0)
        if self._sustained_at_or_above(t_current + self.hysteresis):
            phases = list(TrainingPhase)
            idx = phases.index(self.current_phase)
            if idx < len(phases) - 1:
                self.current_phase = phases[idx + 1]
                return True
        return False

    async def maybe_demote_phase(self) -> bool:
        thresholds = {
            TrainingPhase.BOOTSTRAP: 0.0,
            TrainingPhase.GUIDED: 0.30,
            TrainingPhase.SUPERVISED: 0.50,
            TrainingPhase.AUTONOMOUS: 0.70,
        }
        phases = list(TrainingPhase)
        idx = phases.index(self.current_phase)
        if idx == 0:
            return False
        lower_phase = phases[idx - 1]
        t_lower = thresholds[self.current_phase]
        if self._sustained_below(t_lower - self.hysteresis):
            self.current_phase = lower_phase
            return True
        return False

    def _push_score(self, s: float) -> None:
        self.window_scores.append(float(s))
        if len(self.window_scores) > self.window_size:
            self.window_scores = self.window_scores[-self.window_size :]

    def _sustained_at_or_above(self, threshold: float) -> bool:
        if len(self.window_scores) < self.min_sustain:
            return False
        return all(x >= threshold for x in self.window_scores[-self.min_sustain :])

    def _sustained_below(self, threshold: float) -> bool:
        if len(self.window_scores) < self.min_sustain:
            return False
        return all(x < threshold for x in self.window_scores[-self.min_sustain :])
