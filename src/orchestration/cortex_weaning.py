from __future__ import annotations

"""Basic orchestrator for determining cortex usage phases."""

from enum import Enum
from typing import Any, Dict


class WeaningPhase(Enum):
    FULL = "full"
    PARTIAL = "partial"
    AUTONOMOUS = "autonomous"


class CortexWeaningOrchestrator:
    def __init__(self) -> None:
        self.current_phase = WeaningPhase.FULL

    async def advance_phase_if_ready(self, autonomy_score: float) -> bool:
        if self.current_phase == WeaningPhase.FULL and autonomy_score > 0.8:
            self.current_phase = WeaningPhase.PARTIAL
            return True
        if self.current_phase == WeaningPhase.PARTIAL and autonomy_score > 0.95:
            self.current_phase = WeaningPhase.AUTONOMOUS
            return True
        return False

    async def should_use_cortex(self, confidence: float, context: Dict[str, Any]) -> bool:
        if self.current_phase == WeaningPhase.AUTONOMOUS:
            return confidence < 0.2
        return True
