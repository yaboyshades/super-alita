import pytest
from src.orchestration.cortex_weaning import CortexWeaningOrchestrator, TrainingPhase

import pytest
from src.orchestration.cortex_weaning import CortexWeaningOrchestrator, TrainingPhase


@pytest.mark.asyncio
async def test_should_use_cortex_thresholds_and_phase_lift():
    o = CortexWeaningOrchestrator()
    assert await o.should_use_cortex(0.2, {}) is True
    assert await o.should_use_cortex(0.9, {}) is False
    assert await o.advance_phase_if_ready(0.36) is False
    assert await o.advance_phase_if_ready(0.37) is False
    assert await o.advance_phase_if_ready(0.38) is True
    assert o.current_phase == TrainingPhase.GUIDED


@pytest.mark.asyncio
async def test_phase_advance_multiple():
    o = CortexWeaningOrchestrator()
    await o.advance_phase_if_ready(0.35)
    await o.advance_phase_if_ready(0.36)
    await o.advance_phase_if_ready(0.37)
    await o.advance_phase_if_ready(0.56)
    await o.advance_phase_if_ready(0.57)
    await o.advance_phase_if_ready(0.58)
    assert o.current_phase == TrainingPhase.SUPERVISED


@pytest.mark.asyncio
async def test_phase_demotion_on_sustained_drop():
    o = CortexWeaningOrchestrator()
    await o.advance_phase_if_ready(0.35)
    await o.advance_phase_if_ready(0.36)
    await o.advance_phase_if_ready(0.37)
    await o.advance_phase_if_ready(0.56)
    await o.advance_phase_if_ready(0.57)
    await o.advance_phase_if_ready(0.58)
    assert o.current_phase == TrainingPhase.SUPERVISED
    o.window_scores.clear()
    o._push_score(0.40)
    o._push_score(0.40)
    o._push_score(0.40)
    demoted = await o.maybe_demote_phase()
    assert demoted is True
    assert o.current_phase == TrainingPhase.GUIDED
