import asyncio

import pytest

from src.unified_intelligence.orchestrator import (
    FusionConfig,
    HardenedOrchestrator,
    Request,
)


@pytest.fixture
def orchestrator() -> HardenedOrchestrator:
    return HardenedOrchestrator()


def test_orchestrator_initialization(
    orchestrator: HardenedOrchestrator,
) -> None:
    assert orchestrator.config is not None
    assert hasattr(orchestrator, "_calculate_weights")
    assert hasattr(orchestrator, "_fuse_and_decide")


def test_fusion_config_defaults() -> None:
    config = FusionConfig()
    assert (
        pytest.approx(
            config.mangle_base
            + config.constitution_base
            + config.workflow_base,
            rel=1e-6,
        )
        == 1.0
    )


def test_orchestrator_run_smoke(orchestrator: HardenedOrchestrator) -> None:
    request = Request(
        request_id="test-001",
        ts="2025-01-01T00:00:00Z",
        intent_text="Provide a quick status summary",
        code_refs=[],
        context={},
    )

    async def run() -> None:
        advice = await orchestrator.orchestrate(request)
        assert advice is not None
        assert advice.decision

    asyncio.get_event_loop().run_until_complete(run())
